package com.wellness.engine.api

import com.wellness.engine.attention.AttentionTracker
import com.wellness.engine.capture.ScreenshotProcessor
import com.wellness.engine.capture.ScreenshotProvider
import com.wellness.engine.circadian.CircadianAnalyzer
import com.wellness.engine.model.PostInteraction
import com.wellness.engine.network.AnalyzeRequest
import com.wellness.engine.network.PostPayloadDto
import com.wellness.engine.network.WellnessApiClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.withContext
import java.time.Instant
import java.time.ZoneId
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter
import java.util.UUID

/**
 * Feed Wellness Engine — Single Entry Point
 *
 * Orchestrates the full pipeline:
 * 1. Collect gated interactions from AttentionTracker
 * 2. Capture screenshots for gated posts via ScreenshotProvider
 * 3. Process screenshots to base64 via ScreenshotProcessor
 * 4. Compute circadian score on-device via CircadianAnalyzer
 * 5. Send everything to Flask backend for ML analysis
 * 6. Return structured WellnessReport
 *
 * Usage:
 * ```kotlin
 * val engine = FeedWellnessEngine(
 *     screenshotProvider = myScreenshotProvider,
 *     backendUrl = "https://your-app.up.railway.app"
 * )
 *
 * // During session: feed visibility events
 * engine.tracker.onPostVisible("post_1")
 * engine.tracker.onPostHidden("post_1")
 *
 * // At session end: analyze
 * val report = engine.analyzeSession(userId = "user_123")
 * println(report.wellnessIndex)  // 0-100
 * println(report.recommendation) // SUGGEST_SLEEP_MODE, etc.
 * ```
 *
 * Privacy guarantees:
 * - Screenshots are never persisted to disk
 * - Only aggregated scores are stored in Supabase
 * - Raw images are deleted after embedding extraction on the backend
 * - HTTP logging is disabled (request bodies never logged)
 */
class FeedWellnessEngine(
    private val screenshotProvider: ScreenshotProvider,
    backendUrl: String,
    private val userId: String = "default_user",
    private val timeZone: ZoneId = ZoneId.systemDefault(),
    private val maxPostsPerSession: Int = 30
) {
    /** Phase 1 — Attention tracker. Launcher feeds visibility events here. */
    val tracker = AttentionTracker()

    /** Phase 2 — Screenshot processor (bitmap → base64) */
    private val screenshotProcessor = ScreenshotProcessor()

    /** Phase 7 — Circadian analyzer (on-device, pure Kotlin) */
    private val circadianAnalyzer = CircadianAnalyzer()

    /** Network client for Flask backend */
    private val apiClient = WellnessApiClient(baseUrl = backendUrl)

    /** Session tracking */
    private var sessionStartMillis: Long = System.currentTimeMillis()
    private var sessionId: String = UUID.randomUUID().toString()

    /**
     * Start a new tracking session.
     * Call this when the user opens a feed.
     */
    fun startSession() {
        tracker.reset()
        sessionStartMillis = System.currentTimeMillis()
        sessionId = UUID.randomUUID().toString()
    }

    /**
     * Analyze the current session.
     * Call this when the user closes the feed or after a timeout.
     *
     * Pipeline:
     * 1. Get gated interactions (dwell > 3s)
     * 2. Capture & encode screenshots in parallel
     * 3. Compute circadian score on-device
     * 4. Send to backend for ML analysis
     * 5. Return WellnessReport
     *
     * @return WellnessReport with all scores and recommendation
     */
    suspend fun analyzeSession(): WellnessReport = withContext(Dispatchers.Default) {
        val sessionEndMillis = System.currentTimeMillis()
        val allInteractions = tracker.allInteractions
        val gatedInteractions = tracker.getGatedInteractions()
            .take(maxPostsPerSession)

        // Phase 7: Circadian analysis (on-device)
        val circadianScore = circadianAnalyzer.analyze(
            interactions = allInteractions,
            sessionStartMillis = sessionStartMillis,
            sessionEndMillis = sessionEndMillis,
            timeZone = timeZone
        )

        // Phase 2: Capture and process screenshots in parallel
        val postPayloads = captureScreenshots(gatedInteractions)

        // If no screenshots captured, return neutral report
        if (postPayloads.isEmpty()) {
            return@withContext WellnessReport(
                sessionId = sessionId,
                toxicityScore = 0f,
                addictivenessScore = 0f,
                positivityScore = 0.5f,
                emotionalIntensity = 0.5f,
                circadianScore = circadianScore.disruptionScore,
                wellnessIndex = 50f,
                recommendation = WellnessRecommendation.NO_ACTION.name
            )
        }

        // Build request for backend
        val request = AnalyzeRequest(
            userId = userId,
            sessionId = sessionId,
            posts = postPayloads,
            sessionStartUtc = formatUtc(sessionStartMillis),
            sessionEndUtc = formatUtc(sessionEndMillis),
            circadianScore = circadianScore.disruptionScore,
            nightSessionDurationMinutes = circadianScore.nightSessionDurationMinutes
        )

        // Send to Flask backend for ML analysis
        val response = apiClient.analyze(request)

        // Map response to WellnessReport
        WellnessReport(
            sessionId = response.sessionId,
            toxicityScore = response.toxicityScore,
            addictivenessScore = response.addictivenessScore,
            positivityScore = response.positivityScore,
            emotionalIntensity = response.emotionalIntensity,
            circadianScore = response.circadianScore,
            wellnessIndex = response.wellnessIndex,
            recommendation = response.recommendation
        )
    }

    /**
     * Check if the backend is reachable.
     */
    suspend fun isBackendAvailable(): Boolean = apiClient.healthCheck()

    /**
     * Clean up resources.
     */
    fun destroy() {
        tracker.reset()
        apiClient.close()
    }

    /**
     * Capture screenshots for gated interactions in parallel.
     * Each bitmap is immediately converted to base64 and the bitmap is recycled.
     */
    private suspend fun captureScreenshots(
        interactions: List<PostInteraction>
    ): List<PostPayloadDto> = coroutineScope {
        interactions.map { interaction ->
            async(Dispatchers.IO) {
                val bitmap = screenshotProvider.capture(interaction.postId)
                    ?: return@async null

                val base64 = screenshotProcessor.processToBase64(bitmap)
                    ?: return@async null

                PostPayloadDto(
                    postId = interaction.postId,
                    imageBase64 = base64,
                    dwellTimeSeconds = interaction.dwellTimeSeconds,
                    timestampUtc = interaction.timestampUtc
                )
            }
        }.awaitAll().filterNotNull()
    }

    private fun formatUtc(millis: Long): String {
        return Instant.ofEpochMilli(millis)
            .atOffset(ZoneOffset.UTC)
            .format(DateTimeFormatter.ISO_INSTANT)
    }
}
