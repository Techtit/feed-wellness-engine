package com.wellness.engine.circadian

import com.wellness.engine.model.CircadianScore
import com.wellness.engine.model.PostInteraction
import java.time.Instant
import java.time.ZoneId

/**
 * Phase 7 — Circadian Analysis Module
 *
 * Analyzes late-night usage patterns to compute a circadian disruption score.
 * Runs fully on-device — pure Kotlin, no ML, no network.
 *
 * Night window: 23:00–04:00 (configurable)
 *
 * Formulas:
 * - NightIntensity = Σ(dwellTime × L_i) / Σ(dwellTime)
 *   where L_i = 1 if hour ∈ [nightStartHour, nightEndHour], else 0
 * - CircadianDisruption = η₁ × NightIntensity + η₂ × norm(NightSessionDuration)
 *   clamped to [0, 1]
 */
class CircadianAnalyzer(
    private val nightStartHour: Int = 23,
    private val nightEndHour: Int = 4,
    private val eta1: Float = 0.6f,
    private val eta2: Float = 0.4f,
    private val maxNightSessionMinutes: Float = 120f
) {

    /**
     * Analyze a list of post interactions for circadian disruption.
     *
     * @param interactions All interactions in the session (including ungated)
     * @param sessionStartMillis Session start time in millis
     * @param sessionEndMillis Session end time in millis
     * @param timeZone User's local timezone for hour computation
     * @return CircadianScore with nightIntensity, nightDuration, and disruption score
     */
    fun analyze(
        interactions: List<PostInteraction>,
        sessionStartMillis: Long,
        sessionEndMillis: Long,
        timeZone: ZoneId = ZoneId.systemDefault()
    ): CircadianScore {
        if (interactions.isEmpty()) {
            return CircadianScore(
                nightIntensity = 0f,
                nightSessionDurationMinutes = 0f,
                disruptionScore = 0f
            )
        }

        var nightDwell = 0f
        var totalDwell = 0f

        for (interaction in interactions) {
            val dwell = interaction.dwellTimeSeconds
            totalDwell += dwell

            val hour = Instant.ofEpochMilli(interaction.entryTimeMillis)
                .atZone(timeZone)
                .hour

            if (isNightHour(hour)) {
                nightDwell += dwell
            }
        }

        // Night intensity: fraction of total dwell that occurred at night
        val nightIntensity = if (totalDwell > 0) nightDwell / totalDwell else 0f

        // Night session duration: how many minutes of the session fell in night window
        val nightSessionMinutes = computeNightSessionDuration(
            sessionStartMillis, sessionEndMillis, timeZone
        )

        // Normalized night session duration
        val normalizedNightDuration = (nightSessionMinutes / maxNightSessionMinutes)
            .coerceIn(0f, 1f)

        // Circadian disruption score
        val disruption = (eta1 * nightIntensity + eta2 * normalizedNightDuration)
            .coerceIn(0f, 1f)

        return CircadianScore(
            nightIntensity = nightIntensity,
            nightSessionDurationMinutes = nightSessionMinutes,
            disruptionScore = disruption
        )
    }

    /**
     * Check if a given hour falls in the night window.
     * Handles wrap-around: [23, 0, 1, 2, 3, 4]
     */
    internal fun isNightHour(hour: Int): Boolean {
        return if (nightStartHour > nightEndHour) {
            // Wraps around midnight: e.g., 23-4 means 23,0,1,2,3,4
            hour >= nightStartHour || hour <= nightEndHour
        } else {
            hour in nightStartHour..nightEndHour
        }
    }

    /**
     * Compute how many minutes of the session window overlapped with night hours.
     */
    private fun computeNightSessionDuration(
        startMillis: Long,
        endMillis: Long,
        timeZone: ZoneId
    ): Float {
        val startHour = Instant.ofEpochMilli(startMillis).atZone(timeZone).hour
        val endHour = Instant.ofEpochMilli(endMillis).atZone(timeZone).hour
        val totalMinutes = (endMillis - startMillis) / 60000f

        // Simplified: if session start or end is in night window, estimate overlap
        val startIsNight = isNightHour(startHour)
        val endIsNight = isNightHour(endHour)

        return when {
            startIsNight && endIsNight -> totalMinutes
            startIsNight || endIsNight -> totalMinutes * 0.5f
            else -> 0f
        }
    }
}
