package com.wellness.engine.network

import io.ktor.client.*
import io.ktor.client.call.*
import io.ktor.client.engine.android.*
import io.ktor.client.plugins.contentnegotiation.*
import io.ktor.client.plugins.logging.*
import io.ktor.client.request.*
import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import kotlinx.serialization.json.Json

/**
 * HTTP client for communicating with the Flask wellness backend.
 *
 * Sends session data (screenshots + metadata) to the backend for
 * real ML processing (CLIP, toxic-bert, sentiment analysis).
 * Receives structured WellnessReport.
 *
 * @param baseUrl The Railway-deployed backend URL
 * @param requestTimeoutMs Timeout for the analyze request (ML processing can take time)
 */
class WellnessApiClient(
    private val baseUrl: String,
    private val requestTimeoutMs: Long = 60_000L
) {
    private val client = HttpClient(Android) {
        install(ContentNegotiation) {
            json(Json {
                ignoreUnknownKeys = true
                isLenient = true
                encodeDefaults = true
            })
        }
        install(Logging) {
            level = LogLevel.NONE  // Never log request bodies (privacy)
        }
        engine {
            connectTimeout = 15_000
            socketTimeout = requestTimeoutMs.toInt()
        }
    }

    /**
     * Send session data to the backend for ML analysis.
     *
     * @param request The analyze request with base64 screenshots and metadata
     * @return AnalyzeResponse with all computed wellness scores
     * @throws Exception on network failure or server error
     */
    suspend fun analyze(request: AnalyzeRequest): AnalyzeResponse {
        return client.post("$baseUrl/api/analyze") {
            contentType(ContentType.Application.Json)
            setBody(request)
        }.body()
    }

    /**
     * Health check — verify backend is reachable.
     */
    suspend fun healthCheck(): Boolean {
        return try {
            val response = client.get("$baseUrl/health")
            response.status == HttpStatusCode.OK
        } catch (_: Exception) {
            false
        }
    }

    fun close() {
        client.close()
    }
}
