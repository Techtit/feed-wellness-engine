package com.wellness.engine.network

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Request body sent to the Flask backend /api/analyze endpoint.
 */
@Serializable
data class AnalyzeRequest(
    @SerialName("user_id")
    val userId: String,

    @SerialName("session_id")
    val sessionId: String,

    val posts: List<PostPayloadDto>,

    @SerialName("session_start_utc")
    val sessionStartUtc: String,

    @SerialName("session_end_utc")
    val sessionEndUtc: String,

    @SerialName("circadian_score")
    val circadianScore: Float,

    @SerialName("night_session_duration_minutes")
    val nightSessionDurationMinutes: Float
)

@Serializable
data class PostPayloadDto(
    @SerialName("post_id")
    val postId: String,

    @SerialName("image_base64")
    val imageBase64: String,

    @SerialName("dwell_time_seconds")
    val dwellTimeSeconds: Float,

    @SerialName("timestamp_utc")
    val timestampUtc: String
)
