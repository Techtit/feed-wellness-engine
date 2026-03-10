package com.wellness.engine.model

import kotlinx.serialization.Serializable

/**
 * Full session data submitted to the backend for analysis.
 * Contains all gated post interactions and circadian metadata.
 */
@Serializable
data class SessionData(
    val userId: String,
    val sessionId: String,
    val posts: List<PostPayload>,
    val sessionStartUtc: String,
    val sessionEndUtc: String,
    val circadianScore: Float,
    val nightSessionDurationMinutes: Float
)

/**
 * Individual post payload sent to the backend.
 * Contains the base64-encoded screenshot and behavioral metadata.
 */
@Serializable
data class PostPayload(
    val postId: String,
    val imageBase64: String,
    val dwellTimeSeconds: Float,
    val timestampUtc: String
)
