package com.wellness.engine.model

/**
 * Represents a single tracked post interaction.
 * Created by AttentionTracker when a post becomes visible and then hidden.
 */
data class PostInteraction(
    val postId: String,
    val entryTimeMillis: Long,
    val exitTimeMillis: Long,
    val timestampUtc: String
) {
    /** Dwell time in seconds */
    val dwellTimeSeconds: Float
        get() = (exitTimeMillis - entryTimeMillis) / 1000f

    /** Whether this post passes the attention gate (dwell > 3s) */
    val passesGate: Boolean
        get() = dwellTimeSeconds > DWELL_THRESHOLD_SECONDS

    companion object {
        const val DWELL_THRESHOLD_SECONDS = 3.0f
    }
}
