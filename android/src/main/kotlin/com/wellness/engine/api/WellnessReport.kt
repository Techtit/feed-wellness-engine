package com.wellness.engine.api

import kotlinx.serialization.Serializable

/**
 * Final output of the Feed Wellness Engine.
 *
 * All scores are normalized to [0, 1] except wellnessIndex which is [0, 100].
 * This is the single structured output the launcher consumes.
 */
@Serializable
data class WellnessReport(
    val sessionId: String,
    val toxicityScore: Float,
    val addictivenessScore: Float,
    val positivityScore: Float,
    val emotionalIntensity: Float,
    val circadianScore: Float,
    val wellnessIndex: Float,
    val recommendation: String
) {
    /** Parsed recommendation enum */
    val recommendationEnum: WellnessRecommendation
        get() = try {
            WellnessRecommendation.valueOf(recommendation)
        } catch (_: IllegalArgumentException) {
            WellnessRecommendation.NO_ACTION
        }

    /** Human-readable wellness grade */
    val grade: String
        get() = when {
            wellnessIndex >= 80 -> "Excellent"
            wellnessIndex >= 60 -> "Good"
            wellnessIndex >= 40 -> "Fair"
            wellnessIndex >= 20 -> "Poor"
            else -> "Critical"
        }
}
