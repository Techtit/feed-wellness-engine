package com.wellness.engine.network

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Response from the Flask backend /api/analyze endpoint.
 * Maps directly to WellnessReport.
 */
@Serializable
data class AnalyzeResponse(
    @SerialName("session_id")
    val sessionId: String,

    @SerialName("toxicity_score")
    val toxicityScore: Float,

    @SerialName("addictiveness_score")
    val addictivenessScore: Float,

    @SerialName("positivity_score")
    val positivityScore: Float,

    @SerialName("emotional_intensity")
    val emotionalIntensity: Float,

    @SerialName("circadian_score")
    val circadianScore: Float,

    @SerialName("wellness_index")
    val wellnessIndex: Float,

    val recommendation: String
)
