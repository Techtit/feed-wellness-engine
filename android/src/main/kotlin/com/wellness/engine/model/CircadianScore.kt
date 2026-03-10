package com.wellness.engine.model

/**
 * Circadian disruption analysis result.
 *
 * nightIntensity: fraction of total dwell time spent in night window [23:00–04:00]
 * nightSessionDurationMinutes: total minutes of session time within night window
 * disruptionScore: final normalized circadian disruption score ∈ [0, 1]
 */
data class CircadianScore(
    val nightIntensity: Float,
    val nightSessionDurationMinutes: Float,
    val disruptionScore: Float
) {
    init {
        require(disruptionScore in 0f..1f) {
            "Disruption score must be in [0, 1], got $disruptionScore"
        }
    }
}
