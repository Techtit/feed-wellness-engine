package com.wellness.engine.api

import kotlinx.serialization.Serializable

/**
 * Wellness recommendations the engine can produce.
 * The launcher consumes these to adjust behavior — this module never performs UI changes.
 */
enum class WellnessRecommendation {
    /** User shows high addictiveness + circadian disruption → suggest winding down */
    SUGGEST_SLEEP_MODE,

    /** Feed toxicity is elevated → reduce social feed priority in launcher */
    REDUCE_SOCIAL_PRIORITY,

    /** Feed wellness is within healthy bounds → no intervention needed */
    NO_ACTION
}
