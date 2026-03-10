package com.wellness.engine.attention

import com.wellness.engine.model.PostInteraction
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import java.time.Instant
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter
import java.util.concurrent.ConcurrentHashMap

/**
 * Phase 1 — Attention Gating Module
 *
 * Tracks post visibility events and computes dwell time.
 * Emits [PostInteraction] via SharedFlow only when dwell time exceeds the 3-second gate.
 *
 * This class is:
 * - UI-independent (receives raw visibility callbacks)
 * - ML-free (pure time computation)
 * - Thread-safe (ConcurrentHashMap for tracking)
 *
 * Usage:
 * ```
 * val tracker = AttentionTracker()
 * // In scroll listener:
 * tracker.onPostVisible("post_123")
 * // When post scrolls away:
 * tracker.onPostHidden("post_123")
 * // Collect gated interactions:
 * tracker.gatedInteractions.collect { interaction -> ... }
 * ```
 */
class AttentionTracker {

    companion object {
        /** Posts visible for less than this are discarded */
        const val DWELL_THRESHOLD_SECONDS = 3.0f

        /** Maximum posts tracked per session to bound memory */
        const val MAX_TRACKED_POSTS = 200
    }

    /** Currently visible posts: postId → entry timestamp millis */
    private val visiblePosts = ConcurrentHashMap<String, Long>()

    /** All completed interactions this session (gated + ungated for circadian) */
    private val _allInteractions = mutableListOf<PostInteraction>()
    val allInteractions: List<PostInteraction>
        get() = _allInteractions.toList()

    /** Flow of interactions that pass the dwell gate (> 3s) */
    private val _gatedInteractions = MutableSharedFlow<PostInteraction>(
        replay = 0,
        extraBufferCapacity = 64
    )
    val gatedInteractions: SharedFlow<PostInteraction> = _gatedInteractions.asSharedFlow()

    /**
     * Call when a post becomes visible on screen.
     * Records the entry timestamp for dwell computation.
     */
    fun onPostVisible(postId: String) {
        if (visiblePosts.size >= MAX_TRACKED_POSTS) return
        visiblePosts[postId] = System.currentTimeMillis()
    }

    /**
     * Call when a post leaves the screen.
     * Computes dwell time and emits to gatedInteractions if > 3s.
     */
    fun onPostHidden(postId: String) {
        val entryTime = visiblePosts.remove(postId) ?: return
        val exitTime = System.currentTimeMillis()

        val interaction = PostInteraction(
            postId = postId,
            entryTimeMillis = entryTime,
            exitTimeMillis = exitTime,
            timestampUtc = Instant.ofEpochMilli(entryTime)
                .atOffset(ZoneOffset.UTC)
                .format(DateTimeFormatter.ISO_INSTANT)
        )

        _allInteractions.add(interaction)

        // Only emit if passes the attention gate
        if (interaction.passesGate) {
            _gatedInteractions.tryEmit(interaction)
        }
    }

    /**
     * Returns all interactions that passed the gate.
     * Used at session end to build the analysis payload.
     */
    fun getGatedInteractions(): List<PostInteraction> {
        return _allInteractions.filter { it.passesGate }
    }

    /** Reset tracker state for a new session */
    fun reset() {
        visiblePosts.clear()
        _allInteractions.clear()
    }
}
