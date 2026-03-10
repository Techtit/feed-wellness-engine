package com.wellness.engine

import com.wellness.engine.attention.AttentionTracker
import com.wellness.engine.model.PostInteraction
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

/**
 * Unit tests for AttentionTracker.
 * Tests dwell-time gating, post tracking, and session management.
 */
class AttentionTrackerTest {

    private lateinit var tracker: AttentionTracker

    @Before
    fun setUp() {
        tracker = AttentionTracker()
    }

    @Test
    fun `post with dwell over 3 seconds passes gate`() {
        val interaction = PostInteraction(
            postId = "post_1",
            entryTimeMillis = 0L,
            exitTimeMillis = 4000L,
            timestampUtc = "2026-02-27T12:00:00Z"
        )
        assertTrue(interaction.passesGate)
        assertEquals(4.0f, interaction.dwellTimeSeconds, 0.01f)
    }

    @Test
    fun `post with dwell under 3 seconds is gated out`() {
        val interaction = PostInteraction(
            postId = "post_2",
            entryTimeMillis = 0L,
            exitTimeMillis = 2000L,
            timestampUtc = "2026-02-27T12:00:00Z"
        )
        assertFalse(interaction.passesGate)
    }

    @Test
    fun `post with exactly 3 seconds is gated out`() {
        val interaction = PostInteraction(
            postId = "post_3",
            entryTimeMillis = 0L,
            exitTimeMillis = 3000L,
            timestampUtc = "2026-02-27T12:00:00Z"
        )
        assertFalse(interaction.passesGate)  // > 3, not >= 3
    }

    @Test
    fun `getGatedInteractions returns only gated posts`() {
        // Simulate post interactions via the tracker
        tracker.onPostVisible("short_post")
        Thread.sleep(50)
        tracker.onPostHidden("short_post")

        val allInteractions = tracker.allInteractions
        assertEquals(1, allInteractions.size)

        // Short dwell — should not be in gated list
        val gated = tracker.getGatedInteractions()
        assertEquals(0, gated.size)
    }

    @Test
    fun `reset clears all state`() {
        tracker.onPostVisible("post_1")
        tracker.onPostHidden("post_1")
        assertEquals(1, tracker.allInteractions.size)

        tracker.reset()
        assertEquals(0, tracker.allInteractions.size)
    }

    @Test
    fun `hidden post without visible is ignored`() {
        tracker.onPostHidden("never_visible")
        assertEquals(0, tracker.allInteractions.size)
    }
}
