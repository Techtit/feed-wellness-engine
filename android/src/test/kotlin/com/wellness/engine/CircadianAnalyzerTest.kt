package com.wellness.engine

import com.wellness.engine.circadian.CircadianAnalyzer
import com.wellness.engine.model.PostInteraction
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import java.time.ZoneOffset

/**
 * Unit tests for CircadianAnalyzer.
 * Tests night window detection, intensity computation, and disruption scoring.
 */
class CircadianAnalyzerTest {

    private lateinit var analyzer: CircadianAnalyzer

    @Before
    fun setUp() {
        analyzer = CircadianAnalyzer()
    }

    @Test
    fun `midnight is a night hour`() {
        assertTrue(analyzer.isNightHour(0))
    }

    @Test
    fun `23 is a night hour`() {
        assertTrue(analyzer.isNightHour(23))
    }

    @Test
    fun `4 AM is a night hour`() {
        assertTrue(analyzer.isNightHour(4))
    }

    @Test
    fun `noon is not a night hour`() {
        assertFalse(analyzer.isNightHour(12))
    }

    @Test
    fun `5 AM is not a night hour`() {
        assertFalse(analyzer.isNightHour(5))
    }

    @Test
    fun `22 is not a night hour`() {
        assertFalse(analyzer.isNightHour(22))
    }

    @Test
    fun `empty interactions produce zero disruption`() {
        val score = analyzer.analyze(
            interactions = emptyList(),
            sessionStartMillis = 0L,
            sessionEndMillis = 1000L,
            timeZone = ZoneOffset.UTC
        )
        assertEquals(0f, score.disruptionScore, 0.001f)
        assertEquals(0f, score.nightIntensity, 0.001f)
    }

    @Test
    fun `all night usage produces high disruption`() {
        // Create interactions at 1 AM UTC
        val oneAmUtc = 3600_000L  // 1 hour after epoch = 1 AM UTC
        val interactions = listOf(
            PostInteraction(
                postId = "p1",
                entryTimeMillis = oneAmUtc,
                exitTimeMillis = oneAmUtc + 10_000L,
                timestampUtc = "1970-01-01T01:00:00Z"
            )
        )

        val score = analyzer.analyze(
            interactions = interactions,
            sessionStartMillis = oneAmUtc,
            sessionEndMillis = oneAmUtc + 10_000L,
            timeZone = ZoneOffset.UTC
        )

        assertEquals(1.0f, score.nightIntensity, 0.001f)
        assertTrue(score.disruptionScore > 0.5f)
    }

    @Test
    fun `daytime usage produces zero night intensity`() {
        // Create interactions at noon UTC
        val noonUtc = 12 * 3600_000L
        val interactions = listOf(
            PostInteraction(
                postId = "p1",
                entryTimeMillis = noonUtc,
                exitTimeMillis = noonUtc + 5_000L,
                timestampUtc = "1970-01-01T12:00:00Z"
            )
        )

        val score = analyzer.analyze(
            interactions = interactions,
            sessionStartMillis = noonUtc,
            sessionEndMillis = noonUtc + 5_000L,
            timeZone = ZoneOffset.UTC
        )

        assertEquals(0f, score.nightIntensity, 0.001f)
        assertEquals(0f, score.disruptionScore, 0.001f)
    }

    @Test
    fun `disruption score is clamped to 0 to 1`() {
        val score = analyzer.analyze(
            interactions = emptyList(),
            sessionStartMillis = 0L,
            sessionEndMillis = 1L,
            timeZone = ZoneOffset.UTC
        )
        assertTrue(score.disruptionScore in 0f..1f)
    }
}
