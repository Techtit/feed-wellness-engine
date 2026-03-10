package com.wellness.engine.capture

import android.graphics.Bitmap

/**
 * Phase 2 — Screenshot Provider Interface
 *
 * Abstraction for capturing screenshots. The launcher injects its own
 * implementation using PixelCopy, View.drawToBitmap(), or MediaProjection.
 *
 * This module NEVER captures screenshots itself — it only defines what it needs.
 * The launcher provides the "how".
 *
 * Privacy contract:
 * - The returned Bitmap is processed in-memory only
 * - It is never written to disk
 * - It is nullified immediately after embedding extraction
 */
interface ScreenshotProvider {
    /**
     * Capture a screenshot of the currently visible post content.
     *
     * @param postId Identifier of the post to capture
     * @return Bitmap of the post content, or null if capture fails
     */
    suspend fun capture(postId: String): Bitmap?
}
