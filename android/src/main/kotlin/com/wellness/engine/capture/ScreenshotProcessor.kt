package com.wellness.engine.capture

import android.graphics.Bitmap
import android.util.Base64
import java.io.ByteArrayOutputStream

/**
 * Phase 2 — Screenshot Processor
 *
 * Converts captured Bitmap to base64 for backend transmission.
 * Immediately recycles the bitmap after conversion — never persists raw image.
 *
 * Privacy guarantees:
 * - Bitmap is converted to base64 in-memory
 * - Original bitmap is recycled immediately
 * - Base64 string is only held until HTTP request completes
 * - No disk writes ever occur
 */
class ScreenshotProcessor {

    companion object {
        /** JPEG quality for base64 encoding (balance quality vs payload size) */
        private const val JPEG_QUALITY = 75

        /** Maximum dimension — downscale large screenshots to save bandwidth */
        private const val MAX_DIMENSION = 720
    }

    /**
     * Process a screenshot: downscale → encode to base64 → recycle bitmap.
     *
     * @param bitmap The captured screenshot
     * @return Base64-encoded JPEG string, or null if processing fails
     */
    fun processToBase64(bitmap: Bitmap): String? {
        return try {
            val scaled = downscaleIfNeeded(bitmap)
            val base64 = encodeToBase64(scaled)

            // Recycle bitmaps immediately — privacy critical
            if (scaled !== bitmap) {
                scaled.recycle()
            }
            bitmap.recycle()

            base64
        } catch (e: Exception) {
            bitmap.recycle()
            null
        }
    }

    /**
     * Downscale bitmap if either dimension exceeds MAX_DIMENSION.
     * Maintains aspect ratio.
     */
    private fun downscaleIfNeeded(bitmap: Bitmap): Bitmap {
        val maxDim = maxOf(bitmap.width, bitmap.height)
        if (maxDim <= MAX_DIMENSION) return bitmap

        val scale = MAX_DIMENSION.toFloat() / maxDim
        val newWidth = (bitmap.width * scale).toInt()
        val newHeight = (bitmap.height * scale).toInt()

        return Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true)
    }

    /**
     * Encode bitmap to base64 JPEG string.
     */
    private fun encodeToBase64(bitmap: Bitmap): String {
        val stream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, stream)
        val bytes = stream.toByteArray()
        stream.close()
        return Base64.encodeToString(bytes, Base64.NO_WRAP)
    }
}
