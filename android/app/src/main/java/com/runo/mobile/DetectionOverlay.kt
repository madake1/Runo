// Draws model detections over CameraX preview in matching screen coordinates.
package com.runo.mobile

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.view.View
import kotlin.math.max

/** Displays detection rectangles and labels above the camera preview. */
class DetectionOverlay(context: Context) : View(context) {
    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.RED; style = Paint.Style.STROKE; strokeWidth = 4f
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.RED; textSize = 38f
    }
    private var results = emptyList<Detection>()
    private var sourceWidth = 1
    private var sourceHeight = 1

    /** Stores results and source size, then requests a redraw. */
    fun update(detections: List<Detection>, imageWidth: Int, imageHeight: Int) {
        results = detections
        sourceWidth = imageWidth
        sourceHeight = imageHeight
        invalidate()
    }

    /** Maps image boxes through the preview crop and paints them on screen. */
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        // PreviewView uses FILL_CENTER; map the upright analysis frame likewise.
        val scale = max(width.toFloat() / sourceWidth, height.toFloat() / sourceHeight)
        val dx = (width - sourceWidth * scale) / 2
        val dy = (height - sourceHeight * scale) / 2
        for (item in results) {
            val left = dx + item.left * scale
            val top = dy + item.top * scale
            canvas.drawRect(left, top, dx + item.right * scale, dy + item.bottom * scale, boxPaint)
            canvas.drawText("${item.label} ${"%.2f".format(item.score)}", left,
                max(40f, top - 8f), textPaint)
        }
    }
}
