// Local YOLO/ONNX inference: prepare pixels, decode detections, and apply NMS.
package com.runo.mobile

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONObject
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

/** One predicted box in upright camera-image coordinates. */
data class Detection(val left: Float, val top: Float, val right: Float,
                     val bottom: Float, val score: Float, val label: String)

/** Loads the bundled model and turns a bitmap into filtered fruit detections. */
class PlumDetector(context: Context) : AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val size: Int
    private val labels: List<String>
    private val threshold: Float
    private val nmsThreshold: Float

    init {
        val metadata = JSONObject(context.assets.open("best.json").bufferedReader().use { it.readText() })
        size = metadata.getInt("input_width")
        require(size == metadata.getInt("input_height"))
        labels = metadata.getJSONArray("classes").let { array ->
            (0 until array.length()).map { array.getString(it) }
        }
        threshold = metadata.getDouble("confidence_threshold").toFloat()
        nmsThreshold = metadata.getDouble("nms_iou_threshold").toFloat()
        val bytes = context.assets.open("best.onnx").use { it.readBytes() }
        val options = OrtSession.SessionOptions().apply { setIntraOpNumThreads(2) }
        session = env.createSession(bytes, options)
        options.close()
    }

    /** Letterboxes a frame, runs ONNX inference, and returns NMS-filtered boxes. */
    fun detect(bitmap: Bitmap): List<Detection> {
        val scale = min(size.toFloat() / bitmap.width, size.toFloat() / bitmap.height)
        val width = (bitmap.width * scale).toInt()
        val height = (bitmap.height * scale).toInt()
        val padX = (size - width) / 2f
        val padY = (size - height) / 2f
        val prepared = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        Canvas(prepared).apply {
            drawColor(Color.rgb(114, 114, 114))
            drawBitmap(bitmap, null,
                android.graphics.RectF(padX, padY, padX + width, padY + height),
                Paint(Paint.FILTER_BITMAP_FLAG))
        }
        val pixels = IntArray(size * size)
        prepared.getPixels(pixels, 0, size, 0, 0, size, size)
        prepared.recycle()
        val plane = size * size
        val input = FloatArray(3 * plane)
        for (i in pixels.indices) {
            input[i] = Color.red(pixels[i]) / 255f
            input[plane + i] = Color.green(pixels[i]) / 255f
            input[2 * plane + i] = Color.blue(pixels[i]) / 255f
        }
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), longArrayOf(1, 3, size.toLong(), size.toLong()))
        val candidates = mutableListOf<Detection>()
        tensor.use {
            session.run(mapOf(session.inputNames.first() to tensor)).use { result ->
                @Suppress("UNCHECKED_CAST")
                val channels = (result[0].value as Array<Array<FloatArray>>)[0]
                require(channels.size == labels.size + 4) { "Unexpected YOLO output: ${channels.size} channels" }
                for (i in channels[0].indices) {
                    var bestClass = 0
                    var score = 0f
                    for (c in labels.indices) {
                        val value = channels[c + 4][i]
                        if (value > score) { score = value; bestClass = c }
                    }
                    if (score < threshold) continue
                    val cx = channels[0][i]
                    val cy = channels[1][i]
                    val w = channels[2][i]
                    val h = channels[3][i]
                    val left = ((cx - w / 2 - padX) / scale).coerceIn(0f, bitmap.width.toFloat())
                    val top = ((cy - h / 2 - padY) / scale).coerceIn(0f, bitmap.height.toFloat())
                    val right = ((cx + w / 2 - padX) / scale).coerceIn(0f, bitmap.width.toFloat())
                    val bottom = ((cy + h / 2 - padY) / scale).coerceIn(0f, bitmap.height.toFloat())
                    if (right > left && bottom > top)
                        candidates += Detection(left, top, right, bottom, score, labels[bestClass])
                }
            }
        }
        val kept = mutableListOf<Detection>()
        for (candidate in candidates.sortedByDescending { it.score }) {
            if (kept.size >= 100) break
            if (kept.none { it.label == candidate.label && iou(it, candidate) > nmsThreshold })
                kept += candidate
        }
        return kept
    }

    /** Computes overlap for two boxes during nonmaximum suppression. */
    private fun iou(a: Detection, b: Detection): Float {
        val w = max(0f, min(a.right, b.right) - max(a.left, b.left))
        val h = max(0f, min(a.bottom, b.bottom) - max(a.top, b.top))
        val intersection = w * h
        val areaA = (a.right - a.left) * (a.bottom - a.top)
        val areaB = (b.right - b.left) * (b.bottom - b.top)
        return intersection / max(1e-6f, areaA + areaB - intersection)
    }

    /** Frees the native ONNX Runtime session when the screen closes. */
    override fun close() { session.close() }
}
