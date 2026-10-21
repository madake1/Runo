// Camera screen: permission, preview, frame analysis, detection overlay, and freeze control.
package com.runo.mobile

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors

/** Owns CameraX and coordinates live detection with the screen controls. */
class MainActivity : ComponentActivity() {
    private lateinit var preview: PreviewView
    private lateinit var frozenImage: ImageView
    private lateinit var overlay: DetectionOverlay
    private lateinit var status: TextView
    private lateinit var freezeButton: Button
    private val executor = Executors.newSingleThreadExecutor()
    private var detector: PlumDetector? = null
    private var lastFrame: Bitmap? = null
    private var lastDetections = emptyList<Detection>()
    @Volatile private var frozen = false
    private val permission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) preview.post { startCamera() } else status.text = "需要相机权限"
    }

    /** Builds the preview UI, loads the detector, and requests camera access. */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        preview = PreviewView(this).apply { scaleType = PreviewView.ScaleType.FILL_CENTER }
        frozenImage = ImageView(this).apply {
            scaleType = ImageView.ScaleType.CENTER_CROP
            visibility = View.GONE
        }
        overlay = DetectionOverlay(this)
        status = TextView(this).apply {
            setBackgroundColor(0x99000000.toInt())
            setTextColor(0xffffffff.toInt())
            textSize = 16f
            text = "加载模型中"
            setPadding(16, 12, 16, 12)
        }
        freezeButton = Button(this).apply {
            text = "拍照定格"
            isEnabled = false
            setOnClickListener {
                frozen = !frozen
                if (frozen) {
                    frozenImage.setImageBitmap(lastFrame)
                    frozenImage.visibility = View.VISIBLE
                    text = "继续预览"
                    status.text = "已定格 · ${lastDetections.size} 个李子框"
                } else {
                    frozenImage.visibility = View.GONE
                    text = "拍照定格"
                }
            }
        }
        val stack = FrameLayout(this).apply {
            addView(preview, FrameLayout.LayoutParams(-1, -1))
            addView(frozenImage, FrameLayout.LayoutParams(-1, -1))
            addView(this@MainActivity.overlay, FrameLayout.LayoutParams(-1, -1))
        }
        val column = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(status, LinearLayout.LayoutParams(-1, -2))
            addView(stack, LinearLayout.LayoutParams(-1, 0, 1f))
            addView(freezeButton, LinearLayout.LayoutParams(-1, -2).apply { gravity = Gravity.CENTER })
        }
        setContentView(column)
        try {
            detector = PlumDetector(this)
            status.text = "模型已加载 · 李子果实检测"
        } catch (error: Exception) {
            status.text = "缺少或无法加载模型：请将 best.onnx 和 best.json 放入 app/src/main/assets"
            freezeButton.isEnabled = false
            return
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
            preview.post { startCamera() }
        else permission.launch(Manifest.permission.CAMERA)
    }

    /** Binds preview and analysis streams and sends upright frames to the detector. */
    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val previewUseCase = Preview.Builder().build().also { it.surfaceProvider = preview.surfaceProvider }
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis.setAnalyzer(executor) { image ->
                try {
                    if (frozen) return@setAnalyzer
                    val bitmap = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
                    bitmap.copyPixelsFromBuffer(image.planes[0].buffer)
                    val crop = image.cropRect
                    val matrix = Matrix().apply { postRotate(image.imageInfo.rotationDegrees.toFloat()) }
                    val upright = Bitmap.createBitmap(bitmap, crop.left, crop.top,
                        crop.width(), crop.height(), matrix, true)
                    bitmap.recycle()
                    val boxes = detector?.detect(upright).orEmpty()
                    runOnUiThread {
                        if (!frozen) {
                            lastFrame?.recycle()
                            lastFrame = upright
                            lastDetections = boxes
                            freezeButton.isEnabled = true
                            overlay.update(boxes, upright.width, upright.height)
                            status.text = "李子果实：${boxes.size} 个可见框"
                        } else upright.recycle()
                    }
                } catch (error: Exception) {
                    runOnUiThread { status.text = "推理失败：${error.message}" }
                } finally {
                    image.close()
                }
            }
            provider.unbindAll()
            val group = UseCaseGroup.Builder().addUseCase(previewUseCase).addUseCase(analysis)
            preview.viewPort?.let { group.setViewPort(it) }
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, group.build())
        }, ContextCompat.getMainExecutor(this))
    }

    /** Releases the analysis worker, model session, and cached camera frame. */
    override fun onDestroy() {
        super.onDestroy()
        executor.shutdown()
        detector?.close()
        lastFrame?.recycle()
    }
}
