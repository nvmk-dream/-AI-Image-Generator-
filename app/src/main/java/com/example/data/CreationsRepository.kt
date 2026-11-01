package com.example.data

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream

data class GeneratedCreation(
    val id: String,
    val filePath: String,
    val userPrompt: String,
    val finalGenerationInput: String,
    val resolution: String,
    val steps: Int,
    val seed: Long,
    val durationMs: Long,
    val createdAtEpochMs: Long,
    val isFavorite: Boolean = false
)

class CreationsRepository(private val context: Context) {

    private val creationsDir: File
        get() = File(context.filesDir, "creations").apply { if (!exists()) mkdirs() }

    private val indexFile: File
        get() = File(creationsDir, "creations_index.json")

    fun loadAllCreations(): List<GeneratedCreation> {
        if (!indexFile.exists()) return emptyList()
        return runCatching {
            val arr = JSONArray(indexFile.readText())
            buildList {
                for (i in 0 until arr.length()) {
                    val obj = arr.getJSONObject(i)
                    val path = obj.optString("filePath")
                    if (path.isNotBlank() && File(path).exists()) {
                        add(
                            GeneratedCreation(
                                id = obj.optString("id"),
                                filePath = path,
                                userPrompt = obj.optString("userPrompt"),
                                finalGenerationInput = obj.optString("finalGenerationInput"),
                                resolution = obj.optString("resolution", "512 × 512"),
                                steps = obj.optInt("steps", 6),
                                seed = obj.optLong("seed", 0L),
                                durationMs = obj.optLong("durationMs", 0L),
                                createdAtEpochMs = obj.optLong("createdAtEpochMs", 0L),
                                isFavorite = obj.optBoolean("isFavorite", false)
                            )
                        )
                    }
                }
            }.sortedByDescending { it.createdAtEpochMs }
        }.getOrDefault(emptyList())
    }

    fun saveCreation(
        bitmap: Bitmap,
        userPrompt: String,
        finalGenerationInput: String,
        resolution: String,
        steps: Int,
        seed: Long,
        durationMs: Long
    ): GeneratedCreation {
        val now = System.currentTimeMillis()
        val id = "gen_$now"
        val imgFile = File(creationsDir, "$id.png")
        FileOutputStream(imgFile).use { out ->
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
            out.flush()
        }

        val item = GeneratedCreation(
            id = id,
            filePath = imgFile.absolutePath,
            userPrompt = userPrompt,
            finalGenerationInput = finalGenerationInput,
            resolution = resolution,
            steps = steps,
            seed = seed,
            durationMs = durationMs,
            createdAtEpochMs = now,
            isFavorite = false
        )

        val updated = listOf(item) + loadAllCreations()
        persistIndex(updated)
        return item
    }

    fun toggleFavorite(id: String): List<GeneratedCreation> {
        val updated = loadAllCreations().map {
            if (it.id == id) it.copy(isFavorite = !it.isFavorite) else it
        }
        persistIndex(updated)
        return updated
    }

    fun deleteCreation(id: String): List<GeneratedCreation> {
        val current = loadAllCreations()
        current.firstOrNull { it.id == id }?.let { item ->
            runCatching { File(item.filePath).delete() }
        }
        val updated = current.filterNot { it.id == id }
        persistIndex(updated)
        return updated
    }

    fun exportToPhoneGallery(creation: GeneratedCreation): Result<String> = runCatching {
        val srcFile = File(creation.filePath)
        require(srcFile.exists()) { "Image file not found on device." }

        val resolver = context.contentResolver
        val fileName = "ImageGen_${creation.createdAtEpochMs}.png"
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, fileName)
            put(MediaStore.Images.Media.MIME_TYPE, "image/png")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(
                    MediaStore.Images.Media.RELATIVE_PATH,
                    Environment.DIRECTORY_PICTURES + "/ImageGenerator"
                )
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }
        }

        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            ?: error("Could not create MediaStore entry.")

        resolver.openOutputStream(uri)?.use { out ->
            srcFile.inputStream().use { input -> input.copyTo(out) }
        } ?: error("Could not write image to Gallery.")

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val done = ContentValues().apply {
                put(MediaStore.Images.Media.IS_PENDING, 0)
            }
            resolver.update(uri, done, null, null)
        }
        "Pictures/ImageGenerator/$fileName"
    }

    private fun persistIndex(items: List<GeneratedCreation>) {
        val arr = JSONArray()
        items.forEach { item ->
            arr.put(
                JSONObject().apply {
                    put("id", item.id)
                    put("filePath", item.filePath)
                    put("userPrompt", item.userPrompt)
                    put("finalGenerationInput", item.finalGenerationInput)
                    put("resolution", item.resolution)
                    put("steps", item.steps)
                    put("seed", item.seed)
                    put("durationMs", item.durationMs)
                    put("createdAtEpochMs", item.createdAtEpochMs)
                    put("isFavorite", item.isFavorite)
                }
            )
        }
        indexFile.writeText(arr.toString())
    }
}
