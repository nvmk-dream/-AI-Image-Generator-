package com.example.data

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import com.example.ai.ImageBitmapPreprocessor
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream

data class CharacterProfile(
    val id: String,
    val name: String,
    val appearancePrompt: String,
    val fixedSeed: Long,
    val referencePhotoPath: String? = null,
    val createdAtEpochMs: Long = System.currentTimeMillis()
) {
    fun buildCharacterScenePrompt(scenePrompt: String): String {
        val base = appearancePrompt.trim()
        val scene = scenePrompt.trim()
        return when {
            base.isEmpty() -> scene
            scene.isEmpty() -> base
            else -> "$base, $scene"
        }
    }
}

class CharacterStudioRepository(private val context: Context) {

    private val charsDir: File
        get() = File(context.filesDir, "characters").apply { if (!exists()) mkdirs() }

    private val indexFile: File
        get() = File(charsDir, "characters_index.json")

    fun loadCharacters(): List<CharacterProfile> {
        if (!indexFile.exists()) {
            val starter = listOf(
                CharacterProfile(
                    id = "char_starter_ananya",
                    name = "Ananya (அனன்யா)",
                    appearancePrompt = "graceful young South Indian woman, expressive dark eyes, long wavy black hair, warm natural smile, photorealistic face",
                    fixedSeed = 48291037L,
                    referencePhotoPath = null,
                    createdAtEpochMs = 1700000000000L
                )
            )
            persist(starter)
            return starter
        }

        return runCatching {
            val arr = JSONArray(indexFile.readText())
            buildList {
                for (i in 0 until arr.length()) {
                    val obj = arr.getJSONObject(i)
                    val rawRef = obj.optString("referencePhotoPath").takeIf { it.isNotBlank() }
                    val validRef = rawRef?.takeIf { File(it).exists() }
                    add(
                        CharacterProfile(
                            id = obj.optString("id"),
                            name = obj.optString("name", "Character"),
                            appearancePrompt = obj.optString("appearancePrompt"),
                            fixedSeed = obj.optLong("fixedSeed", 1234567L),
                            referencePhotoPath = validRef,
                            createdAtEpochMs = obj.optLong("createdAtEpochMs", 0L)
                        )
                    )
                }
            }.sortedByDescending { it.createdAtEpochMs }
        }.getOrDefault(emptyList())
    }

    fun saveCharacter(
        name: String,
        appearancePrompt: String,
        fixedSeed: Long,
        referencePhotoUri: Uri?
    ): List<CharacterProfile> {
        val now = System.currentTimeMillis()
        val id = "char_$now"
        val savedPhotoPath = if (referencePhotoUri != null) {
            runCatching {
                val cropped = ImageBitmapPreprocessor.loadAndCenterCropFromUri(
                    context = context,
                    uri = referencePhotoUri,
                    targetWidth = 512,
                    targetHeight = 512
                )
                val dest = File(charsDir, "$id.png")
                FileOutputStream(dest).use { out ->
                    cropped.compress(Bitmap.CompressFormat.PNG, 100, out)
                    out.flush()
                }
                cropped.recycle()
                dest.absolutePath
            }.getOrNull()
        } else {
            null
        }

        val profile = CharacterProfile(
            id = id,
            name = name.trim().ifBlank { "My Character" },
            appearancePrompt = appearancePrompt.trim(),
            fixedSeed = fixedSeed,
            referencePhotoPath = savedPhotoPath,
            createdAtEpochMs = now
        )

        val updated = listOf(profile) + loadCharacters()
        persist(updated)
        return updated
    }

    fun deleteCharacter(id: String): List<CharacterProfile> {
        val existing = loadCharacters()
        existing.firstOrNull { it.id == id }?.referencePhotoPath?.let { path ->
            runCatching { File(path).delete() }
        }
        val updated = existing.filterNot { it.id == id }
        persist(updated)
        return updated
    }

    private fun persist(items: List<CharacterProfile>) {
        val arr = JSONArray()
        items.forEach { item ->
            arr.put(
                JSONObject().apply {
                    put("id", item.id)
                    put("name", item.name)
                    put("appearancePrompt", item.appearancePrompt)
                    put("fixedSeed", item.fixedSeed)
                    put("referencePhotoPath", item.referencePhotoPath ?: "")
                    put("createdAtEpochMs", item.createdAtEpochMs)
                }
            )
        }
        indexFile.writeText(arr.toString())
    }
}
