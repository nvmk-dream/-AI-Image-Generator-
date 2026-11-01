package com.example.ai

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.app.ActivityManager
import android.content.Context
import android.net.Uri
import android.os.StatFs
import androidx.documentfile.provider.DocumentFile
import java.io.File
import java.io.FileOutputStream
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap

enum class DiffusionModelSlot(
    val canonicalFileName: String,
    val folderHint: String,
    val titleEn: String,
    val titleTa: String,
    val descriptionEn: String,
    val descriptionTa: String,
    val requiredForTxt2Img: Boolean
) {
    TEXT_ENCODER(
        canonicalFileName = "text_encoder.onnx",
        folderHint = "text_encoder",
        titleEn = "1. Text Encoder (CLIP)",
        titleTa = "1. Text Encoder (எழுத்து மாற்றி)",
        descriptionEn = "Converts prompt words into embeddings",
        descriptionTa = "உங்கள் வாக்கியத்தை AI புரியும் எண்களாக மாற்றும்",
        requiredForTxt2Img = true
    ),
    UNET(
        canonicalFileName = "unet.onnx",
        folderHint = "unet",
        titleEn = "2. LCM / SD 1.5 UNet",
        titleTa = "2. LCM / SD 1.5 UNet (வேகமான மாடல்)",
        descriptionEn = "Fast 4–8 step 512×512 image generator core",
        descriptionTa = "4–8 படிகளில் வேகமாகப் படம் வரையும் முக்கிய மாடல்",
        requiredForTxt2Img = true
    ),
    VAE_DECODER(
        canonicalFileName = "vae_decoder.onnx",
        folderHint = "vae_decoder",
        titleEn = "3. VAE Decoder",
        titleTa = "3. VAE Decoder (பட உருவாக்கி)",
        descriptionEn = "Converts latents into crisp 512×512 photo",
        descriptionTa = "வரைந்த குறிப்பைத் தெளிவான 512×512 புகைப்படமாக மாற்றும்",
        requiredForTxt2Img = true
    ),
    VAE_ENCODER(
        canonicalFileName = "vae_encoder.onnx",
        folderHint = "vae_encoder",
        titleEn = "4. VAE Encoder (Img2Img)",
        titleTa = "4. VAE Encoder (Image-to-Image)",
        descriptionEn = "Encodes reference photo for Image-to-Image",
        descriptionTa = "Image-to-Image மாற்றத்திற்காக பழைய படத்தை உள்ளே எடுக்கும்",
        requiredForTxt2Img = false
    )
}

data class ModelSlotStatus(
    val slot: DiffusionModelSlot,
    val isInstalled: Boolean,
    val isLoadedInRam: Boolean,
    val sizeBytes: Long,
    val companionWeightsBytes: Long = 0L,
    val statusNote: String
) {
    val totalBytes: Long
        get() = sizeBytes + companionWeightsBytes
}

data class RealDeviceAndModelMetrics(
    val ortRuntimeAvailable: Boolean = true,
    val ortVersion: String = "1.20.0",
    val slots: List<ModelSlotStatus> = emptyList(),
    val totalModelBytes: Long = 0L,
    val cacheBytes: Long = 0L,
    val deviceFreeBytes: Long = 0L,
    val deviceTotalBytes: Long = 0L,
    val deviceTotalRamBytes: Long = 0L,
    val deviceAvailableRamBytes: Long = 0L,
    val hasTokenizerFiles: Boolean = false,
    val activeRamSessionsCount: Int = 0
) {
    val installedCount: Int
        get() = slots.count { it.isInstalled }

    val isTxt2ImgReady: Boolean
        get() = slots.filter { it.slot.requiredForTxt2Img }.all { it.isInstalled }

    val isAllFourReady: Boolean
        get() = slots.isNotEmpty() && slots.all { it.isInstalled }
}

class OnnxModelManager(private val context: Context) {

    private val activeSessions = ConcurrentHashMap<DiffusionModelSlot, OrtSession>()

    private val modelsRootDir: File
        get() = File(context.filesDir, "onnx_models").apply { if (!exists()) mkdirs() }

    private fun slotDir(slot: DiffusionModelSlot): File =
        File(modelsRootDir, slot.folderHint).apply { if (!exists()) mkdirs() }

    fun slotModelFile(slot: DiffusionModelSlot): File =
        File(slotDir(slot), "model.onnx")

    fun inspectCurrentMetrics(): RealDeviceAndModelMetrics {
        val (ortOk, ortVer) = runCatching {
            val env = OrtEnvironment.getEnvironment()
            true to env.version
        }.getOrElse { false to "Unavailable" }

        val slotStatuses = DiffusionModelSlot.entries.map { slot ->
            val dir = slotDir(slot)
            val mainFile = File(dir, "model.onnx")
            val legacyFile = File(modelsRootDir, slot.canonicalFileName)
            val actualFile = when {
                mainFile.exists() && mainFile.length() > 1024L -> mainFile
                legacyFile.exists() && legacyFile.length() > 1024L -> legacyFile
                else -> null
            }
            val extraBytes = dir.listFiles()
                ?.filter { it.isFile && it.name != "model.onnx" }
                ?.sumOf { it.length() } ?: 0L

            val installed = actualFile != null
            val inRam = activeSessions.containsKey(slot)
            ModelSlotStatus(
                slot = slot,
                isInstalled = installed,
                isLoadedInRam = inRam,
                sizeBytes = actualFile?.length() ?: 0L,
                companionWeightsBytes = extraBytes,
                statusNote = when {
                    inRam -> "Loaded in RAM"
                    installed -> "Ready on Device"
                    else -> "Not Imported"
                }
            )
        }

        val tokDir = File(modelsRootDir, "tokenizer")
        val hasTokenizer = File(tokDir, "vocab.json").exists() ||
            File(modelsRootDir, "vocab.json").exists()

        val modelBytes = calculateDirectorySize(modelsRootDir)
        val cacheBytes = calculateDirectorySize(context.cacheDir)

        val statFs = StatFs(context.filesDir.absolutePath)
        val freeBytes = statFs.availableBytes
        val totalBytes = statFs.totalBytes

        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        am?.getMemoryInfo(memInfo)

        return RealDeviceAndModelMetrics(
            ortRuntimeAvailable = ortOk,
            ortVersion = ortVer,
            slots = slotStatuses,
            totalModelBytes = modelBytes,
            cacheBytes = cacheBytes,
            deviceFreeBytes = freeBytes,
            deviceTotalBytes = totalBytes,
            deviceTotalRamBytes = memInfo.totalMem,
            deviceAvailableRamBytes = memInfo.availMem,
            hasTokenizerFiles = hasTokenizer,
            activeRamSessionsCount = activeSessions.size
        )
    }

    fun importSingleSlotFromUri(uri: Uri, slot: DiffusionModelSlot): Result<Long> = runCatching {
        val targetDir = slotDir(slot)
        targetDir.listFiles()?.forEach { it.delete() }
        val destFile = File(targetDir, "model.onnx")
        copyUriToFile(uri, destFile)
        if (destFile.length() < 1024L) {
            destFile.delete()
            error("Selected file is too small to be a valid ONNX model.")
        }
        destFile.length()
    }

    /**
     * One-Touch Folder Upload (modeled after nvmk-dream/face-swap):
     * Scans both flat folders (text_encoder.onnx, unet.onnx, vae_decoder.onnx, vae_encoder.onnx)
     * and nested Hugging Face Optimum folders (text_encoder/model.onnx, unet/model.onnx, etc.),
     * plus optional .onnx_data / .onnx.data external weight files and tokenizer JSON/txt files.
     */
    fun importAllModelsFromFolder(
        treeUri: Uri,
        onProgress: (String) -> Unit
    ): Result<List<DiffusionModelSlot>> = runCatching {
        val rootDoc = DocumentFile.fromTreeUri(context, treeUri)
            ?: error("Could not open selected folder.")

        data class DiscoveredDoc(val parentName: String, val doc: DocumentFile)

        val allFiles = mutableListOf<DiscoveredDoc>()
        fun walk(dir: DocumentFile, parentLabel: String) {
            for (child in dir.listFiles()) {
                if (child.isDirectory) {
                    walk(child, child.name.orEmpty().lowercase(Locale.US))
                } else if (child.isFile) {
                    allFiles.add(DiscoveredDoc(parentLabel, child))
                }
            }
        }
        onProgress("Scanning folder for .onnx models...")
        walk(rootDoc, rootDoc.name.orEmpty().lowercase(Locale.US))

        val importedSlots = mutableListOf<DiffusionModelSlot>()

        for (slot in DiffusionModelSlot.entries) {
            val match = allFiles.firstOrNull { entry ->
                val fileName = entry.doc.name.orEmpty().lowercase(Locale.US)
                val isOnnx = fileName.endsWith(".onnx")
                isOnnx && (
                    fileName == slot.canonicalFileName ||
                        fileName.contains(slot.folderHint) ||
                        (entry.parentName.contains(slot.folderHint) && fileName.endsWith(".onnx"))
                    )
            }
            if (match != null) {
                onProgress("Copying ${slot.canonicalFileName}...")
                val destDir = slotDir(slot)
                destDir.listFiles()?.forEach { it.delete() }
                val destOnnx = File(destDir, "model.onnx")
                copyUriToFile(match.doc.uri, destOnnx)

                // Copy any external data weight file (.onnx_data / .onnx.data) in the same subfolder
                val matchBase = match.doc.name.orEmpty()
                allFiles.filter { other ->
                    val otherName = other.doc.name.orEmpty()
                    other.parentName == match.parentName &&
                        otherName != matchBase &&
                        (otherName.endsWith(".onnx_data") || otherName.endsWith(".onnx.data") || otherName.endsWith(".pb"))
                }.forEach { companion ->
                    val compDest = File(destDir, companion.doc.name ?: "model.onnx_data")
                    copyUriToFile(companion.doc.uri, compDest)
                }

                if (destOnnx.exists() && destOnnx.length() > 1024L) {
                    importedSlots.add(slot)
                }
            }
        }

        // Also copy tokenizer / scheduler config files if present in the selected folder
        val tokDestDir = File(modelsRootDir, "tokenizer").apply { if (!exists()) mkdirs() }
        val configNames = setOf(
            "vocab.json",
            "merges.txt",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "scheduler_config.json"
        )
        allFiles.filter { it.doc.name.orEmpty().lowercase(Locale.US) in configNames }
            .forEach { cfg ->
                val dest = File(tokDestDir, cfg.doc.name ?: return@forEach)
                copyUriToFile(cfg.doc.uri, dest)
            }

        importedSlots
    }

    fun deleteSlotModel(slot: DiffusionModelSlot) {
        activeSessions.remove(slot)?.runCatching { close() }
        slotDir(slot).deleteRecursively()
        File(modelsRootDir, slot.canonicalFileName).delete()
    }

    fun deleteAllModels() {
        unloadSessionsFromRam()
        modelsRootDir.deleteRecursively()
        modelsRootDir.mkdirs()
    }

    fun clearAppCache(): Long {
        val before = calculateDirectorySize(context.cacheDir)
        context.cacheDir.listFiles()?.forEach { it.deleteRecursively() }
        val after = calculateDirectorySize(context.cacheDir)
        return (before - after).coerceAtLeast(0L)
    }

    fun unloadSessionsFromRam(): Int {
        val count = activeSessions.size
        activeSessions.values.forEach { session -> runCatching { session.close() } }
        activeSessions.clear()
        System.gc()
        return count
    }

    private fun copyUriToFile(uri: Uri, dest: File) {
        context.contentResolver.openInputStream(uri)?.use { input ->
            FileOutputStream(dest).use { output ->
                input.copyTo(output, bufferSize = 256 * 1024)
                output.flush()
            }
        } ?: error("Unable to read selected file.")
    }

    private fun calculateDirectorySize(dir: File?): Long {
        if (dir == null || !dir.exists()) return 0L
        if (dir.isFile) return dir.length()
        return dir.walkTopDown().filter { it.isFile }.sumOf { it.length() }
    }

    companion object {
        fun formatBytes(bytes: Long): String {
            if (bytes <= 0L) return "0 MB"
            val mb = bytes.toDouble() / (1024.0 * 1024.0)
            return if (mb >= 1024.0) {
                String.format(Locale.US, "%.2f GB", mb / 1024.0)
            } else {
                String.format(Locale.US, "%.1f MB", mb)
            }
        }
    }
}
