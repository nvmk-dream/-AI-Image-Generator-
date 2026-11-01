package com.example.ai

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.app.ActivityManager
import android.content.Context
import android.net.Uri
import android.os.StatFs
import android.provider.OpenableColumns
import androidx.documentfile.provider.DocumentFile
import java.io.File
import java.io.FileOutputStream
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap

enum class DiffusionModelSlot(
    val canonicalFileName: String,
    val folderHint: String,
    val matchKeywords: List<String>,
    val titleEn: String,
    val titleTa: String,
    val descriptionEn: String,
    val descriptionTa: String,
    val requiredForTxt2Img: Boolean,
    val isFaceSwapSlot: Boolean = false
) {
    TEXT_ENCODER(
        canonicalFileName = "text_encoder.onnx",
        folderHint = "text_encoder",
        matchKeywords = listOf("text_encoder"),
        titleEn = "1. Text Encoder (CLIP)",
        titleTa = "1. Text Encoder (எழுத்து மாற்றி)",
        descriptionEn = "Converts prompt words into embeddings",
        descriptionTa = "உங்கள் வாக்கியத்தை AI புரியும் எண்களாக மாற்றும்",
        requiredForTxt2Img = true
    ),
    UNET(
        canonicalFileName = "unet.onnx",
        folderHint = "unet",
        matchKeywords = listOf("unet"),
        titleEn = "2. LCM / SD 1.5 UNet",
        titleTa = "2. LCM / SD 1.5 UNet (வேகமான மாடல்)",
        descriptionEn = "Fast 4–8 step 512×512 image generator core",
        descriptionTa = "4–8 படிகளில் வேகமாகப் படம் வரையும் முக்கிய மாடல்",
        requiredForTxt2Img = true
    ),
    VAE_DECODER(
        canonicalFileName = "vae_decoder.onnx",
        folderHint = "vae_decoder",
        matchKeywords = listOf("vae_decoder"),
        titleEn = "3. VAE Decoder",
        titleTa = "3. VAE Decoder (பட உருவாக்கி)",
        descriptionEn = "Converts latents into crisp 512×512 photo",
        descriptionTa = "வரைந்த குறிப்பைத் தெளிவான 512×512 புகைப்படமாக மாற்றும்",
        requiredForTxt2Img = true
    ),
    VAE_ENCODER(
        canonicalFileName = "vae_encoder.onnx",
        folderHint = "vae_encoder",
        matchKeywords = listOf("vae_encoder"),
        titleEn = "4. VAE Encoder (Img2Img)",
        titleTa = "4. VAE Encoder (Image-to-Image)",
        descriptionEn = "Encodes reference photo for Image-to-Image",
        descriptionTa = "Image-to-Image மாற்றத்திற்காக பழைய படத்தை உள்ளே எடுக்கும்",
        requiredForTxt2Img = false
    ),
    FACE_DETECTOR(
        canonicalFileName = "det_10g.onnx",
        folderHint = "detection",
        matchKeywords = listOf("det_10g", "det_2.5g", "scrfd"),
        titleEn = "5. Face Detector (det_10g)",
        titleTa = "5. Face Detector (முகம் கண்டறியும் மாடல்)",
        descriptionEn = "SCRFD 5-point landmark detector for Face Swap",
        descriptionTa = "புகைப்படத்தில் உள்ள முகம் & 5 புள்ளிகளைத் துல்லியமாகக் கண்டறியும்",
        requiredForTxt2Img = false,
        isFaceSwapSlot = true
    ),
    FACE_RECOGNIZER(
        canonicalFileName = "w600k_r50.onnx",
        folderHint = "recognition",
        matchKeywords = listOf("w600k_r50", "w600k_mbf", "arcface"),
        titleEn = "6. Face Identity (w600k_r50)",
        titleTa = "6. Face Identity (முக அடையாள மாடல்)",
        descriptionEn = "512-D ArcFace source identity extractor",
        descriptionTa = "Source புகைப்படத்தின் முக சாயலை 512-D அடையாளமாகப் பிரித்தெடுக்கும்",
        requiredForTxt2Img = false,
        isFaceSwapSlot = true
    ),
    FACE_SWAPPER(
        canonicalFileName = "inswapper_128.onnx",
        folderHint = "face_swap",
        matchKeywords = listOf("inswapper", "hyperswap"),
        titleEn = "7. Face Swapper (InSwapper / HyperSwap)",
        titleTa = "7. Face Swapper (முகம் மாற்றும் மாடல்)",
        descriptionEn = "Supports inswapper_128.onnx & hyperswap_1b_256.onnx",
        descriptionTa = "inswapper_128.onnx அல்லது hyperswap_1b_256.onnx மூலம் முகம் மாற்றும்",
        requiredForTxt2Img = false,
        isFaceSwapSlot = true
    )
}

data class ModelSlotStatus(
    val slot: DiffusionModelSlot,
    val isInstalled: Boolean,
    val isLoadedInRam: Boolean,
    val sizeBytes: Long,
    val companionWeightsBytes: Long = 0L,
    val sourceFileLabel: String = "",
    val statusNote: String,
    val requiredCompanionFiles: List<String> = emptyList(),
    val missingCompanionFiles: List<String> = emptyList()
) {
    val totalBytes: Long
        get() = sizeBytes + companionWeightsBytes

    val hasGraphFile: Boolean
        get() = sizeBytes > 1024L

    val isIncompleteExternalGraph: Boolean
        get() = hasGraphFile && missingCompanionFiles.isNotEmpty()
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
    val diffusionSlots: List<ModelSlotStatus>
        get() = slots.filter { !it.slot.isFaceSwapSlot }

    val faceSwapSlots: List<ModelSlotStatus>
        get() = slots.filter { it.slot.isFaceSwapSlot }

    // Count of installed & complete core Diffusion models (0..4)
    val installedCount: Int
        get() = diffusionSlots.count { it.isInstalled }

    val faceSwapInstalledCount: Int
        get() = faceSwapSlots.count { it.isInstalled }

    val totalInstalledCount: Int
        get() = slots.count { it.isInstalled }

    val hasAnySlotFilesOnDisk: Boolean
        get() = slots.any { it.isInstalled || it.isIncompleteExternalGraph }

    val incompleteSlots: List<ModelSlotStatus>
        get() = slots.filter { it.isIncompleteExternalGraph }

    val isTxt2ImgReady: Boolean
        get() = diffusionSlots.filter { it.slot.requiredForTxt2Img }.let { req ->
            req.isNotEmpty() && req.all { it.isInstalled }
        }

    val isAllFourReady: Boolean
        get() = diffusionSlots.isNotEmpty() && diffusionSlots.all { it.isInstalled }

    val isFaceSwapReady: Boolean
        get() {
            val detOk = slots.any { it.slot == DiffusionModelSlot.FACE_DETECTOR && it.isInstalled }
            val swapOk = slots.any { it.slot == DiffusionModelSlot.FACE_SWAPPER && it.isInstalled }
            return detOk && swapOk
        }
}

class OnnxModelManager(private val context: Context) {

    private val activeSessions = ConcurrentHashMap<DiffusionModelSlot, OrtSession>()

    val modelsRootDir: File
        get() = File(context.filesDir, "onnx_models").apply { if (!exists()) mkdirs() }

    private fun slotDir(slot: DiffusionModelSlot): File =
        File(modelsRootDir, slot.folderHint).apply { if (!exists()) mkdirs() }

    fun slotModelFile(slot: DiffusionModelSlot): File =
        File(slotDir(slot), "model.onnx")

    fun resolveExistingSlotFile(slot: DiffusionModelSlot): File? {
        val dir = slotDir(slot)
        val namedOnnx = dir.listFiles()
            ?.firstOrNull { it.isFile && it.name.endsWith(".onnx", ignoreCase = true) && it.length() > 1024L }
        if (namedOnnx != null) return namedOnnx

        val altFile = File(modelsRootDir, slot.canonicalFileName)
        if (altFile.exists() && altFile.length() > 1024L) return altFile
        return null
    }

    fun findMissingCompanionFilesForSlot(slot: DiffusionModelSlot): List<String> {
        val actualFile = resolveExistingSlotFile(slot) ?: return emptyList()
        return findMissingCompanionFiles(actualFile)
    }

    fun inspectCurrentMetrics(): RealDeviceAndModelMetrics {
        purgeOrphanedDuplicateWeights()

        val (ortOk, ortVer) = runCatching {
            val env = OrtEnvironment.getEnvironment()
            true to env.version
        }.getOrElse { false to "Unavailable" }

        val slotStatuses = DiffusionModelSlot.entries.map { slot ->
            val dir = slotDir(slot)
            val actualFile = resolveExistingSlotFile(slot)
            val requiredCompanions = if (actualFile != null) {
                extractExternalDataLocations(actualFile)
            } else {
                emptyList()
            }
            val missingCompanions = if (actualFile != null) {
                findMissingCompanionFiles(actualFile)
            } else {
                emptyList()
            }
            val extraBytes = dir.listFiles()
                ?.filter { it.isFile && it.absolutePath != actualFile?.absolutePath && !it.name.endsWith(".meta") }
                ?.sumOf { it.length() } ?: 0L
            val originLabel = File(dir, "source_name.meta").takeIf { it.exists() }
                ?.readText()?.trim().orEmpty()

            val installed = actualFile != null && missingCompanions.isEmpty()
            val inRam = activeSessions.containsKey(slot)
            ModelSlotStatus(
                slot = slot,
                isInstalled = installed,
                isLoadedInRam = inRam,
                sizeBytes = actualFile?.length() ?: 0L,
                companionWeightsBytes = extraBytes,
                sourceFileLabel = originLabel.ifBlank { actualFile?.name ?: slot.canonicalFileName },
                statusNote = when {
                    inRam -> "Loaded in RAM"
                    actualFile != null && missingCompanions.isNotEmpty() ->
                        "Missing ${missingCompanions.joinToString()}"
                    installed -> "Ready on Device"
                    else -> "Not Imported"
                },
                requiredCompanionFiles = requiredCompanions,
                missingCompanionFiles = missingCompanions
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

    /**
     * Automatically deletes any stray duplicate external-weight files (such as a 3.44 GB weights.pb
     * accidentally copied alongside self-contained text_encoder/vae_decoder/vae_encoder models),
     * while preserving any external data file actually referenced by the slot's .onnx protobuf.
     */
    private fun purgeOrphanedDuplicateWeights() {
        for (slot in DiffusionModelSlot.entries) {
            val dir = slotDir(slot)
            val actualOnnx = resolveExistingSlotFile(slot) ?: continue
            val expectedLower = extractExternalDataLocations(actualOnnx)
                .map { it.lowercase(Locale.US) }
                .toSet()
            val originName = File(dir, "source_name.meta").takeIf { it.exists() }
                ?.readText()?.substringBefore("+")?.trim()?.ifBlank { actualOnnx.name }
                ?: actualOnnx.name

            dir.listFiles()?.forEach { f ->
                if (f.isFile && f.absolutePath != actualOnnx.absolutePath && isCompanionWeightFileName(f.name)) {
                    val keep = if (expectedLower.isNotEmpty()) {
                        f.name.lowercase(Locale.US) in expectedLower
                    } else {
                        shouldCopyCompanionWeight(
                            onnxFileName = originName,
                            onnxSizeBytes = actualOnnx.length(),
                            companionFileName = f.name,
                            onnxFileForRefScan = actualOnnx
                        )
                    }
                    if (!keep) {
                        runCatching { f.delete() }
                    }
                }
            }
        }
    }

    fun importSingleSlotFromUri(
        uri: Uri,
        slot: DiffusionModelSlot,
        originalNameHint: String? = null,
        onProgress: ((String) -> Unit)? = null
    ): Result<Long> = runCatching {
        val normalizedHint = normalizeBrowserDuplicateName(originalNameHint.orEmpty())

        // Smart protection: if this slot already has its .onnx graph shell and the user picked a
        // companion weights file (e.g. weights.pb or model.onnx.data), attach it alongside model.onnx!
        if (normalizedHint.isNotBlank() &&
            isCompanionWeightFileName(normalizedHint) &&
            resolveExistingSlotFile(slot) != null
        ) {
            return importCompanionWeightForSlot(uri, slot, originalNameHint, onProgress).map { it.second }
        }

        val targetDir = slotDir(slot)
        targetDir.listFiles()?.forEach { it.delete() }
        val chosenName = if (slot == DiffusionModelSlot.FACE_SWAPPER &&
            normalizedHint.contains("hyperswap", ignoreCase = true)
        ) {
            "hyperswap_256.onnx"
        } else {
            "model.onnx"
        }
        val destFile = File(targetDir, chosenName)
        val displayLabel = originalNameHint?.takeIf { it.isNotBlank() } ?: slot.canonicalFileName
        val expectedTotal = queryUriLength(uri)
        ensureEnoughFreeSpace(expectedTotal)

        copyUriToFile(uri, destFile, expectedTotal) { copied, total ->
            onProgress?.invoke(formatCopyProgress(displayLabel, copied, total))
        }
        if (destFile.length() < 1024L) {
            destFile.delete()
            error("Selected file is too small to be a valid ONNX model.")
        }
        if (!originalNameHint.isNullOrBlank()) {
            File(targetDir, "source_name.meta").writeText(originalNameHint)
        }
        if (slot == DiffusionModelSlot.FACE_SWAPPER) {
            File(modelsRootDir, "inswapper_emap_512x512.bin").delete()
        }
        destFile.length()
    }

    /**
     * Attaches an external weights companion file (e.g. weights.pb or model.onnx.data) to an
     * existing .onnx slot, automatically normalizing browser-renamed duplicates like
     * "weights (1).pb" or "model.onnx (1).data" to the exact filename expected by ONNX Runtime.
     */
    fun importCompanionWeightForSlot(
        uri: Uri,
        slot: DiffusionModelSlot,
        originalNameHint: String? = null,
        onProgress: ((String) -> Unit)? = null
    ): Result<Pair<String, Long>> = runCatching {
        val existingOnnx = resolveExistingSlotFile(slot)
            ?: error("Please upload the .onnx graph file for ${slot.canonicalFileName} first.")
        val targetDir = existingOnnx.parentFile ?: slotDir(slot)
        val expectedLocations = extractExternalDataLocations(existingOnnx)
        val normalizedHint = normalizeBrowserDuplicateName(originalNameHint.orEmpty())

        val targetCompanionName = when {
            expectedLocations.size == 1 -> expectedLocations.first()
            expectedLocations.isNotEmpty() -> {
                expectedLocations.firstOrNull { it.equals(normalizedHint, ignoreCase = true) }
                    ?: expectedLocations.first()
            }
            normalizedHint.isNotBlank() -> normalizedHint
            else -> "weights.pb"
        }

        val expectedTotal = queryUriLength(uri)
        ensureEnoughFreeSpace(expectedTotal)

        val destCompanion = File(targetDir, targetCompanionName)
        copyUriToFile(uri, destCompanion, expectedTotal) { copied, total ->
            onProgress?.invoke(formatCopyProgress(targetCompanionName, copied, total))
        }
        if (destCompanion.length() < MIN_VALID_COMPANION_BYTES) {
            destCompanion.delete()
            error("Selected companion file ($targetCompanionName) is too small or incomplete.")
        }

        val metaFile = File(targetDir, "source_name.meta")
        val baseLabel = metaFile.takeIf { it.exists() }
            ?.readText()?.substringBefore("+")?.trim()?.ifBlank { existingOnnx.name }
            ?: existingOnnx.name
        metaFile.writeText("$baseLabel + $targetCompanionName")

        activeSessions.remove(slot)?.runCatching { close() }
        targetCompanionName to destCompanion.length()
    }

    /**
     * One-Touch Folder Upload (compatible with nested Hugging Face folders, flat SD 1.5 folders
     * with model.onnx + weights.pb, quantized model.onnx + model.onnx.data, AND Face Swap models).
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

        val candidates = allFiles.mapIndexedNotNull { idx, entry ->
            val rawName = entry.doc.name.orEmpty()
            val normName = normalizeBrowserDuplicateName(rawName)
            if (normName.endsWith(".onnx", ignoreCase = true)) {
                CandidateModelFile(
                    index = idx,
                    fileName = normName,
                    parentNameLower = entry.parentName
                )
            } else {
                null
            }
        }
        val allFileNamesByParent: Map<String, List<String>> = allFiles
            .groupBy(
                keySelector = { it.parentName },
                valueTransform = { normalizeBrowserDuplicateName(it.doc.name.orEmpty()).lowercase(Locale.US) }
            )

        val slotMatches = resolveFolderSlotMatches(candidates, allFileNamesByParent)
        val importedSlots = mutableListOf<DiffusionModelSlot>()

        for ((slot, matchedIndex) in slotMatches) {
            val match = allFiles[matchedIndex]
            val origName = match.doc.name.orEmpty()
            val normOrigName = normalizeBrowserDuplicateName(origName)
            val labelName = origName.ifBlank { slot.canonicalFileName }
            onProgress("Copying $labelName...")

            val destDir = slotDir(slot)
            destDir.listFiles()?.forEach { it.delete() }
            val storedName = if (slot == DiffusionModelSlot.FACE_SWAPPER &&
                normOrigName.contains("hyperswap", ignoreCase = true)
            ) {
                "hyperswap_256.onnx"
            } else {
                "model.onnx"
            }
            val destOnnx = File(destDir, storedName)
            val onnxExpectedLen = match.doc.length()
            ensureEnoughFreeSpace(onnxExpectedLen)
            copyUriToFile(match.doc.uri, destOnnx, onnxExpectedLen) { copied, total ->
                onProgress(formatCopyProgress(labelName, copied, total))
            }

            if (slot == DiffusionModelSlot.FACE_SWAPPER) {
                File(modelsRootDir, "inswapper_emap_512x512.bin").delete()
            }

            // Inspect the imported .onnx protobuf for exact external weight file locations
            val expectedLocations = extractExternalDataLocations(destOnnx)
            val copiedCompanionNames = mutableListOf<String>()

            if (expectedLocations.isNotEmpty()) {
                for (expectedLoc in expectedLocations) {
                    val siblingCandidates = allFiles.filter { other ->
                        val otherRaw = other.doc.name.orEmpty()
                        val otherNorm = normalizeBrowserDuplicateName(otherRaw)
                        otherRaw != origName && isCompanionWeightFileName(otherNorm)
                    }
                    // Prefer companion in same subfolder, else fallback to unique match in tree
                    val matchedCompanion = siblingCandidates.firstOrNull { other ->
                        other.parentName == match.parentName &&
                            normalizeBrowserDuplicateName(other.doc.name.orEmpty())
                                .equals(expectedLoc, ignoreCase = true)
                    } ?: siblingCandidates.firstOrNull { other ->
                        other.parentName == match.parentName &&
                            shouldCopyCompanionWeight(
                                onnxFileName = normOrigName,
                                onnxSizeBytes = destOnnx.length(),
                                companionFileName = normalizeBrowserDuplicateName(other.doc.name.orEmpty()),
                                onnxFileForRefScan = destOnnx
                            )
                    } ?: siblingCandidates.singleOrNull { other ->
                        normalizeBrowserDuplicateName(other.doc.name.orEmpty())
                            .equals(expectedLoc, ignoreCase = true)
                    }

                    if (matchedCompanion != null) {
                        val compDest = File(destDir, expectedLoc)
                        val compTotal = matchedCompanion.doc.length()
                        ensureEnoughFreeSpace(compTotal)
                        onProgress("Copying $expectedLoc (external weights)...")
                        copyUriToFile(matchedCompanion.doc.uri, compDest, compTotal) { copied, total ->
                            onProgress(formatCopyProgress(expectedLoc, copied, total))
                        }
                        copiedCompanionNames.add(expectedLoc)
                    }
                }
            } else {
                // Fallback for stem-matched external data if protobuf header wasn't standard
                allFiles.filter { other ->
                    val otherRaw = other.doc.name.orEmpty()
                    val otherNorm = normalizeBrowserDuplicateName(otherRaw)
                    other.parentName == match.parentName &&
                        otherRaw != origName &&
                        isCompanionWeightFileName(otherNorm) &&
                        shouldCopyCompanionWeight(
                            onnxFileName = normOrigName,
                            onnxSizeBytes = destOnnx.length(),
                            companionFileName = otherNorm,
                            onnxFileForRefScan = destOnnx
                        )
                }.forEach { companion ->
                    val compName = normalizeBrowserDuplicateName(companion.doc.name ?: "model.onnx_data")
                    val compDest = File(destDir, compName)
                    val compTotal = companion.doc.length()
                    ensureEnoughFreeSpace(compTotal)
                    onProgress("Copying $compName (external weights)...")
                    copyUriToFile(companion.doc.uri, compDest, compTotal) { copied, total ->
                        onProgress(formatCopyProgress(compName, copied, total))
                    }
                    copiedCompanionNames.add(compName)
                }
            }

            val labelToSave = if (copiedCompanionNames.isNotEmpty() && origName.isNotBlank()) {
                "$origName + ${copiedCompanionNames.joinToString(" + ")}"
            } else {
                origName
            }
            if (labelToSave.isNotBlank()) {
                File(destDir, "source_name.meta").writeText(labelToSave)
            }

            if (destOnnx.exists() && destOnnx.length() > 1024L && findMissingCompanionFiles(destOnnx).isEmpty()) {
                importedSlots.add(slot)
            }
        }

        // Also copy tokenizer / scheduler config files if present in the selected folder
        val tokDestDir = File(modelsRootDir, "tokenizer").apply { if (!exists()) mkdirs() }
        val configNames = setOf(
            "vocab.json",
            "merges.txt",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "scheduler_config.json",
            "preprocessor_config.json"
        )
        allFiles.forEach { cfg ->
            val normCfgName = normalizeBrowserDuplicateName(cfg.doc.name.orEmpty()).lowercase(Locale.US)
            if (normCfgName in configNames) {
                val dest = File(tokDestDir, normCfgName)
                copyUriToFile(cfg.doc.uri, dest)
            }
        }

        importedSlots
    }

    fun deleteSlotModel(slot: DiffusionModelSlot) {
        activeSessions.remove(slot)?.runCatching { close() }
        slotDir(slot).deleteRecursively()
        File(modelsRootDir, slot.canonicalFileName).delete()
        if (slot == DiffusionModelSlot.FACE_SWAPPER) {
            File(modelsRootDir, "inswapper_emap_512x512.bin").delete()
        }
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

    private fun queryUriLength(uri: Uri): Long = runCatching {
        context.contentResolver.query(uri, arrayOf(OpenableColumns.SIZE), null, null, null)?.use { c ->
            if (c.moveToFirst()) {
                val idx = c.getColumnIndex(OpenableColumns.SIZE)
                if (idx >= 0 && !c.isNull(idx)) c.getLong(idx) else -1L
            } else {
                -1L
            }
        } ?: -1L
    }.getOrDefault(-1L)

    private fun ensureEnoughFreeSpace(requiredBytes: Long) {
        if (requiredBytes <= 0L) return
        val freeBytes = runCatching { StatFs(context.filesDir.absolutePath).availableBytes }.getOrDefault(Long.MAX_VALUE)
        val safetyReserve = 150L * 1024L * 1024L
        if (freeBytes < requiredBytes + safetyReserve) {
            error(
                "Not enough storage space: need ${formatBytes(requiredBytes)}, available ${formatBytes(freeBytes)}."
            )
        }
    }

    private fun copyUriToFile(
        uri: Uri,
        dest: File,
        expectedTotalBytes: Long = -1L,
        onBytesCopied: ((Long, Long) -> Unit)? = null
    ) {
        context.contentResolver.openInputStream(uri)?.use { input ->
            FileOutputStream(dest).use { output ->
                val buffer = ByteArray(512 * 1024)
                var totalCopied = 0L
                var lastReportMs = 0L
                while (true) {
                    val read = input.read(buffer)
                    if (read <= 0) break
                    output.write(buffer, 0, read)
                    totalCopied += read
                    if (onBytesCopied != null) {
                        val now = System.currentTimeMillis()
                        if (now - lastReportMs >= 250L) {
                            lastReportMs = now
                            onBytesCopied(totalCopied, expectedTotalBytes)
                        }
                    }
                }
                output.flush()
                onBytesCopied?.invoke(totalCopied, expectedTotalBytes)
            }
        } ?: error("Unable to read selected file.")
    }

    private fun formatCopyProgress(fileName: String, copiedBytes: Long, totalBytes: Long): String {
        return if (totalBytes > 0L) {
            val pct = ((copiedBytes.toDouble() / totalBytes.toDouble()) * 100.0).toInt().coerceIn(0, 100)
            "Copying $fileName • ${formatBytes(copiedBytes)} / ${formatBytes(totalBytes)} ($pct%)"
        } else {
            "Copying $fileName • ${formatBytes(copiedBytes)}..."
        }
    }

    private fun calculateDirectorySize(dir: File?): Long {
        if (dir == null || !dir.exists()) return 0L
        if (dir.isFile) return dir.length()
        return dir.walkTopDown().filter { it.isFile }.sumOf { it.length() }
    }

    data class CandidateModelFile(
        val index: Int,
        val fileName: String,
        val parentNameLower: String
    ) {
        val fileNameLower: String
            get() = fileName.lowercase(Locale.US)
    }

    companion object {
        private const val MAX_EXTERNAL_GRAPH_SHELL_BYTES = 15L * 1024L * 1024L
        private const val MAX_PROTO_HEADER_SCAN_BYTES = 48L * 1024L * 1024L
        const val MIN_VALID_COMPANION_BYTES = 64L * 1024L

        private data class FileScanCacheKey(
            val path: String,
            val lastModified: Long,
            val length: Long
        )

        private val externalLocationsCache = ConcurrentHashMap<FileScanCacheKey, List<String>>()

        private val PROTO_LOCATION_KEY_PREFIX = byteArrayOf(
            0x0A, 0x08,
            'l'.code.toByte(), 'o'.code.toByte(), 'c'.code.toByte(), 'a'.code.toByte(),
            't'.code.toByte(), 'i'.code.toByte(), 'o'.code.toByte(), 'n'.code.toByte(),
            0x12
        )

        /**
         * Normalizes browser duplicate suffixes from Android Download Manager:
         * e.g. "weights (1).pb" -> "weights.pb", "model.onnx (1).data" -> "model.onnx.data",
         * "model (2).onnx.data" -> "model.onnx.data", "unet (1).onnx" -> "unet.onnx".
         */
        fun normalizeBrowserDuplicateName(fileName: String): String {
            var cleaned = fileName.trim()
            if (cleaned.isEmpty()) return cleaned
            cleaned = cleaned.replace(
                Regex("""\s*\(\d+\)(?=(\.onnx\.data|\.onnx_data|\.onnx|\.pb|\.data|\.json|\.txt)$)""", RegexOption.IGNORE_CASE),
                ""
            )
            cleaned = cleaned.replace(
                Regex("""\s*\(\d+\)$"""),
                ""
            )
            return cleaned.trim()
        }

        /**
         * Parses ONNX TensorProto.external_data StringStringEntryProto entries ("location" -> filename)
         * to discover the exact external weight file(s) required by [onnxFile] (e.g. "weights.pb",
         * "model.onnx.data", or "model.onnx_data").
         */
        fun extractExternalDataLocations(onnxFile: File): List<String> {
            if (!onnxFile.exists() || !onnxFile.isFile || onnxFile.length() <= 0L) return emptyList()
            val cacheKey = FileScanCacheKey(onnxFile.absolutePath, onnxFile.lastModified(), onnxFile.length())
            externalLocationsCache[cacheKey]?.let { return it }

            val discovered = LinkedHashSet<String>()
            val buf = ByteArray(64 * 1024)
            var carry = ByteArray(0)
            var scannedBytes = 0L

            runCatching {
                onnxFile.inputStream().use { input ->
                    while (scannedBytes < MAX_PROTO_HEADER_SCAN_BYTES) {
                        val read = input.read(buf)
                        if (read <= 0) break
                        scannedBytes += read
                        val window = ByteArray(carry.size + read)
                        System.arraycopy(carry, 0, window, 0, carry.size)
                        System.arraycopy(buf, 0, window, carry.size, read)

                        scanProtobufExternalLocations(window, discovered)

                        val keepLen = minOf(window.size, 160)
                        carry = window.copyOfRange(window.size - keepLen, window.size)
                    }
                }
            }

            val result = discovered.toList()
            externalLocationsCache[cacheKey] = result
            return result
        }

        private fun scanProtobufExternalLocations(window: ByteArray, out: MutableSet<String>) {
            val prefix = PROTO_LOCATION_KEY_PREFIX
            val maxIdx = window.size - (prefix.size + 2)
            if (maxIdx < 0) return

            var i = 0
            while (i <= maxIdx) {
                var matched = true
                for (p in prefix.indices) {
                    if (window[i + p] != prefix[p]) {
                        matched = false
                        break
                    }
                }
                if (matched) {
                    val strLen = window[i + prefix.size].toInt() and 0xFF
                    val strStart = i + prefix.size + 1
                    if (strLen in 3..120 && strStart + strLen <= window.size) {
                        val candidate = String(window, strStart, strLen, Charsets.UTF_8).trim()
                        if (isValidExternalLocationName(candidate)) {
                            out.add(candidate)
                        }
                        i = strStart + strLen
                        continue
                    }
                }
                i++
            }
        }

        private fun isValidExternalLocationName(name: String): Boolean {
            if (name.length < 3 || name.length > 120) return false
            if (name.startsWith("/") || name.startsWith("\\") || name.contains("..")) return false
            if (!name.contains('.')) return false
            return name.all { ch ->
                ch in 'a'..'z' || ch in 'A'..'Z' || ch in '0'..'9' ||
                    ch == '.' || ch == '_' || ch == '-' || ch == ' '
            }
        }

        fun findMissingCompanionFiles(onnxFile: File): List<String> {
            val required = extractExternalDataLocations(onnxFile)
            if (required.isEmpty()) return emptyList()
            val dir = onnxFile.parentFile ?: return required
            val filesInDir = dir.listFiles()?.filter { it.isFile }.orEmpty()

            return required.filter { reqName ->
                val exact = File(dir, reqName)
                if (exact.exists() && exact.length() >= MIN_VALID_COMPANION_BYTES) {
                    false
                } else {
                    val caseMatch = filesInDir.firstOrNull {
                        it.name.equals(reqName, ignoreCase = true) && it.length() >= MIN_VALID_COMPANION_BYTES
                    }
                    if (caseMatch != null) {
                        runCatching { caseMatch.renameTo(exact) }
                        false
                    } else {
                        true
                    }
                }
            }
        }

        fun matchesSlot(slot: DiffusionModelSlot, fileNameLower: String, parentNameLower: String): Boolean {
            val normLower = normalizeBrowserDuplicateName(fileNameLower).lowercase(Locale.US)
            if (!normLower.endsWith(".onnx")) return false
            if (normLower == slot.canonicalFileName) return true
            if (slot.matchKeywords.any { kw -> normLower.contains(kw) }) return true
            if (parentNameLower == slot.folderHint && normLower == "model.onnx") return true
            return false
        }

        /**
         * Resolves all slots in 2 passes:
         * Pass 1: Explicit filename or parent-subfolder matches.
         * Pass 2: Flat-folder fallback — if UNET wasn't explicitly named unet.onnx, an unclaimed
         *         model.onnx alongside weights.pb / .onnx_data or sibling Diffusion models is UNET!
         */
        fun resolveFolderSlotMatches(
            candidates: List<CandidateModelFile>,
            allFileNamesByParent: Map<String, List<String>>
        ): Map<DiffusionModelSlot, Int> {
            val matched = LinkedHashMap<DiffusionModelSlot, Int>()
            val claimedIndices = HashSet<Int>()

            for (slot in DiffusionModelSlot.entries) {
                val found = candidates.firstOrNull { cand ->
                    cand.index !in claimedIndices &&
                        matchesSlot(slot, cand.fileNameLower, cand.parentNameLower)
                }
                if (found != null) {
                    matched[slot] = found.index
                    claimedIndices.add(found.index)
                }
            }

            if (DiffusionModelSlot.UNET !in matched) {
                val coreMatchedParents = matched.entries
                    .filter { !it.key.isFaceSwapSlot }
                    .mapNotNull { entry -> candidates.firstOrNull { it.index == entry.value }?.parentNameLower }
                    .toSet()

                val fallbackUnet = candidates.firstOrNull { cand ->
                    val normCand = normalizeBrowserDuplicateName(cand.fileNameLower)
                    if (cand.index in claimedIndices || normCand != "model.onnx") {
                        false
                    } else {
                        val siblings = allFileNamesByParent[cand.parentNameLower].orEmpty()
                        val hasExternalWeights = siblings.any { isCompanionWeightFileName(it) }
                        val sharesDiffusionFolder = cand.parentNameLower in coreMatchedParents
                        hasExternalWeights || sharesDiffusionFolder
                    }
                }
                if (fallbackUnet != null) {
                    matched[DiffusionModelSlot.UNET] = fallbackUnet.index
                    claimedIndices.add(fallbackUnet.index)
                }
            }

            return matched
        }

        fun isCompanionWeightFileName(fileName: String): Boolean {
            val lower = normalizeBrowserDuplicateName(fileName).lowercase(Locale.US)
            return lower.endsWith(".onnx_data") ||
                lower.endsWith(".onnx.data") ||
                lower == "weights.pb" ||
                lower.endsWith("_weights.pb")
        }

        /**
         * Prevents duplicating a 3.44 GB weights.pb across self-contained .onnx slots in flat folders,
         * while always copying external weights explicitly referenced inside an .onnx protobuf.
         */
        fun shouldCopyCompanionWeight(
            onnxFileName: String,
            onnxSizeBytes: Long,
            companionFileName: String,
            onnxFileForRefScan: File? = null
        ): Boolean {
            val onnxLower = normalizeBrowserDuplicateName(onnxFileName).lowercase(Locale.US)
            val onnxStem = onnxLower.removeSuffix(".onnx")
            val compLower = normalizeBrowserDuplicateName(companionFileName).lowercase(Locale.US)

            if (onnxFileForRefScan != null && onnxFileForRefScan.exists()) {
                val protoLocations = extractExternalDataLocations(onnxFileForRefScan)
                if (protoLocations.isNotEmpty()) {
                    return protoLocations.any { it.equals(compLower, ignoreCase = true) } ||
                        compLower == "${onnxLower}_data" ||
                        compLower == "${onnxLower}.data" ||
                        compLower == "${onnxStem}.onnx_data" ||
                        compLower == "${onnxStem}.onnx.data" ||
                        compLower == "${onnxStem}_weights.pb"
                }
            }

            if (compLower == "${onnxLower}_data" ||
                compLower == "${onnxLower}.data" ||
                compLower == "${onnxStem}.onnx_data" ||
                compLower == "${onnxStem}.onnx.data" ||
                compLower == "${onnxStem}_weights.pb"
            ) {
                return true
            }

            // Self-contained ONNX models (> 15 MB without protobuf external_data entries) embed
            // their own tensor initializers and never use a generic flat-folder weights.pb.
            if (onnxSizeBytes > MAX_EXTERNAL_GRAPH_SHELL_BYTES || onnxSizeBytes <= 0L) {
                return false
            }

            if (onnxFileForRefScan != null && onnxFileForRefScan.exists()) {
                return fileReferencesExternalData(onnxFileForRefScan, compLower)
            }

            return true
        }

        fun fileReferencesExternalData(onnxFile: File, companionFileName: String): Boolean {
            val needle = normalizeBrowserDuplicateName(companionFileName)
                .lowercase(Locale.US)
                .toByteArray(Charsets.UTF_8)
            if (needle.isEmpty()) return false
            val buf = ByteArray(64 * 1024)
            var carry = ByteArray(0)

            return runCatching {
                onnxFile.inputStream().use { input ->
                    while (true) {
                        val read = input.read(buf)
                        if (read <= 0) break
                        val window = ByteArray(carry.size + read)
                        System.arraycopy(carry, 0, window, 0, carry.size)
                        System.arraycopy(buf, 0, window, carry.size, read)
                        if (containsAsciiIgnoreCase(window, needle)) {
                            return@use true
                        }
                        val keepLen = minOf(window.size, needle.size)
                        carry = window.copyOfRange(window.size - keepLen, window.size)
                    }
                    false
                }
            }.getOrDefault(false)
        }

        private fun containsAsciiIgnoreCase(haystack: ByteArray, needleLower: ByteArray): Boolean {
            val maxStart = haystack.size - needleLower.size
            if (maxStart < 0) return false
            outer@ for (i in 0..maxStart) {
                for (j in needleLower.indices) {
                    var b = haystack[i + j].toInt() and 0xFF
                    if (b in 65..90) b += 32
                    if (b != (needleLower[j].toInt() and 0xFF)) continue@outer
                }
                return true
            }
            return false
        }

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
