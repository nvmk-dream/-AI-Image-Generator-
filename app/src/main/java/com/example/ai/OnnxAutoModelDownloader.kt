package com.example.ai

import com.example.data.ActiveDiffusionEngine
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean

data class RemoteModelFileSpec(
    val targetFileName: String,
    val url: String,
    val expectedSizeBytes: Long,
    val isOnnxBinary: Boolean,
    val minValidBytes: Long
)

data class AutoDownloadSlotSpec(
    val slot: DedicatedModelSlot,
    val shortCode: String,
    val displayOrder: Int,
    val executionPriority: Int,
    val titleEn: String,
    val titleTa: String,
    val files: List<RemoteModelFileSpec>
) {
    val totalExpectedBytes: Long
        get() = files.sumOf { it.expectedSizeBytes }
}

data class AutoDownloadProgressState(
    val isDownloading: Boolean = false,
    val currentSlot: DedicatedModelSlot? = null,
    val currentSlotShortCode: String = "",
    val completedSlotsCount: Int = 0,
    val totalSelectedSlotsCount: Int = 0,
    val currentFileDownloadedBytes: Long = 0L,
    val currentFileTotalBytes: Long = 0L,
    val overallDownloadedBytes: Long = 0L,
    val overallTotalBytes: Long = 0L,
    val speedBytesPerSec: Long = 0L,
    val statusTextEn: String = "",
    val statusTextTa: String = "",
    val finishedSummaryEn: String? = null,
    val finishedSummaryTa: String? = null,
    val errorTextEn: String? = null,
    val errorTextTa: String? = null
) {
    val overallPercent: Int
        get() = if (overallTotalBytes > 0L) {
            ((overallDownloadedBytes.toDouble() / overallTotalBytes.toDouble()) * 100.0)
                .toInt()
                .coerceIn(0, 100)
        } else {
            0
        }

    val currentFilePercent: Int
        get() = if (currentFileTotalBytes > 0L) {
            ((currentFileDownloadedBytes.toDouble() / currentFileTotalBytes.toDouble()) * 100.0)
                .toInt()
                .coerceIn(0, 100)
        } else {
            0
        }
}

data class AutoDownloadBatchResult(
    val downloadedSlots: List<DedicatedModelSlot>,
    val skippedAlreadyReadySlots: List<DedicatedModelSlot>,
    val pausedLowStorageSlot: DedicatedModelSlot? = null,
    val folderDisplayPath: String
)

class OnnxAutoModelDownloader(
    private val modelManager: OnnxModelManager,
    private val connectionFactory: (URL) -> HttpURLConnection = { url ->
        url.openConnection() as HttpURLConnection
    }
) {
    private val cancelRequested = AtomicBoolean(false)

    @Volatile
    private var activeConnection: HttpURLConnection? = null

    fun requestCancel() {
        cancelRequested.set(true)
        runCatching { activeConnection?.disconnect() }
    }

    suspend fun downloadSelectedCoreSlots(
        selectedSlots: Set<DedicatedModelSlot>,
        onProgress: (AutoDownloadProgressState) -> Unit,
        onSlotCompleted: suspend (DedicatedModelSlot) -> Unit
    ): Result<AutoDownloadBatchResult> = runCatching {
        cancelRequested.set(false)
        modelManager.ensurePhoneModelFoldersCreated()
        val folderPath = modelManager.phoneModelsDisplayPath

        val specsToRun = CORE_SLOT_SPECS
            .filter { it.slot in selectedSlots }
            .sortedBy { it.executionPriority }

        if (specsToRun.isEmpty()) {
            return@runCatching AutoDownloadBatchResult(
                downloadedSlots = emptyList(),
                skippedAlreadyReadySlots = emptyList(),
                pausedLowStorageSlot = null,
                folderDisplayPath = folderPath
            )
        }

        val overallExpectedTotal = specsToRun.sumOf { it.totalExpectedBytes }
        var completedBytesBeforeCurrentFile = 0L
        var completedSlotCounter = 0
        val newlyDownloaded = mutableListOf<DedicatedModelSlot>()
        val alreadyReady = mutableListOf<DedicatedModelSlot>()
        var pausedLowSpaceSlot: DedicatedModelSlot? = null

        for (spec in specsToRun) {
            checkNotCancelled()

            if (modelManager.isDedicatedSlotReady(spec.slot)) {
                alreadyReady.add(spec.slot)
                completedSlotCounter++
                completedBytesBeforeCurrentFile += spec.totalExpectedBytes
                onProgress(
                    AutoDownloadProgressState(
                        isDownloading = true,
                        currentSlot = spec.slot,
                        currentSlotShortCode = spec.shortCode,
                        completedSlotsCount = completedSlotCounter,
                        totalSelectedSlotsCount = specsToRun.size,
                        currentFileDownloadedBytes = spec.totalExpectedBytes,
                        currentFileTotalBytes = spec.totalExpectedBytes,
                        overallDownloadedBytes = completedBytesBeforeCurrentFile,
                        overallTotalBytes = overallExpectedTotal,
                        statusTextEn = "Slot ${spec.shortCode} already ready ✓",
                        statusTextTa = "Slot ${spec.shortCode} ஏற்கனவே தயாராக உள்ளது ✓"
                    )
                )
                continue
            }

            val slotDir = modelManager.targetDownloadFolder(spec.slot).apply { if (!exists()) mkdirs() }

            // Calculate remaining bytes needed on disk considering any existing .download_part resume file
            val existingPartialBytes = spec.files.sumOf { fileSpec ->
                File(slotDir, "${fileSpec.targetFileName}.download_part")
                    .takeIf { it.exists() }?.length() ?: 0L
            }
            val remainingSlotBytes = (spec.totalExpectedBytes - existingPartialBytes).coerceAtLeast(0L)

            if (!modelManager.hasEnoughFreeSpace(remainingSlotBytes)) {
                val freeStr = OnnxModelManager.formatBytes(modelManager.availableStorageBytes())
                val needStr = OnnxModelManager.formatBytes(remainingSlotBytes)
                // If Fast LCM core slots are already ready and only huge 3.44 GB Slot 2B ran out of phone space,
                // pause cleanly without failing the completed Fast LCM setup!
                if (spec.slot == DedicatedModelSlot.STANDARD_SD_WEIGHTS &&
                    (newlyDownloaded.isNotEmpty() || alreadyReady.isNotEmpty())
                ) {
                    pausedLowSpaceSlot = spec.slot
                    break
                } else {
                    error(
                        "போனில் போதிய இடமில்லை (Slot ${spec.shortCode}: தேவை $needStr, இருப்பது $freeStr). Not enough phone storage."
                    )
                }
            }

            for (fileSpec in spec.files) {
                checkNotCancelled()
                val destFile = File(slotDir, fileSpec.targetFileName)
                if (destFile.exists() && destFile.length() >= fileSpec.minValidBytes &&
                    (!fileSpec.isOnnxBinary || OnnxProtoCodec.inspectModelFileHeader(destFile) == null)
                ) {
                    completedBytesBeforeCurrentFile += fileSpec.expectedSizeBytes
                    continue
                }

                downloadSingleFileWithResumeAndRetry(
                    spec = spec,
                    fileSpec = fileSpec,
                    slotDir = slotDir,
                    completedSlotsCount = completedSlotCounter,
                    totalSlotsCount = specsToRun.size,
                    completedBytesBeforeFile = completedBytesBeforeCurrentFile,
                    overallExpectedTotal = overallExpectedTotal,
                    onProgress = onProgress
                )
                completedBytesBeforeCurrentFile += fileSpec.expectedSizeBytes
            }

            finalizeDownloadedSlotMetadataAndLinks(spec.slot, slotDir)
            newlyDownloaded.add(spec.slot)
            completedSlotCounter++
            onSlotCompleted(spec.slot)
        }

        // Ensure pair linking runs cleanly at end of batch for both engines
        OnnxModelManager.linkEnginePairFiles(modelManager.engineFolder(ActiveDiffusionEngine.FAST_LCM))
        OnnxModelManager.linkEnginePairFiles(modelManager.engineFolder(ActiveDiffusionEngine.STANDARD_SD))

        AutoDownloadBatchResult(
            downloadedSlots = newlyDownloaded,
            skippedAlreadyReadySlots = alreadyReady,
            pausedLowStorageSlot = pausedLowSpaceSlot,
            folderDisplayPath = folderPath
        )
    }

    private suspend fun downloadSingleFileWithResumeAndRetry(
        spec: AutoDownloadSlotSpec,
        fileSpec: RemoteModelFileSpec,
        slotDir: File,
        completedSlotsCount: Int,
        totalSlotsCount: Int,
        completedBytesBeforeFile: Long,
        overallExpectedTotal: Long,
        onProgress: (AutoDownloadProgressState) -> Unit
    ) {
        val partFile = File(slotDir, "${fileSpec.targetFileName}.download_part")
        val finalFile = File(slotDir, fileSpec.targetFileName)

        var attempt = 0
        var lastException: Throwable? = null

        while (attempt < MAX_RETRY_ATTEMPTS) {
            checkNotCancelled()
            attempt++
            try {
                streamHttpToPartFile(
                    spec = spec,
                    fileSpec = fileSpec,
                    partFile = partFile,
                    completedSlotsCount = completedSlotsCount,
                    totalSlotsCount = totalSlotsCount,
                    completedBytesBeforeFile = completedBytesBeforeFile,
                    overallExpectedTotal = overallExpectedTotal,
                    onProgress = onProgress
                )

                if (partFile.length() < fileSpec.minValidBytes) {
                    val badLen = partFile.length()
                    partFile.delete()
                    throw IOException(
                        "Downloaded file for Slot ${spec.shortCode} is incomplete (${OnnxModelManager.formatBytes(badLen)})."
                    )
                }

                if (fileSpec.isOnnxBinary) {
                    val headerProblem = OnnxProtoCodec.inspectModelFileHeader(partFile)
                    if (headerProblem != null) {
                        partFile.delete()
                        throw IOException("Slot ${spec.shortCode} header validation failed: $headerProblem")
                    }
                } else {
                    // Quick sanity check that vocab.json / merges.txt isn't an HTML error page
                    val headText = runCatching {
                        partFile.inputStream().use { ins ->
                            val buf = ByteArray(minOf(256, partFile.length().toInt()))
                            val r = ins.read(buf)
                            if (r > 0) String(buf, 0, r, Charsets.UTF_8).trimStart().lowercase(Locale.US) else ""
                        }
                    }.getOrDefault("")
                    if (headText.startsWith("<!doctype html") || headText.startsWith("<html")) {
                        partFile.delete()
                        throw IOException("Slot ${spec.shortCode} received HTML login/error page instead of tokenizer file.")
                    }
                }

                // Clean up previous slot target before atomic promotion
                prepareSlotDirectoryBeforePromotion(spec.slot, slotDir, finalFile)

                if (finalFile.exists()) finalFile.delete()
                if (!partFile.renameTo(finalFile)) {
                    partFile.copyTo(finalFile, overwrite = true)
                    partFile.delete()
                }
                return
            } catch (ce: CancellationException) {
                throw ce
            } catch (t: Throwable) {
                if (cancelRequested.get()) {
                    throw CancellationException("Auto-download cancelled by user.").apply { initCause(t) }
                }
                lastException = t
                if (attempt < MAX_RETRY_ATTEMPTS) {
                    delay(1200L * attempt)
                }
            }
        }

        throw lastException ?: IOException("Failed to download Slot ${spec.shortCode}.")
    }

    private suspend fun streamHttpToPartFile(
        spec: AutoDownloadSlotSpec,
        fileSpec: RemoteModelFileSpec,
        partFile: File,
        completedSlotsCount: Int,
        totalSlotsCount: Int,
        completedBytesBeforeFile: Long,
        overallExpectedTotal: Long,
        onProgress: (AutoDownloadProgressState) -> Unit
    ) {
        val resumeOffset = if (partFile.exists()) partFile.length().coerceAtLeast(0L) else 0L
        val (conn, responseCode) = openFollowingRedirects(fileSpec.url, resumeOffset)
        activeConnection = conn

        try {
            if (responseCode == HttpURLConnection.HTTP_BAD_REQUEST || responseCode == 416) {
                // 416 Range Not Satisfiable -> check if partFile is already complete, else reset and retry
                if (partFile.exists() && partFile.length() >= fileSpec.expectedSizeBytes && fileSpec.expectedSizeBytes > 0L) {
                    return
                }
                partFile.delete()
                throw IOException("HTTP $responseCode while resuming ${fileSpec.targetFileName}; resetting partial file.")
            }

            if (responseCode != HttpURLConnection.HTTP_OK && responseCode != HttpURLConnection.HTTP_PARTIAL) {
                throw IOException("HTTP $responseCode when downloading Slot ${spec.shortCode} (${fileSpec.targetFileName}).")
            }

            val isAppending = (responseCode == HttpURLConnection.HTTP_PARTIAL && resumeOffset > 0L)
            val startingBytes = if (isAppending) resumeOffset else 0L
            val contentLength = conn.contentLengthLong
            val totalFileBytes = when {
                contentLength > 0L && isAppending -> startingBytes + contentLength
                contentLength > 0L -> contentLength
                else -> fileSpec.expectedSizeBytes
            }

            var currentFileBytes = startingBytes
            var speedWindowStartMs = System.currentTimeMillis()
            var speedWindowBytes = 0L
            var currentSpeedBps = 0L
            var lastUiEmitMs = 0L

            emitFileProgress(
                spec = spec,
                fileSpec = fileSpec,
                completedSlotsCount = completedSlotsCount,
                totalSlotsCount = totalSlotsCount,
                currentFileBytes = currentFileBytes,
                totalFileBytes = totalFileBytes,
                completedBytesBeforeFile = completedBytesBeforeFile,
                overallExpectedTotal = overallExpectedTotal,
                speedBps = 0L,
                onProgress = onProgress
            )

            conn.inputStream.use { input ->
                FileOutputStream(partFile, isAppending).use { output ->
                    val buffer = ByteArray(256 * 1024)
                    while (true) {
                        checkNotCancelled()
                        val read = input.read(buffer)
                        if (read <= 0) break
                        output.write(buffer, 0, read)
                        currentFileBytes += read
                        speedWindowBytes += read

                        val now = System.currentTimeMillis()
                        val elapsedSpeedMs = now - speedWindowStartMs
                        if (elapsedSpeedMs >= 800L) {
                            currentSpeedBps = (speedWindowBytes * 1000L) / elapsedSpeedMs.coerceAtLeast(1L)
                            speedWindowStartMs = now
                            speedWindowBytes = 0L
                        }
                        if (now - lastUiEmitMs >= 250L) {
                            lastUiEmitMs = now
                            emitFileProgress(
                                spec = spec,
                                fileSpec = fileSpec,
                                completedSlotsCount = completedSlotsCount,
                                totalSlotsCount = totalSlotsCount,
                                currentFileBytes = currentFileBytes,
                                totalFileBytes = totalFileBytes,
                                completedBytesBeforeFile = completedBytesBeforeFile,
                                overallExpectedTotal = overallExpectedTotal,
                                speedBps = currentSpeedBps,
                                onProgress = onProgress
                            )
                        }
                    }
                    output.flush()
                }
            }

            emitFileProgress(
                spec = spec,
                fileSpec = fileSpec,
                completedSlotsCount = completedSlotsCount,
                totalSlotsCount = totalSlotsCount,
                currentFileBytes = currentFileBytes,
                totalFileBytes = totalFileBytes.coerceAtLeast(currentFileBytes),
                completedBytesBeforeFile = completedBytesBeforeFile,
                overallExpectedTotal = overallExpectedTotal,
                speedBps = currentSpeedBps,
                onProgress = onProgress
            )
        } finally {
            if (activeConnection === conn) {
                activeConnection = null
            }
            runCatching { conn.disconnect() }
        }
    }

    private fun emitFileProgress(
        spec: AutoDownloadSlotSpec,
        fileSpec: RemoteModelFileSpec,
        completedSlotsCount: Int,
        totalSlotsCount: Int,
        currentFileBytes: Long,
        totalFileBytes: Long,
        completedBytesBeforeFile: Long,
        overallExpectedTotal: Long,
        speedBps: Long,
        onProgress: (AutoDownloadProgressState) -> Unit
    ) {
        val curStr = OnnxModelManager.formatBytes(currentFileBytes)
        val totStr = OnnxModelManager.formatBytes(totalFileBytes)
        val pct = if (totalFileBytes > 0L) {
            ((currentFileBytes.toDouble() / totalFileBytes.toDouble()) * 100.0).toInt().coerceIn(0, 100)
        } else 0
        val speedSuffix = if (speedBps > 0L) " • ${OnnxModelManager.formatBytes(speedBps)}/s" else ""
        val stepIndex = (completedSlotsCount + 1).coerceAtMost(totalSlotsCount)

        onProgress(
            AutoDownloadProgressState(
                isDownloading = true,
                currentSlot = spec.slot,
                currentSlotShortCode = spec.shortCode,
                completedSlotsCount = completedSlotsCount,
                totalSelectedSlotsCount = totalSlotsCount,
                currentFileDownloadedBytes = currentFileBytes,
                currentFileTotalBytes = totalFileBytes,
                overallDownloadedBytes = (completedBytesBeforeFile + currentFileBytes)
                    .coerceAtMost(overallExpectedTotal.coerceAtLeast(completedBytesBeforeFile + currentFileBytes)),
                overallTotalBytes = overallExpectedTotal,
                speedBytesPerSec = speedBps,
                statusTextEn = "Downloading Slot ${spec.shortCode} ($stepIndex/$totalSlotsCount): ${fileSpec.targetFileName} • $curStr / $totStr ($pct%)$speedSuffix",
                statusTextTa = "Slot ${spec.shortCode} ($stepIndex/$totalSlotsCount) பதிவிறக்கம்: ${fileSpec.targetFileName} • $curStr / $totStr ($pct%)$speedSuffix"
            )
        )
    }

    private fun openFollowingRedirects(initialUrl: String, resumeOffset: Long): Pair<HttpURLConnection, Int> {
        var currentUrl = URL(initialUrl)
        repeat(MAX_REDIRECT_HOPS) {
            if (cancelRequested.get()) {
                throw CancellationException("Auto-download cancelled by user.")
            }
            val conn = connectionFactory(currentUrl)
            conn.instanceFollowRedirects = false
            conn.connectTimeout = 25_000
            conn.readTimeout = 45_000
            conn.requestMethod = "GET"
            conn.setRequestProperty("User-Agent", "OfflineImageGenerator-Android/2.1")
            conn.setRequestProperty("Accept", "*/*")
            if (resumeOffset > 0L) {
                conn.setRequestProperty("Range", "bytes=$resumeOffset-")
            }
            conn.connect()
            val code = conn.responseCode
            if (code in REDIRECT_CODES) {
                val loc = conn.getHeaderField("Location")
                conn.disconnect()
                if (loc.isNullOrBlank()) {
                    throw IOException("Redirect response HTTP $code missing Location header.")
                }
                currentUrl = URL(currentUrl, loc)
            } else {
                return conn to code
            }
        }
        throw IOException("Too many HTTP redirects while connecting to model server.")
    }

    private fun prepareSlotDirectoryBeforePromotion(
        slot: DedicatedModelSlot,
        slotDir: File,
        upcomingDestFile: File
    ) {
        when (slot) {
            DedicatedModelSlot.FAST_LCM_GRAPH,
            DedicatedModelSlot.STANDARD_SD_GRAPH -> {
                OnnxQuantGraphRepairer.clearGraphRepairMarkerOnly(slotDir)
            }
            DedicatedModelSlot.FAST_LCM_WEIGHTS,
            DedicatedModelSlot.STANDARD_SD_WEIGHTS -> {
                slotDir.listFiles()?.forEach { f ->
                    if (f.isFile &&
                        !f.name.equals("model.onnx", ignoreCase = true) &&
                        !f.name.equals("graph_source.meta", ignoreCase = true) &&
                        !f.name.endsWith(".download_part", ignoreCase = true) &&
                        f.absolutePath != upcomingDestFile.absolutePath
                    ) {
                        runCatching { f.delete() }
                    }
                }
                OnnxQuantGraphRepairer.clearRepairMarkers(slotDir)
            }
            DedicatedModelSlot.TOKENIZER -> {
                // Keep sibling vocab.json / merges.txt intact
            }
            else -> {
                slotDir.listFiles()?.forEach { f ->
                    if (f.isFile &&
                        !f.name.endsWith(".download_part", ignoreCase = true) &&
                        f.absolutePath != upcomingDestFile.absolutePath
                    ) {
                        runCatching { f.delete() }
                    }
                }
            }
        }
    }

    private fun finalizeDownloadedSlotMetadataAndLinks(slot: DedicatedModelSlot, slotDir: File) {
        when (slot) {
            DedicatedModelSlot.FAST_LCM_GRAPH,
            DedicatedModelSlot.STANDARD_SD_GRAPH -> {
                File(slotDir, "graph_source.meta").writeText("model.onnx")
                OnnxModelManager.linkEnginePairFiles(slotDir)
            }
            DedicatedModelSlot.FAST_LCM_WEIGHTS -> {
                File(slotDir, "weights_source.meta").writeText("model_int8.onnx.data")
                OnnxModelManager.linkEnginePairFiles(slotDir)
            }
            DedicatedModelSlot.STANDARD_SD_WEIGHTS -> {
                File(slotDir, "weights_source.meta").writeText("model.onnx_data")
                OnnxModelManager.linkEnginePairFiles(slotDir)
            }
            DedicatedModelSlot.TEXT_ENCODER -> {
                File(slotDir, "source_name.meta").writeText("text_encoder/model.onnx")
            }
            DedicatedModelSlot.VAE_DECODER -> {
                File(slotDir, "source_name.meta").writeText("vae_decoder/model.onnx")
            }
            DedicatedModelSlot.VAE_ENCODER -> {
                File(slotDir, "source_name.meta").writeText("vae_encoder/model.onnx")
            }
            DedicatedModelSlot.TOKENIZER -> {
                File(slotDir, "source_name.meta").writeText("vocab.json + merges.txt")
            }
            DedicatedModelSlot.FACE_AGING_ENGINE -> {
                File(slotDir, "source_name.meta").writeText("face_aging.onnx")
            }
        }
    }

    private suspend fun checkNotCancelled() {
        currentCoroutineContext().ensureActive()
        if (cancelRequested.get()) {
            throw CancellationException("Auto-download cancelled by user.")
        }
    }

    companion object {
        private const val MAX_REDIRECT_HOPS = 8
        private const val MAX_RETRY_ATTEMPTS = 3
        private val REDIRECT_CODES = setOf(301, 302, 303, 307, 308)

        /**
         * Strictly covers all 8 Core Slots (1A, 1B, 2A, 2B, 3, 4, 5, 6).
         *
         * Note on `executionPriority`:
         * Downloads Fast LCM + Shared Core (1A -> 1B -> 3 -> 4 -> 5 -> 6, ~1.69 GB total) FIRST so
         * offline image generation becomes 100% usable quickly, then downloads 2A -> 2B (~3.44 GB).
         */
        val CORE_SLOT_SPECS: List<AutoDownloadSlotSpec> = listOf(
            AutoDownloadSlotSpec(
                slot = DedicatedModelSlot.FAST_LCM_GRAPH,
                shortCode = "1A",
                displayOrder = 1,
                executionPriority = 1,
                titleEn = "Slot 1A: Fast LCM Graph (model.onnx)",
                titleTa = "Slot 1A: Fast LCM வரைபடம் (model.onnx)",
                files = listOf(
                    RemoteModelFileSpec(
                        targetFileName = "model.onnx",
                        url = "https://huggingface.co/latentdivergence/lcm-dreamshaper-int8-onnx/resolve/main/unet/model.onnx",
                        expectedSizeBytes = 3_049_494L,
                        isOnnxBinary = true,
                        minValidBytes = 500_000L
                    )
                )
            ),
            AutoDownloadSlotSpec(
                slot = DedicatedModelSlot.FAST_LCM_WEIGHTS,
                shortCode = "1B",
                displayOrder = 2,
                executionPriority = 2,
                titleEn = "Slot 1B: Fast LCM Weights (model_int8.onnx.data)",
                titleTa = "Slot 1B: Fast LCM எடை ஃபைல் (model_int8.onnx.data)",
                files = listOf(
                    RemoteModelFileSpec(
                        targetFileName = "model_int8.onnx.data",
                        url = "https://huggingface.co/latentdivergence/lcm-dreamshaper-int8-onnx/resolve/main/unet/model_int8.onnx.data",
                        expectedSizeBytes = 860_934_400L,
                        isOnnxBinary = false,
                        minValidBytes = 800_000_000L
                    )
                )
            ),
            AutoDownloadSlotSpec(
                slot = DedicatedModelSlot.STANDARD_SD_GRAPH,
                shortCode = "2A",
                displayOrder = 3,
                executionPriority = 7,
                titleEn = "Slot 2A: Standard SD Graph (model.onnx)",
                titleTa = "Slot 2A: Standard SD வரைபடம் (model.onnx)",
                files = listOf(
                    RemoteModelFileSpec(
                        targetFileName = "model.onnx",
                        url = "https://huggingface.co/TheyCallMeHex/LCM-Dreamshaper-V7-ONNX/resolve/main/unet/model.onnx",
                        expectedSizeBytes = 1_948_508L,
                        isOnnxBinary = true,
                        minValidBytes = 400_000L
                    )
                )
            ),
            AutoDownloadSlotSpec(
                slot = DedicatedModelSlot.STANDARD_SD_WEIGHTS,
                shortCode = "2B",
                displayOrder = 4,
                executionPriority = 8,
                titleEn = "Slot 2B: Standard SD Weights (model.onnx_data)",
                titleTa = "Slot 2B: Standard SD எடை ஃபைல் (~3.44 GB)",
                files = listOf(
                    RemoteModelFileSpec(
                        targetFileName = "model.onnx_data",
                        url = "https://huggingface.co/TheyCallMeHex/LCM-Dreamshaper-V7-ONNX/resolve/main/unet/model.onnx_data",
                        expectedSizeBytes = 3_438_411_520L,
                        isOnnxBinary = false,
                        minValidBytes = 3_000_000_000L
                    )
                )
            ),
            AutoDownloadSlotSpec(
                slot = DedicatedModelSlot.TEXT_ENCODER,
                shortCode = "3",
                displayOrder = 5,
                executionPriority = 3,
                titleEn = "Slot 3: Text Encoder CLIP (model.onnx)",
                titleTa = "Slot 3: Text Encoder CLIP (~492 MB)",
                files = listOf(
                    RemoteModelFileSpec(
                        targetFileName = "model.onnx",
                        url = "https://huggingface.co/latentdivergence/lcm-dreamshaper-int8-onnx/resolve/main/text_encoder/model.onnx",
                        expectedSizeBytes = 492_588_002L,
                        isOnnxBinary = true,
                        minValidBytes = 400_000_000L
                    )
                )
            ),
            AutoDownloadSlotSpec(
                slot = DedicatedModelSlot.VAE_DECODER,
                shortCode = "4",
                displayOrder = 6,
                executionPriority = 4,
                titleEn = "Slot 4: VAE Decoder (model.onnx)",
                titleTa = "Slot 4: VAE Decoder (~198 MB)",
                files = listOf(
                    RemoteModelFileSpec(
                        targetFileName = "model.onnx",
                        url = "https://huggingface.co/latentdivergence/lcm-dreamshaper-int8-onnx/resolve/main/vae_decoder/model.onnx",
                        expectedSizeBytes = 198_078_038L,
                        isOnnxBinary = true,
                        minValidBytes = 150_000_000L
                    )
                )
            ),
            AutoDownloadSlotSpec(
                slot = DedicatedModelSlot.VAE_ENCODER,
                shortCode = "5",
                displayOrder = 7,
                executionPriority = 5,
                titleEn = "Slot 5: VAE Encoder (model.onnx)",
                titleTa = "Slot 5: VAE Encoder (~136 MB • Img2Img)",
                files = listOf(
                    RemoteModelFileSpec(
                        targetFileName = "model.onnx",
                        url = "https://huggingface.co/latentdivergence/lcm-dreamshaper-int8-onnx/resolve/main/vae_encoder/model.onnx",
                        expectedSizeBytes = 136_760_154L,
                        isOnnxBinary = true,
                        minValidBytes = 100_000_000L
                    )
                )
            ),
            AutoDownloadSlotSpec(
                slot = DedicatedModelSlot.TOKENIZER,
                shortCode = "6",
                displayOrder = 8,
                executionPriority = 6,
                titleEn = "Slot 6: Tokenizer (vocab.json & merges.txt)",
                titleTa = "Slot 6: Tokenizer (vocab.json & merges.txt)",
                files = listOf(
                    RemoteModelFileSpec(
                        targetFileName = "vocab.json",
                        url = "https://huggingface.co/latentdivergence/lcm-dreamshaper-int8-onnx/resolve/main/tokenizer/vocab.json",
                        expectedSizeBytes = 1_059_962L,
                        isOnnxBinary = false,
                        minValidBytes = 500_000L
                    ),
                    RemoteModelFileSpec(
                        targetFileName = "merges.txt",
                        url = "https://huggingface.co/latentdivergence/lcm-dreamshaper-int8-onnx/resolve/main/tokenizer/merges.txt",
                        expectedSizeBytes = 524_619L,
                        isOnnxBinary = false,
                        minValidBytes = 200_000L
                    )
                )
            )
        )

        val DEFAULT_SELECTED_CORE_SLOTS: Set<DedicatedModelSlot> =
            CORE_SLOT_SPECS.map { it.slot }.toSet()
    }
}
