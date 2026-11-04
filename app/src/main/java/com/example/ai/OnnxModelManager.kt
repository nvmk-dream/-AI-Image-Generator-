package com.example.ai

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.app.ActivityManager
import android.content.Context
import android.net.Uri
import android.os.StatFs
import android.provider.OpenableColumns
import androidx.documentfile.provider.DocumentFile
import com.example.data.ActiveDiffusionEngine
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap

enum class DedicatedSlotGroup {
    FAST_LCM_ENGINE,
    STANDARD_SD_ENGINE,
    ESSENTIAL_SHARED,
    OPTIONAL_AVATAR_CUSTOM
}

enum class DedicatedModelSlot(
    val slotTag: String,
    val folderName: String,
    val group: DedicatedSlotGroup,
    val expectedFileHint: String,
    val titleEn: String,
    val titleTa: String,
    val descriptionEn: String,
    val descriptionTa: String
) {
    FAST_LCM_GRAPH(
        slotTag = "1a_fast_lcm_graph",
        folderName = "unet_fast_lcm",
        group = DedicatedSlotGroup.FAST_LCM_ENGINE,
        expectedFileHint = "model.onnx (~3.05 MB)",
        titleEn = "Slot 1A: Fast LCM model.onnx (Graph)",
        titleTa = "Slot 1A: Fast LCM model.onnx (Graph ~3.05 MB)",
        descriptionEn = "Fast LCM UNet graph (~3.05 MB) — auto-linked with Slot 1B",
        descriptionTa = "Fast LCM வரைபட ஃபைல் (~3.05 MB) — Slot 1B உடன் தானாக இணைக்கப்படும்"
    ),
    FAST_LCM_WEIGHTS(
        slotTag = "1b_fast_lcm_weights",
        folderName = "unet_fast_lcm",
        group = DedicatedSlotGroup.FAST_LCM_ENGINE,
        expectedFileHint = "model_int8.onnx.data (~860 MB)",
        titleEn = "Slot 1B: Fast LCM model_int8.onnx.data (Weights)",
        titleTa = "Slot 1B: Fast LCM model_int8.onnx.data (Weights ~860 MB)",
        descriptionEn = "Fast LCM weights (~860 MB) — stored side-by-side with Slot 1A",
        descriptionTa = "Fast LCM எடை ஃபைல் (~860 MB) — Slot 1A எதிர்பார்க்கும் பெயருடன் இணைக்கப்படும்"
    ),
    STANDARD_SD_GRAPH(
        slotTag = "2a_standard_sd_graph",
        folderName = "unet_standard_sd",
        group = DedicatedSlotGroup.STANDARD_SD_ENGINE,
        expectedFileHint = "model.onnx (~1.95 MB)",
        titleEn = "Slot 2A: Standard SD model.onnx (Graph)",
        titleTa = "Slot 2A: Standard SD model.onnx (Graph ~1.95 MB)",
        descriptionEn = "Standard SD 1.5 UNet graph (~1.95 MB) — auto-linked with Slot 2B",
        descriptionTa = "Standard SD வரைபட ஃபைல் (~1.95 MB) — Slot 2B உடன் தானாக இணைக்கப்படும்"
    ),
    STANDARD_SD_WEIGHTS(
        slotTag = "2b_standard_sd_weights",
        folderName = "unet_standard_sd",
        group = DedicatedSlotGroup.STANDARD_SD_ENGINE,
        expectedFileHint = "weights.pb / model.onnx_data (~3.44 GB)",
        titleEn = "Slot 2B: Standard SD weights.pb / model.onnx_data",
        titleTa = "Slot 2B: Standard SD weights.pb / model.onnx_data (~3.44 GB)",
        descriptionEn = "High-quality SD 1.5 weights (~3.44 GB • supports wights.pb typo)",
        descriptionTa = "Standard SD எடை ஃபைல் (~3.44 GB) — Slot 2A எதிர்பார்க்கும் பெயருடன் இணைக்கப்படும்"
    ),
    TEXT_ENCODER(
        slotTag = "3_text_encoder",
        folderName = "text_encoder",
        group = DedicatedSlotGroup.ESSENTIAL_SHARED,
        expectedFileHint = "model (3).onnx (~492 MB)",
        titleEn = "Slot 3: Text Encoder CLIP (~492 MB)",
        titleTa = "Slot 3: Text Encoder CLIP (model (3).onnx ~492 MB)",
        descriptionEn = "Converts Tamil & English prompts into CLIP embeddings",
        descriptionTa = "உங்கள் வாக்கியத்தை AI புரியும் எண்களாக மாற்றும் (~492 MB)"
    ),
    VAE_DECODER(
        slotTag = "4_vae_decoder",
        folderName = "vae_decoder",
        group = DedicatedSlotGroup.ESSENTIAL_SHARED,
        expectedFileHint = "model (4/7).onnx (~198 MB)",
        titleEn = "Slot 4: VAE Decoder (~198 MB)",
        titleTa = "Slot 4: VAE Decoder (model (4/7).onnx ~198 MB)",
        descriptionEn = "Decodes latents into crisp 512×512 final photo",
        descriptionTa = "வரைந்த குறிப்பைத் தெளிவான 512×512 புகைப்படமாக மாற்றும் (~198 MB)"
    ),
    VAE_ENCODER(
        slotTag = "5_vae_encoder",
        folderName = "vae_encoder",
        group = DedicatedSlotGroup.ESSENTIAL_SHARED,
        expectedFileHint = "model (8).onnx (~136 MB)",
        titleEn = "Slot 5: VAE Encoder (~136 MB • Img2Img)",
        titleTa = "Slot 5: VAE Encoder (model (8).onnx ~136 MB)",
        descriptionEn = "Encodes reference photo for Image-to-Image transformation",
        descriptionTa = "Image-to-Image மாற்றத்திற்காக பழைய படத்தை உள்ளே எடுக்கும் (~136 MB)"
    ),
    TOKENIZER(
        slotTag = "6_tokenizer",
        folderName = "tokenizer",
        group = DedicatedSlotGroup.ESSENTIAL_SHARED,
        expectedFileHint = "vocab.json & merges.txt (~1 MB)",
        titleEn = "Slot 6: Tokenizer (vocab.json & merges.txt)",
        titleTa = "Slot 6: Tokenizer (vocab.json & merges.txt ~1 MB)",
        descriptionEn = "CLIP vocabulary & merges (~1 MB • built-in backup also active)",
        descriptionTa = "vocab.json & merges.txt (~1 MB • Built-in துணையும் தயாராக உள்ளது)"
    ),
    FACE_AGING_ENGINE(
        slotTag = "7_face_aging",
        folderName = "aging_engine",
        group = DedicatedSlotGroup.OPTIONAL_AVATAR_CUSTOM,
        expectedFileHint = "face_aging.onnx (~300 MB)",
        titleEn = "Slot 7: Aging Engine (~300 MB • Optional)",
        titleTa = "Slot 7: Aging Engine (~300 MB • விருப்ப மாடல்)",
        descriptionEn = "Optional dedicated Face-Aging GAN / StyleGAN ONNX model for Avatar Age Progression (Core 1A–6 also works automatically)",
        descriptionTa = "Avatar வயது மாற்றத்திற்கான பிரத்யேக Face-Aging ONNX மாடல் (~300 MB • இல்லையென்றால் Core 1A–6 தானாக இயங்கும்)"
    );

    val isCoreSlot: Boolean
        get() = this != FACE_AGING_ENGINE
}

data class DedicatedSlotStatus(
    val slot: DedicatedModelSlot,
    val isInstalled: Boolean,
    val isPartial: Boolean = false,
    val isLinkedWithPair: Boolean = false,
    val sizeBytes: Long = 0L,
    val displayFileName: String = "",
    val statusDetailEn: String = "",
    val statusDetailTa: String = ""
)

data class EngineLinkState(
    val isGraphInstalled: Boolean,
    val isWeightsInstalled: Boolean,
    val isPairLinked: Boolean,
    val isSelfContainedSingleFile: Boolean = false,
    val graphBytes: Long = 0L,
    val weightsBytes: Long = 0L,
    val graphDisplayName: String = "",
    val weightsDisplayName: String = "",
    val expectedWeightFileName: String = ""
) {
    val totalBytes: Long
        get() = graphBytes + weightsBytes
}

enum class DiffusionModelSlot(
    val canonicalFileName: String,
    val folderHint: String,
    val matchKeywords: List<String>,
    val titleEn: String,
    val titleTa: String,
    val descriptionEn: String,
    val descriptionTa: String,
    val requiredForTxt2Img: Boolean
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
        folderHint = "unet_fast_lcm",
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
    FACE_AGING(
        canonicalFileName = "face_aging.onnx",
        folderHint = "aging_engine",
        matchKeywords = listOf("face_aging", "sam_aging", "age_progression", "stylegan_age"),
        titleEn = "5. Aging Engine (Optional Slot 7)",
        titleTa = "5. Aging Engine (விருப்ப Slot 7)",
        descriptionEn = "Optional dedicated Face-Aging ONNX model (~300 MB)",
        descriptionTa = "வயது மாற்றத்திற்கான விருப்ப ONNX மாடல் (~300 MB)",
        requiredForTxt2Img = false
    )
}

enum class UnetVariant {
    NONE,
    FAST_LCM_INT8,
    COMPACT_FP16,
    HEAVY_FP32
}

enum class PreferredUnetImportMode {
    AUTO_FAST_MOBILE,
    FORCE_FAST_LCM,
    FORCE_STANDARD_SD
}

data class FolderImportResult(
    val importedSlots: List<DiffusionModelSlot>,
    val importedDedicatedSlots: List<DedicatedModelSlot> = emptyList(),
    val fastLcmReady: Boolean = false,
    val standardSdReady: Boolean = false,
    val dualUnetDetected: Boolean = false,
    val selectedUnetVariant: UnetVariant = UnetVariant.NONE,
    val selectedUnetLabel: String = "",
    val skippedLowSpaceNote: String = ""
)

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
    val dedicatedSlots: List<DedicatedSlotStatus> = emptyList(),
    val fastLcmLink: EngineLinkState = EngineLinkState(false, false, false),
    val standardSdLink: EngineLinkState = EngineLinkState(false, false, false),
    val totalModelBytes: Long = 0L,
    val cacheBytes: Long = 0L,
    val deviceFreeBytes: Long = 0L,
    val deviceTotalBytes: Long = 0L,
    val deviceTotalRamBytes: Long = 0L,
    val deviceAvailableRamBytes: Long = 0L,
    val hasTokenizerFiles: Boolean = false,
    val activeRamSessionsCount: Int = 0,
    val phoneModelsFolderPath: String = "",
    val mediaPipeTasksStatus: MediaPipeTasksStatus = MediaPipeTasksStatus()
) {
    val coreDedicatedSlots: List<DedicatedSlotStatus>
        get() = dedicatedSlots.filter { it.slot.isCoreSlot }

    val fastLcmDedicatedSlots: List<DedicatedSlotStatus>
        get() = dedicatedSlots.filter { it.slot.group == DedicatedSlotGroup.FAST_LCM_ENGINE }

    val standardSdDedicatedSlots: List<DedicatedSlotStatus>
        get() = dedicatedSlots.filter { it.slot.group == DedicatedSlotGroup.STANDARD_SD_ENGINE }

    val essentialSharedDedicatedSlots: List<DedicatedSlotStatus>
        get() = dedicatedSlots.filter { it.slot.group == DedicatedSlotGroup.ESSENTIAL_SHARED }

    val optionalAvatarDedicatedSlots: List<DedicatedSlotStatus>
        get() = dedicatedSlots.filter { it.slot.group == DedicatedSlotGroup.OPTIONAL_AVATAR_CUSTOM }

    val isAgingSlotInstalled: Boolean
        get() = dedicatedSlots.any { it.slot == DedicatedModelSlot.FACE_AGING_ENGINE && it.isInstalled } ||
            slots.any { it.slot == DiffusionModelSlot.FACE_AGING && it.isInstalled }

    val coreInstalledCount: Int
        get() = coreDedicatedSlots.count { it.isInstalled }

    val isFastLcmEngineReady: Boolean
        get() = fastLcmLink.isPairLinked

    val isStandardSdEngineReady: Boolean
        get() = standardSdLink.isPairLinked

    val isTextEncoderReady: Boolean
        get() = dedicatedSlots.any { it.slot == DedicatedModelSlot.TEXT_ENCODER && it.isInstalled } ||
            slots.any { it.slot == DiffusionModelSlot.TEXT_ENCODER && it.isInstalled }

    val isVaeDecoderReady: Boolean
        get() = dedicatedSlots.any { it.slot == DedicatedModelSlot.VAE_DECODER && it.isInstalled } ||
            slots.any { it.slot == DiffusionModelSlot.VAE_DECODER && it.isInstalled }

    val isVaeEncoderReady: Boolean
        get() = dedicatedSlots.any { it.slot == DedicatedModelSlot.VAE_ENCODER && it.isInstalled } ||
            slots.any { it.slot == DiffusionModelSlot.VAE_ENCODER && it.isInstalled }

    fun isEngineReady(engine: ActiveDiffusionEngine): Boolean = when (engine) {
        ActiveDiffusionEngine.FAST_LCM -> isFastLcmEngineReady
        ActiveDiffusionEngine.STANDARD_SD -> isStandardSdEngineReady
    }

    fun isTxt2ImgReadyFor(engine: ActiveDiffusionEngine): Boolean =
        isEngineReady(engine) && isTextEncoderReady && isVaeDecoderReady

    fun isImg2ImgReadyFor(engine: ActiveDiffusionEngine): Boolean =
        isTxt2ImgReadyFor(engine) && isVaeEncoderReady

    fun missingSummaryFor(engine: ActiveDiffusionEngine, requireImg2Img: Boolean, isTamil: Boolean): String {
        val missingParts = mutableListOf<String>()
        when (engine) {
            ActiveDiffusionEngine.FAST_LCM -> {
                if (!fastLcmLink.isGraphInstalled) missingParts.add("Slot 1A (model.onnx)")
                if (!fastLcmLink.isWeightsInstalled) missingParts.add("Slot 1B (model_int8.onnx.data)")
            }
            ActiveDiffusionEngine.STANDARD_SD -> {
                if (!standardSdLink.isGraphInstalled) missingParts.add("Slot 2A (model.onnx)")
                if (!standardSdLink.isWeightsInstalled) missingParts.add("Slot 2B (weights.pb)")
            }
        }
        if (!isTextEncoderReady) missingParts.add("Slot 3 (Text Encoder)")
        if (!isVaeDecoderReady) missingParts.add("Slot 4 (VAE Decoder)")
        if (requireImg2Img && !isVaeEncoderReady) missingParts.add("Slot 5 (VAE Encoder)")

        if (missingParts.isEmpty()) {
            return if (isTamil) "தயாராக உள்ளது" else "Ready"
        }
        val first = missingParts.first()
        return if (missingParts.size == 1) {
            if (isTamil) "$first ஏற்றவும்" else "Upload $first"
        } else {
            if (isTamil) "$first + ${missingParts.size - 1} மாடல் ஏற்றவும்"
            else "Upload $first + ${missingParts.size - 1} more"
        }
    }

    fun activeEngineSummaryLine(engine: ActiveDiffusionEngine, isTamil: Boolean): String {
        return when (engine) {
            ActiveDiffusionEngine.FAST_LCM -> {
                when {
                    fastLcmLink.isPairLinked -> {
                        val sizeStr = OnnxModelManager.formatBytes(fastLcmLink.totalBytes)
                        if (isTamil) {
                            "🔗 Fast LCM தயார்: Slot 1A + Slot 1B இணைக்கப்பட்டது ($sizeStr • 4–8 Steps)"
                        } else {
                            "🔗 Fast LCM Ready: Slot 1A + Slot 1B linked ($sizeStr • 4–8 Steps)"
                        }
                    }
                    fastLcmLink.isGraphInstalled && !fastLcmLink.isWeightsInstalled -> {
                        val exp = fastLcmLink.expectedWeightFileName.ifBlank { "model_int8.onnx.data" }
                        if (isTamil) "⚠️ Slot 1A உள்ளது • இனி Slot 1B ($exp) ஏற்றவும்"
                        else "⚠️ Slot 1A uploaded • Now upload Slot 1B ($exp)"
                    }
                    !fastLcmLink.isGraphInstalled && fastLcmLink.isWeightsInstalled -> {
                        if (isTamil) "⚠️ Slot 1B உள்ளது • இனி Slot 1A (model.onnx) ஏற்றவும்"
                        else "⚠️ Slot 1B uploaded • Now upload Slot 1A (model.onnx)"
                    }
                    else -> {
                        if (isTamil) "Slot 1A (model.onnx) & Slot 1B (model_int8.onnx.data) ஏற்றவும்"
                        else "Upload Slot 1A (model.onnx) & Slot 1B (model_int8.onnx.data)"
                    }
                }
            }
            ActiveDiffusionEngine.STANDARD_SD -> {
                when {
                    standardSdLink.isPairLinked -> {
                        val sizeStr = OnnxModelManager.formatBytes(standardSdLink.totalBytes)
                        if (isTamil) {
                            "🔗 Standard SD தயார்: Slot 2A + Slot 2B இணைக்கப்பட்டது ($sizeStr • High Quality)"
                        } else {
                            "🔗 Standard SD Ready: Slot 2A + Slot 2B linked ($sizeStr • High Quality)"
                        }
                    }
                    standardSdLink.isGraphInstalled && !standardSdLink.isWeightsInstalled -> {
                        val exp = standardSdLink.expectedWeightFileName.ifBlank { "weights.pb" }
                        if (isTamil) "⚠️ Slot 2A உள்ளது • இனி Slot 2B ($exp) ஏற்றவும்"
                        else "⚠️ Slot 2A uploaded • Now upload Slot 2B ($exp)"
                    }
                    !standardSdLink.isGraphInstalled && standardSdLink.isWeightsInstalled -> {
                        if (isTamil) "⚠️ Slot 2B உள்ளது • இனி Slot 2A (model.onnx) ஏற்றவும்"
                        else "⚠️ Slot 2B uploaded • Now upload Slot 2A (model.onnx)"
                    }
                    else -> {
                        if (isTamil) "Slot 2A (model.onnx) & Slot 2B (weights.pb) ஏற்றவும்"
                        else "Upload Slot 2A (model.onnx) & Slot 2B (weights.pb)"
                    }
                }
            }
        }
    }

    val diffusionSlots: List<ModelSlotStatus>
        get() = slots

    val unetSlotStatus: ModelSlotStatus?
        get() = slots.firstOrNull { it.slot == DiffusionModelSlot.UNET }

    val unetVariant: UnetVariant
        get() {
            if (isFastLcmEngineReady) return UnetVariant.FAST_LCM_INT8
            if (isStandardSdEngineReady) return UnetVariant.HEAVY_FP32
            val u = unetSlotStatus ?: return UnetVariant.NONE
            if (!u.isInstalled) return UnetVariant.NONE
            return OnnxModelManager.classifyUnetVariant(u.sourceFileLabel, u.totalBytes)
        }

    fun activeUnetBadgeLabel(isTamil: Boolean, engine: ActiveDiffusionEngine = ActiveDiffusionEngine.FAST_LCM): String {
        return when (engine) {
            ActiveDiffusionEngine.FAST_LCM -> {
                if (isFastLcmEngineReady) {
                    "⚡ Fast LCM (${OnnxModelManager.formatBytes(fastLcmLink.totalBytes)})"
                } else {
                    if (isTamil) "⚡ Fast LCM (ஏற்றப்படவில்லை)" else "⚡ Fast LCM (Not uploaded)"
                }
            }
            ActiveDiffusionEngine.STANDARD_SD -> {
                if (isStandardSdEngineReady) {
                    "🎨 Standard SD (${OnnxModelManager.formatBytes(standardSdLink.totalBytes)})"
                } else {
                    if (isTamil) "🎨 Standard SD (ஏற்றப்படவில்லை)" else "🎨 Standard SD (Not uploaded)"
                }
            }
        }
    }

    val installedCount: Int
        get() = diffusionSlots.count { it.isInstalled }

    val totalInstalledCount: Int
        get() = dedicatedSlots.count { it.isInstalled }

    val hasAnySlotFilesOnDisk: Boolean
        get() = dedicatedSlots.any { it.isInstalled || it.isPartial } ||
            slots.any { it.isInstalled || it.isIncompleteExternalGraph }

    val incompleteSlots: List<ModelSlotStatus>
        get() = slots.filter { it.isIncompleteExternalGraph }

    val isTxt2ImgReady: Boolean
        get() = isTxt2ImgReadyFor(ActiveDiffusionEngine.FAST_LCM) ||
            isTxt2ImgReadyFor(ActiveDiffusionEngine.STANDARD_SD) ||
            diffusionSlots.filter { it.slot.requiredForTxt2Img }.let { req ->
                req.isNotEmpty() && req.all { it.isInstalled }
            }

    val isAllFourReady: Boolean
        get() = isImg2ImgReadyFor(ActiveDiffusionEngine.FAST_LCM) ||
            isImg2ImgReadyFor(ActiveDiffusionEngine.STANDARD_SD) ||
            (diffusionSlots.isNotEmpty() && diffusionSlots.all { it.isInstalled })
}

class OnnxModelManager(private val context: Context) {

    private val activeSessions = ConcurrentHashMap<String, OrtSession>()

    val legacyInternalRootDir: File
        get() = File(context.filesDir, "onnx_models")

    val modelsRootDir: File
        get() = resolvePrimaryModelsRootDir(context)

    val phoneModelsDisplayPath: String
        get() = formatPhoneModelsDisplayPath(context, modelsRootDir)

    init {
        ensurePhoneModelFoldersCreated()
    }

    /**
     * Immediately creates the phone-visible `Android/media/<package>/IG_Models/` directory and
     * all 6 core slot folders on first open so users and the in-app downloader have a ready folder.
     * Also attempts zero-copy `renameTo` migration from legacy `filesDir/onnx_models` if same mount.
     */
    fun ensurePhoneModelFoldersCreated(): File {
        val root = modelsRootDir
        val coreFolderNames = listOf(
            DedicatedModelSlot.FAST_LCM_GRAPH.folderName,
            DedicatedModelSlot.STANDARD_SD_GRAPH.folderName,
            DedicatedModelSlot.TEXT_ENCODER.folderName,
            DedicatedModelSlot.VAE_DECODER.folderName,
            DedicatedModelSlot.VAE_ENCODER.folderName,
            DedicatedModelSlot.TOKENIZER.folderName
        )
        for (dirName in coreFolderNames) {
            File(root, dirName).apply { if (!exists()) mkdirs() }
        }

        val legacyRoot = legacyInternalRootDir
        if (legacyRoot.exists() && legacyRoot.absolutePath != root.absolutePath) {
            migrateLegacyUnetDir(legacyRoot)
            legacyRoot.listFiles()?.filter { it.isDirectory }?.forEach { legacySub ->
                val targetSub = File(root, legacySub.name).apply { if (!exists()) mkdirs() }
                val targetHasPayload = directoryHasPayloadFiles(targetSub)
                if (!targetHasPayload) {
                    val children = legacySub.listFiles().orEmpty()
                    if (children.isNotEmpty()) {
                        // Only move if ALL files in the folder can be atomically renamed on the same mount
                        // so paired graph + weights are never split across two directories.
                        val movedFiles = mutableListOf<Pair<File, File>>()
                        var allMoved = true
                        for (child in children) {
                            val dest = File(targetSub, child.name)
                            if (child.renameTo(dest)) {
                                movedFiles.add(dest to child)
                            } else {
                                allMoved = false
                                break
                            }
                        }
                        if (allMoved) {
                            legacySub.deleteRecursively()
                        } else {
                            // Roll back partial renames so pair stays intact in legacySub
                            for ((movedDest, origLoc) in movedFiles) {
                                movedDest.renameTo(origLoc)
                            }
                        }
                    }
                }
            }
        }
        return root
    }

    /**
     * Resolves the active folder for a given slot folder name:
     * - Uses `Android/media/<package>/IG_Models/<folderName>` for all fresh installs & new downloads.
     * - Seamlessly preserves existing models in legacy `filesDir/onnx_models/<folderName>` on App Updates
     *   when Android FUSE prevents cross-mount renameTo, ensuring zero data loss and zero duplicate space.
     */
    private fun activeStorageFolder(folderName: String): File {
        val primaryDir = File(modelsRootDir, folderName).apply { if (!exists()) mkdirs() }
        if (directoryHasPayloadFiles(primaryDir)) return primaryDir
        val legacyRoot = legacyInternalRootDir
        if (legacyRoot.exists() && legacyRoot.absolutePath != modelsRootDir.absolutePath) {
            val legacyDir = File(legacyRoot, folderName)
            if (directoryHasPayloadFiles(legacyDir)) return legacyDir
        }
        return primaryDir
    }

    fun dedicatedFolder(slot: DedicatedModelSlot): File =
        activeStorageFolder(slot.folderName)

    /**
     * Always returns the phone-visible primary folder inside `IG_Models/<slot.folderName>`,
     * unless an unmigrated companion pair already resides in legacy storage for this exact folder.
     */
    fun targetDownloadFolder(slot: DedicatedModelSlot): File =
        activeStorageFolder(slot.folderName)

    fun engineFolder(engine: ActiveDiffusionEngine): File {
        val dirName = when (engine) {
            ActiveDiffusionEngine.FAST_LCM -> DedicatedModelSlot.FAST_LCM_GRAPH.folderName
            ActiveDiffusionEngine.STANDARD_SD -> DedicatedModelSlot.STANDARD_SD_GRAPH.folderName
        }
        return activeStorageFolder(dirName)
    }

    private fun slotDir(slot: DiffusionModelSlot): File =
        activeStorageFolder(slot.folderHint)

    fun slotModelFile(slot: DiffusionModelSlot): File =
        File(slotDir(slot), "model.onnx")

    fun resolveEngineUnetFile(engine: ActiveDiffusionEngine): File? {
        migrateLegacyUnetDir(modelsRootDir)
        if (legacyInternalRootDir.exists() && legacyInternalRootDir.absolutePath != modelsRootDir.absolutePath) {
            migrateLegacyUnetDir(legacyInternalRootDir)
        }
        val dir = engineFolder(engine)
        linkEnginePairFiles(dir)
        val onnx = dir.listFiles()
            ?.firstOrNull { it.isFile && it.name.endsWith(".onnx", ignoreCase = true) && it.length() > 1024L }
        return onnx
    }

    fun resolveExistingSlotFile(slot: DiffusionModelSlot): File? {
        migrateLegacyUnetDir(modelsRootDir)
        if (legacyInternalRootDir.exists() && legacyInternalRootDir.absolutePath != modelsRootDir.absolutePath) {
            migrateLegacyUnetDir(legacyInternalRootDir)
        }
        if (slot == DiffusionModelSlot.UNET) {
            val fastFile = resolveEngineUnetFile(ActiveDiffusionEngine.FAST_LCM)
            if (fastFile != null && findMissingCompanionFiles(fastFile).isEmpty()) return fastFile
            val stdFile = resolveEngineUnetFile(ActiveDiffusionEngine.STANDARD_SD)
            if (stdFile != null && findMissingCompanionFiles(stdFile).isEmpty()) return stdFile
            if (fastFile != null) return fastFile
            if (stdFile != null) return stdFile
        }
        val dir = slotDir(slot)
        val namedOnnx = dir.listFiles()
            ?.firstOrNull { it.isFile && it.name.endsWith(".onnx", ignoreCase = true) && it.length() > 1024L }
        if (namedOnnx != null) return namedOnnx

        val altFile = File(modelsRootDir, slot.canonicalFileName)
        if (altFile.exists() && altFile.length() > 1024L) return altFile
        val legacyAlt = File(legacyInternalRootDir, slot.canonicalFileName)
        if (legacyAlt.exists() && legacyAlt.length() > 1024L) return legacyAlt
        return null
    }

    fun findMissingCompanionFilesForSlot(slot: DiffusionModelSlot): List<String> {
        val actualFile = resolveExistingSlotFile(slot) ?: return emptyList()
        return findMissingCompanionFiles(actualFile)
    }

    fun inspectCurrentMetrics(): RealDeviceAndModelMetrics {
        migrateLegacyUnetDir(modelsRootDir)
        purgeOrphanedDuplicateWeights()

        val fastDir = engineFolder(ActiveDiffusionEngine.FAST_LCM)
        val stdDir = engineFolder(ActiveDiffusionEngine.STANDARD_SD)
        val fastLink = linkEnginePairFiles(fastDir)
        val stdLink = linkEnginePairFiles(stdDir)

        val (ortOk, ortVer) = runCatching {
            val env = OrtEnvironment.getEnvironment()
            true to env.version
        }.getOrElse { false to "Unavailable" }

        val dedicatedStatuses = buildDedicatedSlotStatuses(fastLink, stdLink)

        val slotStatuses = DiffusionModelSlot.entries.map { slot ->
            val actualFile = resolveExistingSlotFile(slot)
            val dir = actualFile?.parentFile ?: slotDir(slot)
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
                ?.filter {
                    it.isFile &&
                        it.absolutePath != actualFile?.absolutePath &&
                        !it.name.endsWith(".meta") &&
                        !it.name.endsWith(".repair_tmp") &&
                        !it.name.endsWith(".download_part")
                }
                ?.sumOf { it.length() } ?: 0L
            val originLabel = File(dir, "source_name.meta").takeIf { it.exists() }
                ?.readText()?.trim().orEmpty()

            val installed = actualFile != null && missingCompanions.isEmpty()
            val inRam = activeSessions.containsKey(slot.folderHint)
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

        val tokDir = dedicatedFolder(DedicatedModelSlot.TOKENIZER)
        val hasTokenizer = (File(tokDir, "vocab.json").exists() && File(tokDir, "merges.txt").exists()) ||
            File(modelsRootDir, "vocab.json").exists() ||
            File(legacyInternalRootDir, "vocab.json").exists()

        val primaryBytes = calculateDirectorySize(modelsRootDir)
        val legacyBytes = if (legacyInternalRootDir.exists() &&
            legacyInternalRootDir.absolutePath != modelsRootDir.absolutePath
        ) {
            calculateDirectorySize(legacyInternalRootDir)
        } else {
            0L
        }
        val modelBytes = primaryBytes + legacyBytes
        val cacheBytes = calculateDirectorySize(context.cacheDir)

        val statFs = runCatching { StatFs(modelsRootDir.absolutePath) }
            .getOrElse { StatFs(context.filesDir.absolutePath) }
        val freeBytes = statFs.availableBytes
        val totalBytes = statFs.totalBytes

        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        am?.getMemoryInfo(memInfo)

        val mpDir = File(modelsRootDir, MediaPipeAvatarVisionEngine.VISION_FOLDER_NAME)
        val mpFaceOk = File(mpDir, MediaPipeAvatarVisionEngine.FACE_TASK_FILENAME).let { it.exists() && it.length() > 500_000L }
        val mpPoseOk = File(mpDir, MediaPipeAvatarVisionEngine.POSE_TASK_FILENAME).let { it.exists() && it.length() > 500_000L }
        val mpSegOk = File(mpDir, MediaPipeAvatarVisionEngine.MULTICLASS_SEG_FILENAME).let { it.exists() && it.length() > 200_000L }
        val mpBytes = listOf(
            File(mpDir, MediaPipeAvatarVisionEngine.FACE_TASK_FILENAME),
            File(mpDir, MediaPipeAvatarVisionEngine.POSE_TASK_FILENAME),
            File(mpDir, MediaPipeAvatarVisionEngine.MULTICLASS_SEG_FILENAME)
        ).filter { it.exists() }.sumOf { it.length() }

        return RealDeviceAndModelMetrics(
            ortRuntimeAvailable = ortOk,
            ortVersion = ortVer,
            slots = slotStatuses,
            dedicatedSlots = dedicatedStatuses,
            fastLcmLink = fastLink,
            standardSdLink = stdLink,
            totalModelBytes = modelBytes,
            cacheBytes = cacheBytes,
            deviceFreeBytes = freeBytes,
            deviceTotalBytes = totalBytes,
            deviceTotalRamBytes = memInfo.totalMem,
            deviceAvailableRamBytes = memInfo.availMem,
            hasTokenizerFiles = hasTokenizer,
            activeRamSessionsCount = activeSessions.size,
            phoneModelsFolderPath = phoneModelsDisplayPath,
            mediaPipeTasksStatus = MediaPipeTasksStatus(
                faceLandmarkerInstalled = mpFaceOk,
                poseLandmarkerInstalled = mpPoseOk,
                multiclassSegmenterInstalled = mpSegOk,
                totalInstalledBytes = mpBytes
            )
        )
    }

    private fun buildDedicatedSlotStatuses(
        fastLink: EngineLinkState,
        stdLink: EngineLinkState
    ): List<DedicatedSlotStatus> {
        val result = ArrayList<DedicatedSlotStatus>(DedicatedModelSlot.entries.size)

        // Slot 1A: Fast LCM Graph
        result.add(
            DedicatedSlotStatus(
                slot = DedicatedModelSlot.FAST_LCM_GRAPH,
                isInstalled = fastLink.isGraphInstalled,
                isLinkedWithPair = fastLink.isPairLinked,
                sizeBytes = fastLink.graphBytes,
                displayFileName = fastLink.graphDisplayName.ifBlank { "model.onnx" },
                statusDetailEn = when {
                    fastLink.isPairLinked -> "🔗 Linked with Slot 1B • Ready"
                    fastLink.isGraphInstalled -> {
                        val exp = fastLink.expectedWeightFileName.ifBlank { "model_int8.onnx.data" }
                        "✅ Uploaded (${formatBytes(fastLink.graphBytes)}) • Upload Slot 1B ($exp)"
                    }
                    else -> "Not uploaded yet • Expected: model.onnx (~3.05 MB)"
                },
                statusDetailTa = when {
                    fastLink.isPairLinked -> "🔗 Slot 1B உடன் இணைக்கப்பட்டது • தயார்"
                    fastLink.isGraphInstalled -> {
                        val exp = fastLink.expectedWeightFileName.ifBlank { "model_int8.onnx.data" }
                        "✅ ஏற்றப்பட்டது (${formatBytes(fastLink.graphBytes)}) • இனி Slot 1B ($exp) ஏற்றவும்"
                    }
                    else -> "இன்னும் ஏற்றப்படவில்லை • model.onnx (~3.05 MB)"
                }
            )
        )

        // Slot 1B: Fast LCM Weights
        result.add(
            DedicatedSlotStatus(
                slot = DedicatedModelSlot.FAST_LCM_WEIGHTS,
                isInstalled = fastLink.isWeightsInstalled,
                isLinkedWithPair = fastLink.isPairLinked,
                sizeBytes = fastLink.weightsBytes,
                displayFileName = when {
                    fastLink.isSelfContainedSingleFile -> "Embedded in Slot 1A"
                    fastLink.weightsDisplayName.isNotBlank() -> fastLink.weightsDisplayName
                    fastLink.expectedWeightFileName.isNotBlank() -> fastLink.expectedWeightFileName
                    else -> "model_int8.onnx.data"
                },
                statusDetailEn = when {
                    fastLink.isSelfContainedSingleFile -> "🔗 Embedded inside single-file Slot 1A • Ready"
                    fastLink.isPairLinked -> "🔗 Linked as ${fastLink.weightsDisplayName} (${formatBytes(fastLink.weightsBytes)})"
                    fastLink.isWeightsInstalled -> "✅ Uploaded (${formatBytes(fastLink.weightsBytes)}) • Now upload Slot 1A (model.onnx)"
                    else -> {
                        val exp = fastLink.expectedWeightFileName.ifBlank { "model_int8.onnx.data" }
                        "Not uploaded yet • Expected: $exp (~860 MB)"
                    }
                },
                statusDetailTa = when {
                    fastLink.isSelfContainedSingleFile -> "🔗 Slot 1A உள்ளேயே இணைந்துள்ளது • தயார்"
                    fastLink.isPairLinked -> "🔗 ${fastLink.weightsDisplayName} (${formatBytes(fastLink.weightsBytes)}) இணைக்கப்பட்டது"
                    fastLink.isWeightsInstalled -> "✅ ஏற்றப்பட்டது (${formatBytes(fastLink.weightsBytes)}) • இனி Slot 1A (model.onnx) ஏற்றவும்"
                    else -> {
                        val exp = fastLink.expectedWeightFileName.ifBlank { "model_int8.onnx.data" }
                        "இன்னும் ஏற்றப்படவில்லை • $exp (~860 MB)"
                    }
                }
            )
        )

        // Slot 2A: Standard SD Graph
        result.add(
            DedicatedSlotStatus(
                slot = DedicatedModelSlot.STANDARD_SD_GRAPH,
                isInstalled = stdLink.isGraphInstalled,
                isLinkedWithPair = stdLink.isPairLinked,
                sizeBytes = stdLink.graphBytes,
                displayFileName = stdLink.graphDisplayName.ifBlank { "model.onnx" },
                statusDetailEn = when {
                    stdLink.isPairLinked -> "🔗 Linked with Slot 2B • Ready"
                    stdLink.isGraphInstalled -> {
                        val exp = stdLink.expectedWeightFileName.ifBlank { "weights.pb" }
                        "✅ Uploaded (${formatBytes(stdLink.graphBytes)}) • Upload Slot 2B ($exp)"
                    }
                    else -> "Not uploaded yet • Expected: model.onnx (~1.95 MB)"
                },
                statusDetailTa = when {
                    stdLink.isPairLinked -> "🔗 Slot 2B உடன் இணைக்கப்பட்டது • தயார்"
                    stdLink.isGraphInstalled -> {
                        val exp = stdLink.expectedWeightFileName.ifBlank { "weights.pb" }
                        "✅ ஏற்றப்பட்டது (${formatBytes(stdLink.graphBytes)}) • இனி Slot 2B ($exp) ஏற்றவும்"
                    }
                    else -> "இன்னும் ஏற்றப்படவில்லை • model.onnx (~1.95 MB)"
                }
            )
        )

        // Slot 2B: Standard SD Weights
        result.add(
            DedicatedSlotStatus(
                slot = DedicatedModelSlot.STANDARD_SD_WEIGHTS,
                isInstalled = stdLink.isWeightsInstalled,
                isLinkedWithPair = stdLink.isPairLinked,
                sizeBytes = stdLink.weightsBytes,
                displayFileName = when {
                    stdLink.isSelfContainedSingleFile -> "Embedded in Slot 2A"
                    stdLink.weightsDisplayName.isNotBlank() -> stdLink.weightsDisplayName
                    stdLink.expectedWeightFileName.isNotBlank() -> stdLink.expectedWeightFileName
                    else -> "weights.pb"
                },
                statusDetailEn = when {
                    stdLink.isSelfContainedSingleFile -> "🔗 Embedded inside single-file Slot 2A • Ready"
                    stdLink.isPairLinked -> "🔗 Linked as ${stdLink.weightsDisplayName} (${formatBytes(stdLink.weightsBytes)})"
                    stdLink.isWeightsInstalled -> "✅ Uploaded (${formatBytes(stdLink.weightsBytes)}) • Now upload Slot 2A (model.onnx)"
                    else -> {
                        val exp = stdLink.expectedWeightFileName.ifBlank { "weights.pb / model.onnx_data" }
                        "Not uploaded yet • Expected: $exp (~3.44 GB)"
                    }
                },
                statusDetailTa = when {
                    stdLink.isSelfContainedSingleFile -> "🔗 Slot 2A உள்ளேயே இணைந்துள்ளது • தயார்"
                    stdLink.isPairLinked -> "🔗 ${stdLink.weightsDisplayName} (${formatBytes(stdLink.weightsBytes)}) இணைக்கப்பட்டது"
                    stdLink.isWeightsInstalled -> "✅ ஏற்றப்பட்டது (${formatBytes(stdLink.weightsBytes)}) • இனி Slot 2A (model.onnx) ஏற்றவும்"
                    else -> {
                        val exp = stdLink.expectedWeightFileName.ifBlank { "weights.pb / model.onnx_data" }
                        "இன்னும் ஏற்றப்படவில்லை • $exp (~3.44 GB)"
                    }
                }
            )
        )

        // Slots 3, 4, 5 (Self-contained ONNX slots)
        val standaloneMapping = listOf(
            DedicatedModelSlot.TEXT_ENCODER to DiffusionModelSlot.TEXT_ENCODER,
            DedicatedModelSlot.VAE_DECODER to DiffusionModelSlot.VAE_DECODER,
            DedicatedModelSlot.VAE_ENCODER to DiffusionModelSlot.VAE_ENCODER
        )
        for ((dedSlot, diffSlot) in standaloneMapping) {
            val file = resolveExistingSlotFile(diffSlot)
            val dir = slotDir(diffSlot)
            val missing = if (file != null) findMissingCompanionFiles(file) else emptyList()
            val installed = file != null && missing.isEmpty()
            val partial = file != null && missing.isNotEmpty()
            val srcName = File(dir, "source_name.meta").takeIf { it.exists() }
                ?.readText()?.trim().orEmpty().ifBlank { file?.name ?: dedSlot.expectedFileHint }
            val bytes = file?.length() ?: 0L

            // Insert Slot 6 (Tokenizer) right after Slot 5 (VAE_ENCODER) so order matches enum 1..6
            result.add(
                DedicatedSlotStatus(
                    slot = dedSlot,
                    isInstalled = installed,
                    isPartial = partial,
                    sizeBytes = bytes,
                    displayFileName = if (installed || partial) srcName else dedSlot.expectedFileHint,
                    statusDetailEn = when {
                        installed -> "✅ $srcName • ${formatBytes(bytes)}"
                        partial -> "⚠️ Missing external weights: ${missing.joinToString()}"
                        else -> "Not uploaded yet • ${dedSlot.expectedFileHint}"
                    },
                    statusDetailTa = when {
                        installed -> "✅ $srcName • ${formatBytes(bytes)}"
                        partial -> "⚠️ விடுபட்டுள்ளது: ${missing.joinToString()}"
                        else -> "இன்னும் ஏற்றப்படவில்லை • ${dedSlot.expectedFileHint}"
                    }
                )
            )

            if (dedSlot == DedicatedModelSlot.VAE_ENCODER) {
                val tokDir = dedicatedFolder(DedicatedModelSlot.TOKENIZER)
                val vocabFile = File(tokDir, "vocab.json")
                val mergesFile = File(tokDir, "merges.txt")
                val hasVocab = vocabFile.exists() && vocabFile.length() > 1024L
                val hasMerges = mergesFile.exists() && mergesFile.length() > 1024L
                val tokInstalled = hasVocab && hasMerges
                val tokPartial = hasVocab xor hasMerges
                val tokBytes = (if (hasVocab) vocabFile.length() else 0L) +
                    (if (hasMerges) mergesFile.length() else 0L)

                result.add(
                    DedicatedSlotStatus(
                        slot = DedicatedModelSlot.TOKENIZER,
                        isInstalled = tokInstalled,
                        isPartial = tokPartial,
                        sizeBytes = tokBytes,
                        displayFileName = when {
                            tokInstalled -> "vocab.json + merges.txt"
                            hasVocab -> "vocab.json (merges.txt missing)"
                            hasMerges -> "merges.txt (vocab.json missing)"
                            else -> "vocab.json & merges.txt"
                        },
                        statusDetailEn = when {
                            tokInstalled -> "✅ vocab.json + merges.txt • ${formatBytes(tokBytes)}"
                            hasVocab -> "⚠️ vocab.json ready • Select merges.txt too"
                            hasMerges -> "⚠️ merges.txt ready • Select vocab.json too"
                            else -> "Not uploaded yet (Built-in CLIP tokenizer active)"
                        },
                        statusDetailTa = when {
                            tokInstalled -> "✅ vocab.json + merges.txt • ${formatBytes(tokBytes)}"
                            hasVocab -> "⚠️ vocab.json உள்ளது • merges.txt ஃபைலையும் தேர்வு செய்க"
                            hasMerges -> "⚠️ merges.txt உள்ளது • vocab.json ஃபைலையும் தேர்வு செய்க"
                            else -> "இன்னும் ஏற்றப்படவில்லை (Built-in Tokenizer தயாராக உள்ளது)"
                        }
                    )
                )
            }
        }

        // Slot 7: Optional Face-Aging ONNX Model (~300 MB)
        val agingFile = resolveExistingSlotFile(DiffusionModelSlot.FACE_AGING)
        val agingDir = slotDir(DiffusionModelSlot.FACE_AGING)
        val agingMissing = if (agingFile != null) findMissingCompanionFiles(agingFile) else emptyList()
        val agingInstalled = agingFile != null && agingMissing.isEmpty()
        val agingPartial = agingFile != null && agingMissing.isNotEmpty()
        val agingSrcName = File(agingDir, "source_name.meta").takeIf { it.exists() }
            ?.readText()?.trim().orEmpty().ifBlank { agingFile?.name ?: DedicatedModelSlot.FACE_AGING_ENGINE.expectedFileHint }
        val agingBytes = agingFile?.length() ?: 0L

        result.add(
            DedicatedSlotStatus(
                slot = DedicatedModelSlot.FACE_AGING_ENGINE,
                isInstalled = agingInstalled,
                isPartial = agingPartial,
                sizeBytes = agingBytes,
                displayFileName = if (agingInstalled || agingPartial) agingSrcName else DedicatedModelSlot.FACE_AGING_ENGINE.expectedFileHint,
                statusDetailEn = when {
                    agingInstalled -> "✅ $agingSrcName • ${formatBytes(agingBytes)} (Dedicated Aging Active)"
                    agingPartial -> "⚠️ Missing external weights: ${agingMissing.joinToString()}"
                    else -> "Optional (~300 MB) • Core 1A–6 Age Inpainting active automatically"
                },
                statusDetailTa = when {
                    agingInstalled -> "✅ $agingSrcName • ${formatBytes(agingBytes)} (பிரத்யேக Aging தயார்)"
                    agingPartial -> "⚠️ விடுபட்டுள்ளது: ${agingMissing.joinToString()}"
                    else -> "விருப்ப மாடல் (~300 MB) • இது இல்லாமலும் Core 1A–6 மூலம் Age மாற்றம் இயங்கும்"
                }
            )
        )

        return result
    }

    /**
     * Purges stray duplicate external weights accidentally copied alongside self-contained
     * non-UNet slots (Slots 3, 4, 5), without ever touching unet_fast_lcm or
     * unet_standard_sd (where weights may legitimately be uploaded before model.onnx!).
     */
    private fun purgeOrphanedDuplicateWeights() {
        for (slot in DiffusionModelSlot.entries) {
            if (slot == DiffusionModelSlot.UNET) continue
            val dir = slotDir(slot)
            val actualOnnx = resolveExistingSlotFile(slot) ?: continue
            val expectedLower = extractExternalDataLocations(actualOnnx)
                .map { it.lowercase(Locale.US) }
                .toSet()

            dir.listFiles()?.forEach { f ->
                if (f.isFile && f.absolutePath != actualOnnx.absolutePath && isCompanionWeightFileName(f.name)) {
                    val keep = expectedLower.isNotEmpty() && f.name.lowercase(Locale.US) in expectedLower
                    if (!keep) {
                        runCatching { f.delete() }
                    }
                }
            }
        }
    }

    /**
     * Uploads a single file into any of the Dedicated Split Slots (1A, 1B, 2A, 2B, 3, 4, 5, or 6),
     * supporting any upload order (e.g. 1B before 1A, or 2B wights.pb before 2A model.onnx) and
     * automatically linking the pair in the shared engine folder!
     */
    fun importDedicatedSlotFromUri(
        uri: Uri,
        slot: DedicatedModelSlot,
        originalNameHint: String? = null,
        onProgress: ((String) -> Unit)? = null
    ): Result<Pair<DedicatedModelSlot, Long>> = runCatching {
        migrateLegacyUnetDir(modelsRootDir)
        if (slot == DedicatedModelSlot.TOKENIZER) {
            val bytes = importTokenizerUris(listOf(uri), listOf(originalNameHint), onProgress).getOrThrow()
            return@runCatching DedicatedModelSlot.TOKENIZER to bytes
        }

        val rawHint = originalNameHint?.trim().orEmpty()
        val normalizedHint = normalizeBrowserDuplicateName(rawHint)
        val expectedTotal = queryUriLength(uri)

        // Smart auto-routing if user tapped Graph button (1A/2A) and picked a .data/.pb weight file,
        // or tapped Weights button (1B/2B) and picked a small .onnx graph file!
        val effectiveSlot = when (slot) {
            DedicatedModelSlot.FAST_LCM_GRAPH -> {
                if (normalizedHint.isNotBlank() && isCompanionWeightFileName(normalizedHint)) {
                    DedicatedModelSlot.FAST_LCM_WEIGHTS
                } else slot
            }
            DedicatedModelSlot.FAST_LCM_WEIGHTS -> {
                if (normalizedHint.endsWith(".onnx", ignoreCase = true) &&
                    !isCompanionWeightFileName(normalizedHint)
                ) {
                    DedicatedModelSlot.FAST_LCM_GRAPH
                } else slot
            }
            DedicatedModelSlot.STANDARD_SD_GRAPH -> {
                if (normalizedHint.isNotBlank() && isCompanionWeightFileName(normalizedHint)) {
                    DedicatedModelSlot.STANDARD_SD_WEIGHTS
                } else slot
            }
            DedicatedModelSlot.STANDARD_SD_WEIGHTS -> {
                if (normalizedHint.endsWith(".onnx", ignoreCase = true) &&
                    !isCompanionWeightFileName(normalizedHint)
                ) {
                    DedicatedModelSlot.STANDARD_SD_GRAPH
                } else slot
            }
            else -> slot
        }

        val targetDir = dedicatedFolder(effectiveSlot)
        activeSessions.remove(effectiveSlot.folderName)?.runCatching { close() }

        when (effectiveSlot) {
            DedicatedModelSlot.FAST_LCM_GRAPH,
            DedicatedModelSlot.STANDARD_SD_GRAPH -> {
                ensureEnoughFreeSpace(expectedTotal)
                val tempGraph = File(targetDir, "model.onnx.uploading_tmp")
                val label = rawHint.ifBlank { "model.onnx" }
                copyUriToFile(uri, tempGraph, expectedTotal) { copied, total ->
                    onProgress?.invoke(formatCopyProgress(label, copied, total))
                }
                if (tempGraph.length() < 1024L) {
                    tempGraph.delete()
                    error("Selected file is too small to be a valid ONNX graph file.")
                }
                OnnxProtoCodec.inspectModelFileHeader(tempGraph)?.let { badHeaderReason ->
                    tempGraph.delete()
                    error("Invalid ONNX graph file ($label): $badHeaderReason")
                }
                val protoLocations = extractExternalDataLocations(tempGraph)
                if (tempGraph.length() in 30_000_000L..400_000_000L && protoLocations.isEmpty()) {
                    val badLen = tempGraph.length()
                    tempGraph.delete()
                    error(
                        "தேர்ந்தெடுக்கப்பட்ட ஃபைல் (${formatBytes(badLen)}) UNet மாடல் அல்ல; இது VAE மாடல் போல உள்ளது (Slot 4 / Slot 5-ல் ஏற்றவும்)."
                    )
                }

                // Replace ONLY model.onnx + graph marker — preserve companion weights & .companion_u8.meta in 1B/2B!
                OnnxQuantGraphRepairer.clearGraphRepairMarkerOnly(targetDir)
                val destGraph = File(targetDir, "model.onnx")
                if (destGraph.exists()) destGraph.delete()
                if (!tempGraph.renameTo(destGraph)) {
                    tempGraph.copyTo(destGraph, overwrite = true)
                    tempGraph.delete()
                }
                File(targetDir, "graph_source.meta").writeText(label)
                linkEnginePairFiles(targetDir)
                effectiveSlot to destGraph.length()
            }

            DedicatedModelSlot.FAST_LCM_WEIGHTS,
            DedicatedModelSlot.STANDARD_SD_WEIGHTS -> {
                ensureEnoughFreeSpace(expectedTotal)
                val existingGraph = File(targetDir, "model.onnx").takeIf { it.exists() && it.length() > 1024L }
                val expectedLocations = if (existingGraph != null) {
                    extractExternalDataLocations(existingGraph)
                } else {
                    emptyList()
                }
                val defaultWeightName = if (effectiveSlot == DedicatedModelSlot.FAST_LCM_WEIGHTS) {
                    "model_int8.onnx.data"
                } else {
                    "weights.pb"
                }
                val targetWeightName = when {
                    expectedLocations.size == 1 -> expectedLocations.first()
                    expectedLocations.isNotEmpty() -> {
                        expectedLocations.firstOrNull { companionMatchesExpectedLocation(normalizedHint, it) }
                            ?: expectedLocations.first()
                    }
                    normalizedHint.isNotBlank() && isCompanionWeightFileName(normalizedHint) -> normalizedHint
                    else -> defaultWeightName
                }

                // Delete ONLY previous companion weight files — NEVER delete model.onnx in 1A/2A!
                targetDir.listFiles()?.forEach { f ->
                    if (f.isFile &&
                        !f.name.equals("model.onnx", ignoreCase = true) &&
                        !f.name.equals("graph_source.meta", ignoreCase = true)
                    ) {
                        runCatching { f.delete() }
                    }
                }
                OnnxQuantGraphRepairer.clearRepairMarkers(targetDir)

                val destWeight = File(targetDir, targetWeightName)
                val displayLabel = rawHint.ifBlank { targetWeightName }
                copyUriToFile(uri, destWeight, expectedTotal) { copied, total ->
                    onProgress?.invoke(formatCopyProgress(displayLabel, copied, total))
                }
                if (destWeight.length() < MIN_VALID_COMPANION_BYTES) {
                    destWeight.delete()
                    error("Selected weights file ($displayLabel) is too small or incomplete.")
                }
                File(targetDir, "weights_source.meta").writeText(displayLabel)
                linkEnginePairFiles(targetDir)
                effectiveSlot to destWeight.length()
            }

            else -> {
                // Standalone ONNX model slots: Slot 3, Slot 4, Slot 5
                ensureEnoughFreeSpace(expectedTotal)
                targetDir.listFiles()?.forEach { it.delete() }
                val destFile = File(targetDir, "model.onnx")
                val displayLabel = rawHint.ifBlank { effectiveSlot.expectedFileHint }
                copyUriToFile(uri, destFile, expectedTotal) { copied, total ->
                    onProgress?.invoke(formatCopyProgress(displayLabel, copied, total))
                }
                if (destFile.length() < 1024L) {
                    destFile.delete()
                    error("Selected file is too small to be a valid ONNX model.")
                }
                if (rawHint.isNotBlank()) {
                    File(targetDir, "source_name.meta").writeText(rawHint)
                }
                effectiveSlot to destFile.length()
            }
        }
    }

    /**
     * Imports one or both CLIP Tokenizer files (vocab.json & merges.txt) into Slot 6 without
     * wiping the sibling file if uploaded one-by-one.
     */
    fun importTokenizerUris(
        uris: List<Uri>,
        nameHints: List<String?> = emptyList(),
        onProgress: ((String) -> Unit)? = null
    ): Result<Long> = runCatching {
        if (uris.isEmpty()) error("No tokenizer file selected.")
        val tokDir = dedicatedFolder(DedicatedModelSlot.TOKENIZER)
        var totalSaved = 0L

        uris.forEachIndexed { idx, uri ->
            val rawName = nameHints.getOrNull(idx)?.trim().orEmpty()
            val normLower = normalizeBrowserDuplicateName(rawName).lowercase(Locale.US)
            val tempFile = File(tokDir, "tok_upload_$idx.tmp")
            copyUriToFile(uri, tempFile, queryUriLength(uri)) { copied, total ->
                onProgress?.invoke(formatCopyProgress(rawName.ifBlank { "tokenizer" }, copied, total))
            }
            if (tempFile.length() < 128L) {
                tempFile.delete()
                error("Selected tokenizer file is empty or invalid.")
            }

            val headSample = runCatching {
                tempFile.inputStream().use { ins ->
                    val b = ByteArray(minOf(2048, tempFile.length().toInt()))
                    val r = ins.read(b)
                    if (r > 0) String(b, 0, r, Charsets.UTF_8).trimStart() else ""
                }
            }.getOrDefault("")

            val targetFileName = when {
                normLower == "vocab.json" || normLower.contains("vocab") -> "vocab.json"
                normLower == "merges.txt" || normLower.contains("merges") -> "merges.txt"
                normLower == "tokenizer_config.json" -> "tokenizer_config.json"
                normLower == "special_tokens_map.json" -> "special_tokens_map.json"
                headSample.startsWith("{") -> "vocab.json"
                else -> "merges.txt"
            }
            val dest = File(tokDir, targetFileName)
            if (dest.exists()) dest.delete()
            if (!tempFile.renameTo(dest)) {
                tempFile.copyTo(dest, overwrite = true)
                tempFile.delete()
            }
            totalSaved += dest.length()
        }
        totalSaved
    }

    fun deleteDedicatedSlot(slot: DedicatedModelSlot) {
        activeSessions.remove(slot.folderName)?.runCatching { close() }
        val dirsToClean = buildList {
            add(File(modelsRootDir, slot.folderName))
            if (legacyInternalRootDir.exists() && legacyInternalRootDir.absolutePath != modelsRootDir.absolutePath) {
                add(File(legacyInternalRootDir, slot.folderName))
            }
        }
        for (dir in dirsToClean) {
            if (!dir.exists()) continue
            when (slot) {
                DedicatedModelSlot.FAST_LCM_GRAPH,
                DedicatedModelSlot.STANDARD_SD_GRAPH -> {
                    File(dir, "model.onnx").delete()
                    File(dir, "model.onnx.download_part").delete()
                    File(dir, "graph_source.meta").delete()
                    File(dir, "source_name.meta").delete()
                    OnnxQuantGraphRepairer.clearRepairMarkers(dir)
                    linkEnginePairFiles(dir)
                }
                DedicatedModelSlot.FAST_LCM_WEIGHTS,
                DedicatedModelSlot.STANDARD_SD_WEIGHTS -> {
                    dir.listFiles()?.forEach { f ->
                        if (f.isFile &&
                            !f.name.equals("model.onnx", ignoreCase = true) &&
                            !f.name.equals("graph_source.meta", ignoreCase = true)
                        ) {
                            f.delete()
                        }
                    }
                    OnnxQuantGraphRepairer.clearRepairMarkers(dir)
                    linkEnginePairFiles(dir)
                }
                DedicatedModelSlot.TOKENIZER -> {
                    dir.deleteRecursively()
                    dir.mkdirs()
                }
                else -> {
                    dir.deleteRecursively()
                    if (slot.isCoreSlot) dir.mkdirs()
                }
            }
        }
        if (slot == DedicatedModelSlot.TOKENIZER) {
            File(modelsRootDir, "vocab.json").delete()
            File(modelsRootDir, "merges.txt").delete()
            File(legacyInternalRootDir, "vocab.json").delete()
            File(legacyInternalRootDir, "merges.txt").delete()
        }
    }

    // Legacy compatibility wrappers
    fun importSingleSlotFromUri(
        uri: Uri,
        slot: DiffusionModelSlot,
        originalNameHint: String? = null,
        onProgress: ((String) -> Unit)? = null
    ): Result<Long> {
        val mapped = when (slot) {
            DiffusionModelSlot.TEXT_ENCODER -> DedicatedModelSlot.TEXT_ENCODER
            DiffusionModelSlot.UNET -> DedicatedModelSlot.FAST_LCM_GRAPH
            DiffusionModelSlot.VAE_DECODER -> DedicatedModelSlot.VAE_DECODER
            DiffusionModelSlot.VAE_ENCODER -> DedicatedModelSlot.VAE_ENCODER
            DiffusionModelSlot.FACE_AGING -> DedicatedModelSlot.FACE_AGING_ENGINE
        }
        return importDedicatedSlotFromUri(uri, mapped, originalNameHint, onProgress).map { it.second }
    }

    fun importCompanionWeightForSlot(
        uri: Uri,
        slot: DiffusionModelSlot,
        originalNameHint: String? = null,
        onProgress: ((String) -> Unit)? = null
    ): Result<Pair<String, Long>> = runCatching {
        val targetSlot = if (slot == DiffusionModelSlot.UNET) {
            val norm = normalizeBrowserDuplicateName(originalNameHint.orEmpty()).lowercase(Locale.US)
            if (norm.endsWith(".pb") && !engineFolder(ActiveDiffusionEngine.FAST_LCM).resolve("model.onnx").exists()) {
                DedicatedModelSlot.STANDARD_SD_WEIGHTS
            } else {
                DedicatedModelSlot.FAST_LCM_WEIGHTS
            }
        } else {
            error("Companion weights are only supported for UNet engine slots.")
        }
        val (_, bytes) = importDedicatedSlotFromUri(uri, targetSlot, originalNameHint, onProgress).getOrThrow()
        val link = linkEnginePairFiles(dedicatedFolder(targetSlot))
        link.weightsDisplayName.ifBlank { "weights" } to bytes
    }

    /**
     * Self-heals ONLY `unet_fast_lcm/model.onnx` (~3.05 MB Slot 1A graph) from any persisted SAF
     * folder tree (`Phone > IG`) if an older v1 build corrupted its `NodeProto` attributes on disk,
     * WITHOUT re-copying `model_int8.onnx.data` (~860 MB Slot 1B) or touching any other slot.
     */
    fun restoreCorruptedFastLcmGraphIfPossible(onProgress: ((String) -> Unit)? = null): Boolean {
        val perms = runCatching {
            context.contentResolver.persistedUriPermissions.filter { it.isReadPermission }
        }.getOrDefault(emptyList())
        if (perms.isEmpty()) return false

        val fastDir = engineFolder(ActiveDiffusionEngine.FAST_LCM)
        OnnxQuantGraphRepairer.recordExistingCompanionUint8StateIfAny(fastDir)

        for (perm in perms) {
            val restored = runCatching {
                val rootDoc = DocumentFile.fromTreeUri(context, perm.uri) ?: return@runCatching false
                data class SmallGraphDoc(val parentName: String, val doc: DocumentFile, val sizeBytes: Long)
                val candidates = mutableListOf<SmallGraphDoc>()
                fun walk(dir: DocumentFile, parentLabel: String) {
                    for (child in dir.listFiles()) {
                        if (child.isDirectory) {
                            walk(child, child.name.orEmpty().lowercase(Locale.US))
                        } else if (child.isFile) {
                            val name = child.name.orEmpty()
                            val len = child.length()
                            if (name.endsWith(".onnx", ignoreCase = true) &&
                                !isCompanionWeightFileName(name) &&
                                len in 1025L..MAX_GRAPH_SHELL_STREAM_SCAN_BYTES
                            ) {
                                candidates.add(SmallGraphDoc(parentLabel, child, len))
                            }
                        }
                    }
                }
                walk(rootDoc, rootDoc.name.orEmpty().lowercase(Locale.US))
                if (candidates.isEmpty()) return@runCatching false

                var bestDoc: DocumentFile? = null
                for (cand in candidates) {
                    val scan = runCatching {
                        context.contentResolver.openInputStream(cand.doc.uri)?.use { ins ->
                            scanGraphShellStream(ins, MAX_GRAPH_SHELL_STREAM_SCAN_BYTES)
                        }
                    }.getOrNull() ?: continue
                    val extLocs = scan.first
                    val hasInt8OrLcm = scan.second
                    if (extLocs.any { it.contains("int8", ignoreCase = true) || it.equals("model_int8.onnx.data", ignoreCase = true) } ||
                        (hasInt8OrLcm && extLocs.isNotEmpty())
                    ) {
                        bestDoc = cand.doc
                        break
                    }
                }
                val chosen = bestDoc ?: return@runCatching false
                onProgress?.invoke("Restoring clean Slot 1A Fast LCM model.onnx (~3.05 MB)...")
                importDedicatedSlotFromUri(
                    uri = chosen.uri,
                    slot = DedicatedModelSlot.FAST_LCM_GRAPH,
                    originalNameHint = chosen.name,
                    onProgress = onProgress
                ).getOrThrow()
                true
            }.getOrDefault(false)
            if (restored) return true
        }
        return false
    }

    /**
     * Upgraded One-Touch Folder Auto-Import (supports Phone > IG flat folder with browser-renamed
     * files like model.onnx, model (1).onnx, model (3).onnx, model (7).onnx, model (8).onnx,
     * model_int8.onnx.data, weights.pb / wights.pb, vocab.json, merges.txt, as well as nested
     * Hugging Face folders).
     * Automatically populates ALL 8 Core Dedicated Split Slots cleanly!
     */
    fun importAllModelsFromFolder(
        treeUri: Uri,
        preferredUnetMode: PreferredUnetImportMode = PreferredUnetImportMode.AUTO_FAST_MOBILE,
        onProgress: (String) -> Unit
    ): Result<FolderImportResult> = runCatching {
        migrateLegacyUnetDir(modelsRootDir)
        val rootDoc = DocumentFile.fromTreeUri(context, treeUri)
            ?: error("Could not open selected folder.")

        data class RawDocEntry(val parentName: String, val doc: DocumentFile, val sizeBytes: Long)

        val rawDocs = mutableListOf<RawDocEntry>()
        fun walk(dir: DocumentFile, parentLabel: String) {
            for (child in dir.listFiles()) {
                if (child.isDirectory) {
                    walk(child, child.name.orEmpty().lowercase(Locale.US))
                } else if (child.isFile) {
                    rawDocs.add(RawDocEntry(parentLabel, child, child.length()))
                }
            }
        }
        onProgress("Scanning folder for ONNX & Tokenizer files...")
        walk(rootDoc, rootDoc.name.orEmpty().lowercase(Locale.US))

        // Inspect small .onnx graph shells (<= 25 MB) directly via stream so we know their exact
        // protobuf external_data location ("model_int8.onnx.data" vs "weights.pb" / "model.onnx_data")
        // and whether they contain INT8/LCM ops!
        val discoveredFiles = rawDocs.mapIndexed { idx, entry ->
            val rawName = entry.doc.name.orEmpty()
            val normName = normalizeBrowserDuplicateName(rawName)
            var extLocs = emptyList<String>()
            var hasLcmOrInt8Ops = false
            if (normName.endsWith(".onnx", ignoreCase = true) &&
                entry.sizeBytes in 1025L..MAX_GRAPH_SHELL_STREAM_SCAN_BYTES
            ) {
                runCatching {
                    context.contentResolver.openInputStream(entry.doc.uri)?.use { ins ->
                        val scan = scanGraphShellStream(ins, MAX_GRAPH_SHELL_STREAM_SCAN_BYTES)
                        extLocs = scan.first
                        hasLcmOrInt8Ops = scan.second
                    }
                }
            }
            DiscoveredFolderFile(
                index = idx,
                rawName = rawName,
                normalizedName = normName,
                parentNameLower = entry.parentName,
                sizeBytes = entry.sizeBytes,
                externalLocations = extLocs,
                hasLcmOrInt8GraphOps = hasLcmOrInt8Ops
            )
        }

        val plan = resolveSplitFolderMatches(discoveredFiles)
        val importedDedicated = mutableListOf<DedicatedModelSlot>()
        var skippedLowSpaceNote = ""

        // Import order: Fast LCM (1A, 1B) + Essential Shared (3, 4, 5, 6) FIRST,
        // then Standard SD (2A, 2B) so a huge 3.44 GB weights.pb never starves Fast LCM on 64GB phones!
        val orderedSlots = listOf(
            DedicatedModelSlot.FAST_LCM_GRAPH,
            DedicatedModelSlot.FAST_LCM_WEIGHTS,
            DedicatedModelSlot.TEXT_ENCODER,
            DedicatedModelSlot.VAE_DECODER,
            DedicatedModelSlot.VAE_ENCODER,
            DedicatedModelSlot.TOKENIZER,
            DedicatedModelSlot.FACE_AGING_ENGINE,
            DedicatedModelSlot.STANDARD_SD_GRAPH,
            DedicatedModelSlot.STANDARD_SD_WEIGHTS
        )

        for (slot in orderedSlots) {
            val indices = plan.slotFileIndices[slot].orEmpty()
            if (indices.isEmpty()) continue

            if (slot == DedicatedModelSlot.TOKENIZER) {
                val uris = indices.map { rawDocs[it].doc.uri }
                val hints = indices.map { rawDocs[it].doc.name }
                onProgress("Importing Slot 6 Tokenizer (vocab.json & merges.txt)...")
                importTokenizerUris(uris, hints, onProgress).getOrThrow()
                val tokDir = dedicatedFolder(DedicatedModelSlot.TOKENIZER)
                if (File(tokDir, "vocab.json").exists() && File(tokDir, "merges.txt").exists()) {
                    importedDedicated.add(DedicatedModelSlot.TOKENIZER)
                }
                continue
            }

            val fileIdx = indices.first()
            val entry = rawDocs[fileIdx]
            val origName = entry.doc.name.orEmpty().ifBlank { slot.expectedFileHint }
            val expectedLen = entry.sizeBytes

            // Fast-path: skip re-copying large intact files already on disk with identical byte size!
            if (isDedicatedSlotAlreadyIntactOnDisk(slot, expectedLen)) {
                onProgress("Verified ${slot.name} ($origName already on device ✓)")
                importedDedicated.add(slot)
                continue
            }

            // Graceful disk-space check for optional secondary 3.44 GB Standard SD weights when Fast LCM is already present
            if (slot == DedicatedModelSlot.STANDARD_SD_WEIGHTS &&
                DedicatedModelSlot.FAST_LCM_WEIGHTS in importedDedicated &&
                !hasEnoughFreeSpace(expectedLen)
            ) {
                skippedLowSpaceNote = "Standard SD 3.44 GB weights skipped (insufficient phone storage); Fast LCM is Ready!"
                continue
            }

            val (_, _) = importDedicatedSlotFromUri(
                uri = entry.doc.uri,
                slot = slot,
                originalNameHint = origName,
                onProgress = onProgress
            ).getOrThrow()
            importedDedicated.add(slot)
        }

        val fastLink = linkEnginePairFiles(engineFolder(ActiveDiffusionEngine.FAST_LCM))
        val stdLink = linkEnginePairFiles(engineFolder(ActiveDiffusionEngine.STANDARD_SD))

        // Clean any PNDMScheduler pollution if Fast LCM is present
        val tokDestDir = dedicatedFolder(DedicatedModelSlot.TOKENIZER)
        if (fastLink.isPairLinked) {
            File(tokDestDir, "scheduler_config.json").delete()
        }

        val legacyImported = mutableListOf<DiffusionModelSlot>()
        if (DedicatedModelSlot.TEXT_ENCODER in importedDedicated) legacyImported.add(DiffusionModelSlot.TEXT_ENCODER)
        if (fastLink.isPairLinked || stdLink.isPairLinked) legacyImported.add(DiffusionModelSlot.UNET)
        if (DedicatedModelSlot.VAE_DECODER in importedDedicated) legacyImported.add(DiffusionModelSlot.VAE_DECODER)
        if (DedicatedModelSlot.VAE_ENCODER in importedDedicated) legacyImported.add(DiffusionModelSlot.VAE_ENCODER)
        if (DedicatedModelSlot.FACE_AGING_ENGINE in importedDedicated) legacyImported.add(DiffusionModelSlot.FACE_AGING)

        val primaryVariant = when {
            fastLink.isPairLinked -> UnetVariant.FAST_LCM_INT8
            stdLink.isPairLinked -> UnetVariant.HEAVY_FP32
            else -> UnetVariant.NONE
        }
        val primaryLabel = when {
            fastLink.isPairLinked -> "${fastLink.graphDisplayName} + ${fastLink.weightsDisplayName}"
            stdLink.isPairLinked -> "${stdLink.graphDisplayName} + ${stdLink.weightsDisplayName}"
            else -> ""
        }

        FolderImportResult(
            importedSlots = legacyImported,
            importedDedicatedSlots = importedDedicated,
            fastLcmReady = fastLink.isPairLinked,
            standardSdReady = stdLink.isPairLinked,
            dualUnetDetected = fastLink.isGraphInstalled && stdLink.isGraphInstalled,
            selectedUnetVariant = primaryVariant,
            selectedUnetLabel = primaryLabel,
            skippedLowSpaceNote = skippedLowSpaceNote
        )
    }

    private fun isDedicatedSlotAlreadyIntactOnDisk(slot: DedicatedModelSlot, expectedLen: Long): Boolean {
        if (expectedLen <= 1024L) return false
        val dir = dedicatedFolder(slot)
        return when (slot) {
            DedicatedModelSlot.FAST_LCM_GRAPH,
            DedicatedModelSlot.STANDARD_SD_GRAPH -> {
                val f = File(dir, "model.onnx")
                f.exists() && f.length() == expectedLen
            }
            DedicatedModelSlot.FAST_LCM_WEIGHTS,
            DedicatedModelSlot.STANDARD_SD_WEIGHTS -> {
                dir.listFiles()?.any { f ->
                    f.isFile &&
                        !f.name.equals("model.onnx", ignoreCase = true) &&
                        !f.name.endsWith(".meta") &&
                        f.length() == expectedLen
                } == true
            }
            DedicatedModelSlot.TOKENIZER -> false
            else -> {
                val onnx = dir.listFiles()?.firstOrNull { it.isFile && it.name.endsWith(".onnx", true) }
                onnx != null && onnx.length() == expectedLen && findMissingCompanionFiles(onnx).isEmpty()
            }
        }
    }

    fun isDedicatedSlotReady(slot: DedicatedModelSlot): Boolean {
        val dir = dedicatedFolder(slot)
        return when (slot) {
            DedicatedModelSlot.FAST_LCM_GRAPH,
            DedicatedModelSlot.STANDARD_SD_GRAPH -> {
                val f = File(dir, "model.onnx")
                f.exists() && f.length() > 1024L && OnnxProtoCodec.inspectModelFileHeader(f) == null
            }
            DedicatedModelSlot.FAST_LCM_WEIGHTS,
            DedicatedModelSlot.STANDARD_SD_WEIGHTS -> {
                val link = linkEnginePairFiles(dir)
                link.isWeightsInstalled
            }
            DedicatedModelSlot.TOKENIZER -> {
                val v = File(dir, "vocab.json")
                val m = File(dir, "merges.txt")
                v.exists() && v.length() > 1024L && m.exists() && m.length() > 1024L
            }
            else -> {
                val onnx = dir.listFiles()?.firstOrNull {
                    it.isFile && it.name.endsWith(".onnx", ignoreCase = true) && it.length() > 1024L
                }
                onnx != null && findMissingCompanionFiles(onnx).isEmpty()
            }
        }
    }

    fun deleteSlotModel(slot: DiffusionModelSlot) {
        activeSessions.remove(slot.folderHint)?.runCatching { close() }
        if (slot == DiffusionModelSlot.UNET) {
            deleteDedicatedSlot(DedicatedModelSlot.FAST_LCM_GRAPH)
            deleteDedicatedSlot(DedicatedModelSlot.FAST_LCM_WEIGHTS)
            return
        }
        File(modelsRootDir, slot.folderHint).deleteRecursively()
        if (legacyInternalRootDir.exists() && legacyInternalRootDir.absolutePath != modelsRootDir.absolutePath) {
            File(legacyInternalRootDir, slot.folderHint).deleteRecursively()
        }
        File(modelsRootDir, slot.canonicalFileName).delete()
        File(legacyInternalRootDir, slot.canonicalFileName).delete()
    }

    fun deleteAllModels() {
        unloadSessionsFromRam()
        modelsRootDir.deleteRecursively()
        if (legacyInternalRootDir.exists() && legacyInternalRootDir.absolutePath != modelsRootDir.absolutePath) {
            legacyInternalRootDir.deleteRecursively()
        }
        ensurePhoneModelFoldersCreated()
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

    fun availableStorageBytes(): Long {
        val statBytes = runCatching { StatFs(modelsRootDir.absolutePath).availableBytes }.getOrDefault(0L)
        if (statBytes > 0L) return statBytes
        val usable = runCatching { modelsRootDir.usableSpace }.getOrDefault(0L)
        if (usable > 0L) return usable
        val internalStat = runCatching { StatFs(context.filesDir.absolutePath).availableBytes }.getOrDefault(0L)
        if (internalStat > 0L) return internalStat
        val internalUsable = runCatching { context.filesDir.usableSpace }.getOrDefault(0L)
        return if (internalUsable > 0L) internalUsable else Long.MAX_VALUE
    }

    fun hasEnoughFreeSpace(requiredBytes: Long): Boolean {
        if (requiredBytes <= 0L) return true
        val freeBytes = availableStorageBytes()
        val safetyReserve = 150L * 1024L * 1024L
        return freeBytes >= requiredBytes + safetyReserve
    }

    fun ensureEnoughFreeSpace(requiredBytes: Long) {
        if (requiredBytes <= 0L) return
        val freeBytes = availableStorageBytes()
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

    data class DiscoveredFolderFile(
        val index: Int,
        val rawName: String,
        val normalizedName: String,
        val parentNameLower: String,
        val sizeBytes: Long,
        val externalLocations: List<String> = emptyList(),
        val hasLcmOrInt8GraphOps: Boolean = false
    ) {
        val normLower: String
            get() = normalizedName.lowercase(Locale.US)
    }

    data class SplitFolderMatchPlan(
        val slotFileIndices: Map<DedicatedModelSlot, List<Int>>
    )

    data class CandidateModelFile(
        val index: Int,
        val fileName: String,
        val parentNameLower: String,
        val fileSizeBytes: Long = 0L
    ) {
        val fileNameLower: String
            get() = fileName.lowercase(Locale.US)
    }

    companion object {
        const val PHONE_MODELS_DIR_NAME = "IG_Models"
        private const val MAX_EXTERNAL_GRAPH_SHELL_BYTES = 15L * 1024L * 1024L
        private const val MAX_GRAPH_SHELL_STREAM_SCAN_BYTES = 25L * 1024L * 1024L
        private const val MAX_PROTO_HEADER_SCAN_BYTES = 48L * 1024L * 1024L
        const val MIN_VALID_COMPANION_BYTES = 64L * 1024L

        @Suppress("DEPRECATION")
        fun resolvePrimaryModelsRootDir(context: Context): File {
            val extMedia = runCatching {
                context.externalMediaDirs?.firstOrNull { it != null }
            }.getOrNull()
            val fallbackExt = runCatching {
                context.getExternalFilesDir(null)
            }.getOrNull()
            val base = when {
                extMedia != null -> File(extMedia, PHONE_MODELS_DIR_NAME)
                fallbackExt != null -> File(fallbackExt, PHONE_MODELS_DIR_NAME)
                else -> File(context.filesDir, PHONE_MODELS_DIR_NAME)
            }
            if (!base.exists()) {
                base.mkdirs()
            }
            return base
        }

        fun resolveTokenizerDir(context: Context): File {
            val primaryTok = File(resolvePrimaryModelsRootDir(context), "tokenizer")
            if (directoryHasPayloadFiles(primaryTok)) return primaryTok
            val legacyTok = File(context.filesDir, "onnx_models/tokenizer")
            if (directoryHasPayloadFiles(legacyTok)) return legacyTok
            if (!primaryTok.exists()) primaryTok.mkdirs()
            return primaryTok
        }

        fun formatPhoneModelsDisplayPath(context: Context, rootDir: File): String {
            val abs = rootDir.absolutePath
            val marker = "Android/media/"
            val idx = abs.indexOf(marker)
            return if (idx >= 0) {
                abs.substring(idx)
            } else {
                "Android/media/${context.packageName}/$PHONE_MODELS_DIR_NAME"
            }
        }

        fun directoryHasPayloadFiles(dir: File): Boolean {
            if (!dir.exists() || !dir.isDirectory) return false
            return dir.listFiles()?.any { f ->
                f.isFile &&
                    !f.name.endsWith(".meta", ignoreCase = true) &&
                    !f.name.endsWith(".repair_tmp", ignoreCase = true) &&
                    !f.name.endsWith(".uploading_tmp", ignoreCase = true) &&
                    !f.name.endsWith(".download_part", ignoreCase = true) &&
                    f.length() > 128L
            } == true
        }

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
        private val CONV_INTEGER_ASCII = "convinteger".toByteArray(Charsets.UTF_8)
        private val TIMESTEP_COND_ASCII = "timestep_cond".toByteArray(Charsets.UTF_8)

        /**
         * Automatically migrates any legacy `onnx_models/unet/` installation into
         * `unet_fast_lcm/` (for ~860 MB Fast LCM / INT8 models) or `unet_standard_sd/`
         * (for 3.44 GB FP32 Standard SD models) with zero file copying via renameTo.
         */
        fun migrateLegacyUnetDir(modelsRootDir: File): ActiveDiffusionEngine? {
            val legacyDir = File(modelsRootDir, "unet")
            if (!legacyDir.exists() || !legacyDir.isDirectory) return null
            val files = legacyDir.listFiles()?.filter { it.isFile }.orEmpty()
            if (files.isEmpty()) {
                legacyDir.deleteRecursively()
                return null
            }

            val totalBytes = files.sumOf { it.length() }
            val namesJoined = files.joinToString(" ") { it.name.lowercase(Locale.US) } + " " +
                (File(legacyDir, "source_name.meta").takeIf { it.exists() }?.readText()?.lowercase(Locale.US).orEmpty())

            val targetEngine = if (
                namesJoined.contains("int8") ||
                namesJoined.contains("lcm") ||
                namesJoined.contains(".onnx.data") ||
                (totalBytes in 1025L..1_800_000_000L && !namesJoined.contains("weights.pb"))
            ) {
                ActiveDiffusionEngine.FAST_LCM
            } else {
                ActiveDiffusionEngine.STANDARD_SD
            }

            val destDir = File(
                modelsRootDir,
                if (targetEngine == ActiveDiffusionEngine.FAST_LCM) "unet_fast_lcm" else "unet_standard_sd"
            ).apply { if (!exists()) mkdirs() }

            val destAlreadyHasGraph = File(destDir, "model.onnx").exists()
            if (!destAlreadyHasGraph) {
                for (f in files) {
                    val targetFile = File(destDir, f.name)
                    if (!f.renameTo(targetFile)) {
                        runCatching {
                            f.copyTo(targetFile, overwrite = true)
                            f.delete()
                        }
                    }
                }
                linkEnginePairFiles(destDir)
            }
            legacyDir.deleteRecursively()
            return targetEngine
        }

        /**
         * Order-independent Internal Code Linker for `(Slot 1A + Slot 1B)` in `unet_fast_lcm/`
         * and `(Slot 2A + Slot 2B)` in `unet_standard_sd/`.
         * Ensures the companion weight file in [engineDir] is automatically renamed in O(1)
         * to the exact external_data `location` expected inside `model.onnx`!
         */
        fun linkEnginePairFiles(engineDir: File): EngineLinkState {
            if (!engineDir.exists() || !engineDir.isDirectory) {
                return EngineLinkState(isGraphInstalled = false, isWeightsInstalled = false, isPairLinked = false)
            }
            val allFiles = engineDir.listFiles()?.filter { it.isFile }.orEmpty()

            // Ensure any .onnx graph file is named model.onnx
            var graphFile = allFiles.firstOrNull { it.name.equals("model.onnx", ignoreCase = true) && it.length() > 1024L }
            if (graphFile == null) {
                val otherOnnx = allFiles.firstOrNull {
                    it.name.endsWith(".onnx", ignoreCase = true) &&
                        !isCompanionWeightFileName(it.name) &&
                        it.length() > 1024L
                }
                if (otherOnnx != null) {
                    val canonical = File(engineDir, "model.onnx")
                    if (otherOnnx.renameTo(canonical)) {
                        graphFile = canonical
                    } else {
                        graphFile = otherOnnx
                    }
                }
            }

            val companionCandidates = engineDir.listFiles()?.filter { f ->
                f.isFile &&
                    f.absolutePath != graphFile?.absolutePath &&
                    !f.name.endsWith(".meta", ignoreCase = true) &&
                    !f.name.endsWith(".repair_tmp", ignoreCase = true) &&
                    !f.name.endsWith(".uploading_tmp", ignoreCase = true) &&
                    !f.name.endsWith(".download_part", ignoreCase = true) &&
                    f.length() >= MIN_VALID_COMPANION_BYTES
            }.orEmpty()

            val graphOrigin = File(engineDir, "graph_source.meta").takeIf { it.exists() }
                ?.readText()?.trim().orEmpty()
                .ifBlank {
                    File(engineDir, "source_name.meta").takeIf { it.exists() }
                        ?.readText()?.substringBefore("+")?.trim().orEmpty()
                }

            val weightsOrigin = File(engineDir, "weights_source.meta").takeIf { it.exists() }
                ?.readText()?.trim().orEmpty()

            if (graphFile == null) {
                val comp = companionCandidates.maxByOrNull { it.length() }
                return if (comp != null) {
                    EngineLinkState(
                        isGraphInstalled = false,
                        isWeightsInstalled = true,
                        isPairLinked = false,
                        weightsBytes = comp.length(),
                        weightsDisplayName = weightsOrigin.ifBlank { comp.name }
                    )
                } else {
                    EngineLinkState(isGraphInstalled = false, isWeightsInstalled = false, isPairLinked = false)
                }
            }

            val expectedLocations = extractExternalDataLocations(graphFile)
            if (expectedLocations.isEmpty()) {
                // Self-contained single-file UNet (must be > 40 MB to be a real standalone model)
                val selfContained = graphFile.length() > 40_000_000L
                val gDisplay = graphOrigin.ifBlank { graphFile.name }
                return EngineLinkState(
                    isGraphInstalled = true,
                    isWeightsInstalled = selfContained || companionCandidates.isNotEmpty(),
                    isPairLinked = selfContained || companionCandidates.isNotEmpty(),
                    isSelfContainedSingleFile = selfContained && companionCandidates.isEmpty(),
                    graphBytes = graphFile.length(),
                    weightsBytes = companionCandidates.sumOf { it.length() },
                    graphDisplayName = gDisplay,
                    weightsDisplayName = companionCandidates.firstOrNull()?.name.orEmpty()
                )
            }

            val primaryExpected = expectedLocations.first()
            var exactCompanion = File(engineDir, primaryExpected).takeIf {
                it.exists() && it.length() >= MIN_VALID_COMPANION_BYTES
            }

            if (exactCompanion == null && companionCandidates.isNotEmpty()) {
                val bestCandidate = companionCandidates.firstOrNull {
                    companionMatchesExpectedLocation(it.name, primaryExpected)
                } ?: companionCandidates.maxByOrNull { it.length() }

                if (bestCandidate != null) {
                    val target = File(engineDir, primaryExpected)
                    if (bestCandidate.renameTo(target)) {
                        exactCompanion = target
                        OnnxQuantGraphRepairer.clearRepairMarkers(engineDir)
                    } else {
                        exactCompanion = bestCandidate
                    }
                }
            }

            val gDisplay = graphOrigin.ifBlank { graphFile.name }
            return if (exactCompanion != null) {
                File(engineDir, "source_name.meta").writeText("$gDisplay + ${exactCompanion.name}")
                EngineLinkState(
                    isGraphInstalled = true,
                    isWeightsInstalled = true,
                    isPairLinked = true,
                    graphBytes = graphFile.length(),
                    weightsBytes = exactCompanion.length(),
                    graphDisplayName = gDisplay,
                    weightsDisplayName = exactCompanion.name,
                    expectedWeightFileName = primaryExpected
                )
            } else {
                EngineLinkState(
                    isGraphInstalled = true,
                    isWeightsInstalled = false,
                    isPairLinked = false,
                    graphBytes = graphFile.length(),
                    graphDisplayName = gDisplay,
                    expectedWeightFileName = primaryExpected
                )
            }
        }

        /**
         * Scans an `.onnx` graph shell InputStream (up to [maxBytes]) in one pass to extract
         * protobuf `external_data` `location` strings AND detect INT8/LCM operators (`ConvInteger` / `timestep_cond`).
         */
        fun scanGraphShellStream(input: InputStream, maxBytes: Long = MAX_GRAPH_SHELL_STREAM_SCAN_BYTES): Pair<List<String>, Boolean> {
            val discovered = LinkedHashSet<String>()
            var hasLcmOrInt8 = false
            val buf = ByteArray(64 * 1024)
            var carry = ByteArray(0)
            var scanned = 0L

            while (scanned < maxBytes) {
                val read = input.read(buf)
                if (read <= 0) break
                scanned += read
                val window = ByteArray(carry.size + read)
                System.arraycopy(carry, 0, window, 0, carry.size)
                System.arraycopy(buf, 0, window, carry.size, read)

                scanProtobufExternalLocations(window, discovered)
                if (!hasLcmOrInt8) {
                    if (containsAsciiIgnoreCase(window, CONV_INTEGER_ASCII) ||
                        containsAsciiIgnoreCase(window, TIMESTEP_COND_ASCII)
                    ) {
                        hasLcmOrInt8 = true
                    }
                }

                val keepLen = minOf(window.size, 160)
                carry = window.copyOfRange(window.size - keepLen, window.size)
            }
            return discovered.toList() to hasLcmOrInt8
        }

        /**
         * Resolves all 8 Core Dedicated Split Slots simultaneously
         * from a flat folder (`Phone > IG`) or nested Hugging Face folder tree!
         */
        fun resolveSplitFolderMatches(allFiles: List<DiscoveredFolderFile>): SplitFolderMatchPlan {
            val assigned = LinkedHashMap<DedicatedModelSlot, List<Int>>()
            val claimedFileIndices = HashSet<Int>()

            val onnxCandidates = allFiles.filter {
                it.normLower.endsWith(".onnx") && !isCompanionWeightFileName(it.normLower)
            }
            val companionFiles = allFiles.filter {
                it.sizeBytes >= MIN_VALID_COMPANION_BYTES && isCompanionWeightFileName(it.normLower)
            }

            // 1. Match explicit non-UNet ONNX slots by keyword or subfolder name (excluding tiny external graph shells)
            val explicitNonUnet = listOf(
                DedicatedModelSlot.TEXT_ENCODER to DiffusionModelSlot.TEXT_ENCODER,
                DedicatedModelSlot.VAE_DECODER to DiffusionModelSlot.VAE_DECODER,
                DedicatedModelSlot.VAE_ENCODER to DiffusionModelSlot.VAE_ENCODER,
                DedicatedModelSlot.FACE_AGING_ENGINE to DiffusionModelSlot.FACE_AGING
            )
            for ((dedSlot, diffSlot) in explicitNonUnet) {
                val match = onnxCandidates.firstOrNull { cand ->
                    cand.index !in claimedFileIndices &&
                        matchesSlot(diffSlot, cand.normLower, cand.parentNameLower) &&
                        (cand.externalLocations.isEmpty() || cand.parentNameLower == diffSlot.folderHint)
                }
                if (match != null) {
                    assigned[dedSlot] = listOf(match.index)
                    claimedFileIndices.add(match.index)
                }
            }

            // 2. Flat-folder size-based fallback for self-contained Slots 3, 4, 5 (e.g. model (3).onnx, model (7).onnx, model (8).onnx)
            if (DedicatedModelSlot.TEXT_ENCODER !in assigned) {
                val textEnc = onnxCandidates.firstOrNull { cand ->
                    cand.index !in claimedFileIndices &&
                        cand.externalLocations.isEmpty() &&
                        cand.sizeBytes in 300_000_000L..750_000_000L
                }
                if (textEnc != null) {
                    assigned[DedicatedModelSlot.TEXT_ENCODER] = listOf(textEnc.index)
                    claimedFileIndices.add(textEnc.index)
                }
            }
            if (DedicatedModelSlot.VAE_DECODER !in assigned) {
                val vaeDec = onnxCandidates.firstOrNull { cand ->
                    cand.index !in claimedFileIndices &&
                        cand.externalLocations.isEmpty() &&
                        cand.sizeBytes in 170_000_000L..230_000_000L
                }
                if (vaeDec != null) {
                    assigned[DedicatedModelSlot.VAE_DECODER] = listOf(vaeDec.index)
                    claimedFileIndices.add(vaeDec.index)
                }
            }
            if (DedicatedModelSlot.VAE_ENCODER !in assigned) {
                val vaeEnc = onnxCandidates.firstOrNull { cand ->
                    cand.index !in claimedFileIndices &&
                        cand.externalLocations.isEmpty() &&
                        cand.sizeBytes in 100_000_000L..169_000_000L
                }
                if (vaeEnc != null) {
                    assigned[DedicatedModelSlot.VAE_ENCODER] = listOf(vaeEnc.index)
                    claimedFileIndices.add(vaeEnc.index)
                }
            }

            // 3. Classify remaining eligible UNet .onnx files into Slot 1A (Fast LCM) and/or Slot 2A (Standard SD)
            val remainingUnetGraphs = onnxCandidates.filter { cand ->
                cand.index !in claimedFileIndices && (
                    cand.externalLocations.isNotEmpty() ||
                        cand.sizeBytes <= MAX_EXTERNAL_GRAPH_SHELL_BYTES && companionFiles.isNotEmpty() ||
                        cand.sizeBytes > 230_000_000L ||
                        cand.normLower.contains("unet") ||
                        cand.parentNameLower.contains("unet")
                    )
            }

            fun findBestCompanionForGraph(
                graph: DiscoveredFolderFile,
                preferCompact: Boolean
            ): DiscoveredFolderFile? {
                val avail = companionFiles.filter { it.index !in claimedFileIndices }
                if (avail.isEmpty()) return null
                val expLoc = graph.externalLocations.firstOrNull()
                if (expLoc != null) {
                    val exactSameDir = avail.filter {
                        it.parentNameLower == graph.parentNameLower &&
                            it.normLower.equals(expLoc, ignoreCase = true)
                    }
                    if (exactSameDir.isNotEmpty()) {
                        return if (preferCompact) exactSameDir.minByOrNull { it.sizeBytes }
                        else exactSameDir.maxByOrNull { it.sizeBytes }
                    }
                    val compatSameDir = avail.filter {
                        it.parentNameLower == graph.parentNameLower &&
                            companionMatchesExpectedLocation(it.rawName, expLoc)
                    }
                    if (compatSameDir.isNotEmpty()) {
                        return if (preferCompact) compatSameDir.minByOrNull { it.sizeBytes }
                        else compatSameDir.maxByOrNull { it.sizeBytes }
                    }
                    val compatAny = avail.filter { companionMatchesExpectedLocation(it.rawName, expLoc) }
                    if (compatAny.isNotEmpty()) {
                        return if (preferCompact) compatAny.minByOrNull { it.sizeBytes }
                        else compatAny.maxByOrNull { it.sizeBytes }
                    }
                }
                return if (preferCompact) {
                    avail.filter { it.sizeBytes <= 1_800_000_000L && !it.normLower.endsWith(".pb") }
                        .minByOrNull { it.sizeBytes }
                } else {
                    avail.filter { it.sizeBytes > 1_800_000_000L || it.normLower.endsWith(".pb") }
                        .maxByOrNull { it.sizeBytes }
                }
            }

            fun isGraphFastLcm(graph: DiscoveredFolderFile): Boolean {
                val locsLower = graph.externalLocations.map { it.lowercase(Locale.US) }
                if (locsLower.any { it.contains("int8") || it.contains("lcm") }) return true
                if (locsLower.any { it == "weights.pb" || it.endsWith("_weights.pb") }) return false
                if (graph.hasLcmOrInt8GraphOps) return true
                if (graph.rawName.contains("int8", true) || graph.rawName.contains("lcm", true) ||
                    graph.parentNameLower.contains("int8") || graph.parentNameLower.contains("lcm")
                ) return true
                if (graph.externalLocations.isEmpty() && graph.sizeBytes in 230_000_001L..1_800_000_000L) {
                    return true
                }
                val expLoc = locsLower.firstOrNull()
                if (expLoc != null) {
                    val matchedComp = companionFiles.firstOrNull {
                        it.parentNameLower == graph.parentNameLower && it.normLower == expLoc
                    } ?: companionFiles.firstOrNull { companionMatchesExpectedLocation(it.rawName, expLoc) }
                    if (matchedComp != null) {
                        return matchedComp.sizeBytes <= 1_800_000_000L && !matchedComp.normLower.endsWith(".pb")
                    }
                }
                // If no protobuf header had external_data (e.g. unit test placeholder), check folder companions
                val sameDirComps = companionFiles.filter { it.parentNameLower == graph.parentNameLower }
                val hasCompactData = sameDirComps.any { it.normLower.contains(".data") && it.sizeBytes <= 1_800_000_000L }
                val hasHeavyPb = sameDirComps.any { it.normLower.endsWith(".pb") || it.sizeBytes > 1_800_000_000L }
                if (hasCompactData && !hasHeavyPb) return true
                if (hasHeavyPb && !hasCompactData) return false
                return graph.normLower != "unet.onnx"
            }

            val fastGraphs = remainingUnetGraphs.filter { isGraphFastLcm(it) }
            val stdGraphs = remainingUnetGraphs.filter { !isGraphFastLcm(it) }

            val chosenFastGraph = fastGraphs.firstOrNull()
            if (chosenFastGraph != null) {
                assigned[DedicatedModelSlot.FAST_LCM_GRAPH] = listOf(chosenFastGraph.index)
                claimedFileIndices.add(chosenFastGraph.index)
                if (chosenFastGraph.externalLocations.isNotEmpty() || chosenFastGraph.sizeBytes <= MAX_EXTERNAL_GRAPH_SHELL_BYTES) {
                    val fastComp = findBestCompanionForGraph(chosenFastGraph, preferCompact = true)
                    if (fastComp != null) {
                        assigned[DedicatedModelSlot.FAST_LCM_WEIGHTS] = listOf(fastComp.index)
                        claimedFileIndices.add(fastComp.index)
                    }
                }
            }

            val chosenStdGraph = stdGraphs.firstOrNull { it.index !in claimedFileIndices }
                ?: fastGraphs.drop(1).firstOrNull { it.index !in claimedFileIndices }
            if (chosenStdGraph != null) {
                assigned[DedicatedModelSlot.STANDARD_SD_GRAPH] = listOf(chosenStdGraph.index)
                claimedFileIndices.add(chosenStdGraph.index)
                if (chosenStdGraph.externalLocations.isNotEmpty() || chosenStdGraph.sizeBytes <= MAX_EXTERNAL_GRAPH_SHELL_BYTES) {
                    val stdComp = findBestCompanionForGraph(chosenStdGraph, preferCompact = false)
                    if (stdComp != null) {
                        assigned[DedicatedModelSlot.STANDARD_SD_WEIGHTS] = listOf(stdComp.index)
                        claimedFileIndices.add(stdComp.index)
                    }
                }
            }

            // 4. Even if one of the graph shells wasn't downloaded yet, still populate unclaimed weight slots!
            if (DedicatedModelSlot.FAST_LCM_WEIGHTS !in assigned) {
                val looseFastWeight = companionFiles.firstOrNull {
                    it.index !in claimedFileIndices &&
                        !it.normLower.endsWith(".pb") &&
                        (it.normLower.contains("int8") || it.sizeBytes <= 1_800_000_000L)
                }
                if (looseFastWeight != null) {
                    assigned[DedicatedModelSlot.FAST_LCM_WEIGHTS] = listOf(looseFastWeight.index)
                    claimedFileIndices.add(looseFastWeight.index)
                }
            }
            if (DedicatedModelSlot.STANDARD_SD_WEIGHTS !in assigned) {
                val looseStdWeight = companionFiles.firstOrNull {
                    it.index !in claimedFileIndices &&
                        (it.normLower.endsWith(".pb") || it.sizeBytes > 1_800_000_000L)
                }
                if (looseStdWeight != null) {
                    assigned[DedicatedModelSlot.STANDARD_SD_WEIGHTS] = listOf(looseStdWeight.index)
                    claimedFileIndices.add(looseStdWeight.index)
                }
            }

            // 5. Slot 6: Tokenizer files (vocab.json & merges.txt)
            val tokIndices = mutableListOf<Int>()
            allFiles.firstOrNull { it.normLower == "vocab.json" && it.sizeBytes > 128L }?.let {
                tokIndices.add(it.index)
            }
            allFiles.firstOrNull { it.normLower == "merges.txt" && it.sizeBytes > 128L }?.let {
                tokIndices.add(it.index)
            }
            if (tokIndices.isNotEmpty()) {
                assigned[DedicatedModelSlot.TOKENIZER] = tokIndices
            }

            return SplitFolderMatchPlan(slotFileIndices = assigned)
        }

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
                    val compatMatch = filesInDir.firstOrNull {
                        it.absolutePath != onnxFile.absolutePath &&
                            !it.name.endsWith(".meta") &&
                            !it.name.endsWith(".repair_tmp") &&
                            !it.name.endsWith(".uploading_tmp") &&
                            !it.name.endsWith(".download_part") &&
                            it.length() >= MIN_VALID_COMPANION_BYTES &&
                            (it.name.equals(reqName, ignoreCase = true) ||
                                companionMatchesExpectedLocation(it.name, reqName))
                    }
                    if (compatMatch != null) {
                        runCatching { compatMatch.renameTo(exact) }
                        !exact.exists() && !compatMatch.exists()
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
            val legacyFolderMatch = (slot == DiffusionModelSlot.UNET && parentNameLower == "unet")
            if ((parentNameLower == slot.folderHint || legacyFolderMatch) && normLower == "model.onnx") return true
            return false
        }

        fun classifyUnetVariant(sourceFileLabel: String, totalBytes: Long): UnetVariant {
            if (totalBytes <= 1024L) return UnetVariant.NONE
            val lower = sourceFileLabel.lowercase(Locale.US)
            if (totalBytes > 2_300_000_000L && !lower.contains("lcm")) {
                return UnetVariant.HEAVY_FP32
            }
            if (lower.contains("lcm") ||
                lower.contains("int8") ||
                lower.contains(".onnx.data") ||
                lower.contains(".onnx_data") ||
                totalBytes <= 1_450_000_000L
            ) {
                return UnetVariant.FAST_LCM_INT8
            }
            return UnetVariant.COMPACT_FP16
        }

        private fun collectEligibleUnetCandidates(
            candidates: List<CandidateModelFile>,
            claimedNonUnetIndices: Set<Int>,
            coreMatchedParents: Set<String>,
            allFileNamesByParent: Map<String, List<String>>
        ): List<CandidateModelFile> {
            return candidates.filter { cand ->
                if (cand.index in claimedNonUnetIndices) return@filter false
                if (matchesSlot(DiffusionModelSlot.UNET, cand.fileNameLower, cand.parentNameLower)) {
                    return@filter true
                }
                val normCand = normalizeBrowserDuplicateName(cand.fileNameLower).lowercase(Locale.US)
                if (normCand == "model.onnx") {
                    val siblings = allFileNamesByParent[cand.parentNameLower].orEmpty()
                    val hasExternalWeights = siblings.any { isCompanionWeightFileName(it) }
                    val sharesDiffusionFolder = cand.parentNameLower in coreMatchedParents
                    (cand.fileSizeBytes <= MAX_EXTERNAL_GRAPH_SHELL_BYTES && hasExternalWeights) ||
                        cand.fileSizeBytes > 1_500_000_000L ||
                        (cand.fileSizeBytes == 0L && sharesDiffusionFolder)
                } else {
                    false
                }
            }
        }

        fun hasMultipleUnetCandidates(
            candidates: List<CandidateModelFile>,
            allFileNamesByParent: Map<String, List<String>>
        ): Boolean {
            val claimed = HashSet<Int>()
            val coreParents = HashSet<String>()
            for (slot in DiffusionModelSlot.entries) {
                if (slot == DiffusionModelSlot.UNET) continue
                val found = candidates.firstOrNull { cand ->
                    cand.index !in claimed && matchesSlot(slot, cand.fileNameLower, cand.parentNameLower)
                }
                if (found != null) {
                    claimed.add(found.index)
                    coreParents.add(found.parentNameLower)
                }
            }
            return collectEligibleUnetCandidates(candidates, claimed, coreParents, allFileNamesByParent).size >= 2
        }

        private fun scoreUnetCandidate(
            cand: CandidateModelFile,
            allFileNamesByParent: Map<String, List<String>>,
            companionSizesByParent: Map<String, Map<String, Long>>,
            preferredUnetMode: PreferredUnetImportMode
        ): Int {
            val normCand = normalizeBrowserDuplicateName(cand.fileNameLower).lowercase(Locale.US)
            val stem = normCand.removeSuffix(".onnx")
            val siblings = allFileNamesByParent[cand.parentNameLower].orEmpty()
            val compSizes = companionSizesByParent[cand.parentNameLower].orEmpty()

            val stemDataNames = listOf(
                "${normCand}.data",
                "${normCand}_data",
                "${stem}.onnx.data",
                "${stem}.onnx_data",
                "${stem}_int8.onnx.data",
                "${stem}_int8.onnx_data",
                "${stem}_quant.onnx.data",
                "${stem}_quant.onnx_data"
            )
            val matchedStemData = siblings.firstOrNull { it in stemDataNames }
                ?: siblings.firstOrNull {
                    it.startsWith("${stem}_") && (it.endsWith(".onnx.data") || it.endsWith(".onnx_data"))
                }

            val hasLcmOrInt8Keyword = normCand.contains("lcm") ||
                normCand.contains("int8") ||
                cand.parentNameLower.contains("lcm") ||
                cand.parentNameLower.contains("int8") ||
                (matchedStemData != null && matchedStemData.contains("int8"))

            val stemDataBytes = matchedStemData?.let { compSizes[it] } ?: 0L
            val weightsPbBytes = compSizes["weights.pb"]
                ?: compSizes["wights.pb"]
                ?: compSizes["${stem}_weights.pb"]
                ?: compSizes["${stem}_wights.pb"]
                ?: compSizes.entries.firstOrNull { it.key.endsWith(".pb") }?.value
                ?: 0L

            val estimatedTotalBytes = when {
                stemDataBytes > 0L -> cand.fileSizeBytes + stemDataBytes
                cand.fileSizeBytes > MAX_EXTERNAL_GRAPH_SHELL_BYTES -> cand.fileSizeBytes
                weightsPbBytes > 0L -> cand.fileSizeBytes + weightsPbBytes
                else -> cand.fileSizeBytes
            }

            val isFastCompactUnet = hasLcmOrInt8Keyword ||
                (matchedStemData != null && (stemDataBytes == 0L || stemDataBytes <= 1_600_000_000L)) ||
                (estimatedTotalBytes in 200_000_000L..1_600_000_000L)

            val isHeavyFp32Unet = estimatedTotalBytes > 2_300_000_000L ||
                (matchedStemData == null && normCand == "unet.onnx" && ("weights.pb" in siblings || "wights.pb" in siblings))

            return when (preferredUnetMode) {
                PreferredUnetImportMode.AUTO_FAST_MOBILE,
                PreferredUnetImportMode.FORCE_FAST_LCM -> {
                    when {
                        hasLcmOrInt8Keyword -> 100
                        isFastCompactUnet -> 90
                        !isHeavyFp32Unet -> 50
                        else -> 20
                    }
                }
                PreferredUnetImportMode.FORCE_STANDARD_SD -> {
                    when {
                        isHeavyFp32Unet -> 100
                        normCand == "unet.onnx" -> 90
                        else -> 30
                    }
                }
            }
        }

        fun resolveFolderSlotMatches(
            candidates: List<CandidateModelFile>,
            allFileNamesByParent: Map<String, List<String>>,
            preferredUnetMode: PreferredUnetImportMode = PreferredUnetImportMode.AUTO_FAST_MOBILE,
            companionSizesByParent: Map<String, Map<String, Long>> = emptyMap()
        ): Map<DiffusionModelSlot, Int> {
            val matched = LinkedHashMap<DiffusionModelSlot, Int>()
            val claimedIndices = HashSet<Int>()

            for (slot in DiffusionModelSlot.entries) {
                if (slot == DiffusionModelSlot.UNET) continue
                val found = candidates.firstOrNull { cand ->
                    cand.index !in claimedIndices &&
                        matchesSlot(slot, cand.fileNameLower, cand.parentNameLower)
                }
                if (found != null) {
                    matched[slot] = found.index
                    claimedIndices.add(found.index)
                }
            }

            if (DiffusionModelSlot.TEXT_ENCODER !in matched) {
                val textEnc = candidates.filter { it.index !in claimedIndices }.firstOrNull { cand ->
                    cand.fileSizeBytes in 300_000_000L..750_000_000L
                }
                if (textEnc != null) {
                    matched[DiffusionModelSlot.TEXT_ENCODER] = textEnc.index
                    claimedIndices.add(textEnc.index)
                }
            }
            if (DiffusionModelSlot.VAE_DECODER !in matched) {
                val vaeDec = candidates.filter { it.index !in claimedIndices }.firstOrNull { cand ->
                    cand.fileSizeBytes in 170_000_000L..230_000_000L
                }
                if (vaeDec != null) {
                    matched[DiffusionModelSlot.VAE_DECODER] = vaeDec.index
                    claimedIndices.add(vaeDec.index)
                }
            }
            if (DiffusionModelSlot.VAE_ENCODER !in matched) {
                val vaeEnc = candidates.filter { it.index !in claimedIndices }.firstOrNull { cand ->
                    cand.fileSizeBytes in 100_000_000L..169_000_000L
                }
                if (vaeEnc != null) {
                    matched[DiffusionModelSlot.VAE_ENCODER] = vaeEnc.index
                    claimedIndices.add(vaeEnc.index)
                }
            }

            val coreMatchedParents = matched.entries
                .mapNotNull { entry -> candidates.firstOrNull { it.index == entry.value }?.parentNameLower }
                .toSet()

            val unetCandidates = collectEligibleUnetCandidates(
                candidates = candidates,
                claimedNonUnetIndices = claimedIndices,
                coreMatchedParents = coreMatchedParents,
                allFileNamesByParent = allFileNamesByParent
            )

            val bestUnet = unetCandidates.maxByOrNull { cand ->
                scoreUnetCandidate(
                    cand = cand,
                    allFileNamesByParent = allFileNamesByParent,
                    companionSizesByParent = companionSizesByParent,
                    preferredUnetMode = preferredUnetMode
                )
            }
            if (bestUnet != null) {
                matched[DiffusionModelSlot.UNET] = bestUnet.index
                claimedIndices.add(bestUnet.index)
            }

            return matched
        }

        fun isCompanionWeightFileName(fileName: String): Boolean {
            val lower = normalizeBrowserDuplicateName(fileName).lowercase(Locale.US)
            return lower.endsWith(".onnx_data") ||
                lower.endsWith(".onnx.data") ||
                lower == "weights.pb" ||
                lower == "wights.pb" ||
                lower.endsWith("_weights.pb") ||
                lower.endsWith("_wights.pb") ||
                lower.endsWith(".pb")
        }

        fun companionMatchesExpectedLocation(siblingName: String, expectedLoc: String): Boolean {
            val sNorm = normalizeBrowserDuplicateName(siblingName).lowercase(Locale.US)
            val eNorm = expectedLoc.lowercase(Locale.US)
            if (sNorm == eNorm) return true
            if ((eNorm == "weights.pb" || eNorm.endsWith("weights.pb")) &&
                (sNorm == "wights.pb" || sNorm.endsWith("wights.pb") || sNorm.endsWith(".pb"))
            ) return true
            if ((eNorm.contains(".onnx.data") || eNorm.contains(".onnx_data") || eNorm.endsWith(".data")) &&
                (sNorm.contains(".onnx.data") || sNorm.contains(".onnx_data") || sNorm.endsWith(".data"))
            ) return true
            return false
        }

        fun shouldCopyCompanionWeight(
            onnxFileName: String,
            onnxSizeBytes: Long,
            companionFileName: String,
            onnxFileForRefScan: File? = null
        ): Boolean {
            val onnxLower = normalizeBrowserDuplicateName(onnxFileName).lowercase(Locale.US)
            val onnxStem = onnxLower.removeSuffix(".onnx")
            val compLower = normalizeBrowserDuplicateName(companionFileName).lowercase(Locale.US)

            val isStemPrefixedData = compLower.startsWith("${onnxStem}_") &&
                (compLower.endsWith(".onnx.data") || compLower.endsWith(".onnx_data"))

            if (onnxFileForRefScan != null && onnxFileForRefScan.exists()) {
                val protoLocations = extractExternalDataLocations(onnxFileForRefScan)
                if (protoLocations.isNotEmpty()) {
                    return protoLocations.any { companionMatchesExpectedLocation(compLower, it) } ||
                        compLower == "${onnxLower}_data" ||
                        compLower == "${onnxLower}.data" ||
                        compLower == "${onnxStem}.onnx_data" ||
                        compLower == "${onnxStem}.onnx.data" ||
                        compLower == "${onnxStem}_weights.pb" ||
                        isStemPrefixedData
                }
            }

            if (compLower == "${onnxLower}_data" ||
                compLower == "${onnxLower}.data" ||
                compLower == "${onnxStem}.onnx_data" ||
                compLower == "${onnxStem}.onnx.data" ||
                compLower == "${onnxStem}_weights.pb" ||
                isStemPrefixedData
            ) {
                return true
            }

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
