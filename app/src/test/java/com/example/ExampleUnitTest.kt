package com.example

import com.example.ai.DiffusionSchedulers
import com.example.ai.MandatorySafetyGuard
import com.example.ai.OfflineTamilPromptTranslator
import com.example.ai.OnnxDiffusionEngine
import com.example.data.AppPreferencesState
import com.example.data.ContentSafetyDefaults
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class ExampleUnitTest {

    @Test
    fun contentSafetyPrompt_combinesWithUserPromptWithoutOverwriting() {
        val state = AppPreferencesState(
            contentFilterEnabled = true,
            savedContentSafetyPrompt = ContentSafetyDefaults.DEFAULT_CONTENT_SAFETY_PROMPT,
            savedUserImagePrompt = "Portrait in pink silk saree"
        )

        val combined = state.buildFinalGenerationInput("Graceful woman in rose garden")
        assertTrue(combined.startsWith("Graceful woman in rose garden, "))
        assertTrue(combined.contains("சட்டவிரோதமான தீங்கு விளைவிக்கும் உள்ளடக்கம்"))
        // Verify original saved user prompt and safety prompt remain distinct
        assertEquals("Portrait in pink silk saree", state.savedUserImagePrompt)
        assertEquals(ContentSafetyDefaults.DEFAULT_CONTENT_SAFETY_PROMPT, state.savedContentSafetyPrompt)
    }

    @Test
    fun contentSafetyPrompt_disabledReturnsUserPromptOnly() {
        val state = AppPreferencesState(
            contentFilterEnabled = false,
            savedContentSafetyPrompt = ContentSafetyDefaults.DEFAULT_CONTENT_SAFETY_PROMPT
        )
        assertEquals(
            "Studio fashion portrait",
            state.buildFinalGenerationInput("  Studio fashion portrait  ")
        )
    }

    @Test
    fun fp16Conversion_roundTripsAccurately() {
        val testValues = floatArrayOf(0.0f, 1.0f, -1.0f, 0.18215f, -3.5f, 12.25f)
        for (v in testValues) {
            val half = OnnxDiffusionEngine.floatToHalfBits(v)
            val restored = OnnxDiffusionEngine.halfBitsToFloat(half)
            assertTrue("Expected $v ~= $restored", abs(v - restored) < 0.01f)
        }
    }

    @Test
    fun offlineTamilTranslator_translatesTamilAndStripsSafetyBoilerplateForClip() {
        val raw = "அழகான பட்டுப்புடவை அணிந்த பெண், இயற்கை ஒளி, ${ContentSafetyDefaults.DEFAULT_CONTENT_SAFETY_PROMPT}"
        val clipNormalized = OfflineTamilPromptTranslator.normalizeForClip(raw)
        assertFalse(OfflineTamilPromptTranslator.containsTamil(clipNormalized))
        assertTrue(clipNormalized.contains("traditional silk saree"))
        assertTrue(clipNormalized.contains("soft natural lighting"))
    }

    @Test
    fun schedulers_stepProducesFiniteLatents() {
        val lcm = DiffusionSchedulers.LcmSession(6)
        assertEquals(6, lcm.timesteps.size)
        val init = DiffusionSchedulers.createRandomLatents(size = 4 * 8 * 8, seed = 42L)
        val noise = FloatArray(init.size) { 0.05f }
        val next = lcm.step(noise, 0, init, seed = 42L)
        assertEquals(init.size, next.size)
        assertTrue(next.all { it.isFinite() })

        val euler = DiffusionSchedulers.EulerSession(12)
        assertEquals(12, euler.timesteps.size)
        val scaled = euler.scaleModelInput(init, 0)
        val eulerNext = euler.step(noise, 0, scaled)
        assertTrue(eulerNext.all { it.isFinite() })
    }

    @Test
    fun safetyGuard_blocksProhibitedAndAllowsCreativeAdultFashion() {
        assertTrue(
            MandatorySafetyGuard.validateRequest(
                userPrompt = "Glamour fashion beach swimwear portrait of adult model, golden sunset",
                hasRealPersonReferencePhoto = false
            ).allowed
        )
        assertFalse(
            MandatorySafetyGuard.validateRequest(
                userPrompt = "minor child nude",
                hasRealPersonReferencePhoto = false
            ).allowed
        )
        assertFalse(
            MandatorySafetyGuard.validateRequest(
                userPrompt = "explicit nude photo",
                hasRealPersonReferencePhoto = true
            ).allowed
        )
    }

    @Test
    fun img2ImgStrengthAndNoiseInjection_behavesConsistently() {
        // Low strength (0.25 on 8 steps -> 2 active steps -> startStepIndex = 6)
        assertEquals(6, DiffusionSchedulers.computeImg2ImgStartStep(totalSteps = 8, strength = 0.25f))
        // Medium strength (0.50 on 8 steps -> 4 active steps -> startStepIndex = 4)
        assertEquals(4, DiffusionSchedulers.computeImg2ImgStartStep(totalSteps = 8, strength = 0.50f))
        // High strength (0.90 on 10 steps -> 9 active steps -> startStepIndex = 1)
        assertEquals(1, DiffusionSchedulers.computeImg2ImgStartStep(totalSteps = 10, strength = 0.90f))

        val cleanLatents = FloatArray(4 * 8 * 8) { 0.25f }
        val lcm = DiffusionSchedulers.LcmSession(8)
        val lcmNoisy = lcm.addNoise(cleanLatents, startStepIndex = 4, seed = 99L)
        assertEquals(cleanLatents.size, lcmNoisy.size)
        assertTrue(lcmNoisy.all { it.isFinite() })

        val euler = DiffusionSchedulers.EulerSession(16)
        val eulerNoisy = euler.addNoise(cleanLatents, startStepIndex = 6, seed = 99L)
        assertEquals(cleanLatents.size, eulerNoisy.size)
        assertTrue(eulerNoisy.all { it.isFinite() })
    }

    @Test
    fun pixelNormalization_mapsBytesToMinusOneToOneRange() {
        val black = 0xFF000000.toInt()
        val white = 0xFFFFFFFF.toInt()
        val midRed = 0xFF800000.toInt()
        val pixels = intArrayOf(black, white, midRed)

        val nchw = com.example.ai.ImageBitmapPreprocessor.pixelsToNormalizedFloats(pixels, isNhwc = false)
        assertEquals(9, nchw.size)
        // R channel: black -> -1.0, white -> +1.0, midRed (128) -> ~0.0039
        assertTrue(abs(nchw[0] - (-1.0f)) < 1e-4f)
        assertTrue(abs(nchw[1] - 1.0f) < 1e-4f)
        assertTrue(abs(nchw[2] - ((128f / 127.5f) - 1.0f)) < 1e-4f)

        val nhwc = com.example.ai.ImageBitmapPreprocessor.pixelsToNormalizedFloats(pixels, isNhwc = true)
        assertEquals(9, nhwc.size)
        assertTrue(abs(nhwc[0] - (-1.0f)) < 1e-4f)
        assertTrue(abs(nhwc[3] - 1.0f) < 1e-4f)
    }

    @Test
    fun phase6ModelSlotMatcher_detectsDiffusionAndFaceSwapRepoModels() {
        assertTrue(
            com.example.ai.OnnxModelManager.matchesSlot(
                com.example.ai.DiffusionModelSlot.TEXT_ENCODER,
                "model.onnx",
                "text_encoder"
            )
        )
        assertTrue(
            com.example.ai.OnnxModelManager.matchesSlot(
                com.example.ai.DiffusionModelSlot.FACE_DETECTOR,
                "det_10g.onnx",
                "models"
            )
        )
        assertTrue(
            com.example.ai.OnnxModelManager.matchesSlot(
                com.example.ai.DiffusionModelSlot.FACE_RECOGNIZER,
                "w600k_r50.onnx",
                "models"
            )
        )
        assertTrue(
            com.example.ai.OnnxModelManager.matchesSlot(
                com.example.ai.DiffusionModelSlot.FACE_SWAPPER,
                "inswapper_128.onnx",
                "models"
            )
        )
        assertTrue(
            com.example.ai.OnnxModelManager.matchesSlot(
                com.example.ai.DiffusionModelSlot.FACE_SWAPPER,
                "hyperswap_1b_256.onnx",
                "models"
            )
        )
    }

    @Test
    fun phase6CharacterProfileAndEditorMatrix_produceValidOutputs() {
        val charProfile = com.example.data.CharacterProfile(
            id = "c1",
            name = "Ananya",
            appearancePrompt = "young Indian woman, curly black hair",
            fixedSeed = 48291037L
        )
        assertEquals(
            "young Indian woman, curly black hair, wearing golden saree at temple",
            charProfile.buildCharacterScenePrompt("wearing golden saree at temple")
        )

        val defaultMatrix = com.example.ai.OfflineImageEditorEngine.buildColorMatrixValues(
            com.example.ai.ImageEditorAdjustments()
        )
        assertEquals(20, defaultMatrix.size)
        // Identity diagonal = 1.0f, translations = 0.0f
        assertTrue(abs(defaultMatrix[0] - 1.0f) < 1e-4f)
        assertTrue(abs(defaultMatrix[6] - 1.0f) < 1e-4f)
        assertTrue(abs(defaultMatrix[12] - 1.0f) < 1e-4f)
        assertTrue(abs(defaultMatrix[4]) < 1e-4f)
    }

    @Test
    fun flatFolderModelOnnxAndWeightsPb_matchesAllFourSlotsWithoutDuplication() {
        val parent = "image generator"
        val candidates = listOf(
            com.example.ai.OnnxModelManager.CandidateModelFile(0, "text_encoder.onnx", parent),
            com.example.ai.OnnxModelManager.CandidateModelFile(1, "model.onnx", parent),
            com.example.ai.OnnxModelManager.CandidateModelFile(2, "vae_decoder.onnx", parent),
            com.example.ai.OnnxModelManager.CandidateModelFile(3, "vae_encoder.onnx", parent)
        )
        val siblings = mapOf(
            parent to listOf(
                "text_encoder.onnx",
                "model.onnx",
                "weights.pb",
                "vae_decoder.onnx",
                "vae_encoder.onnx",
                "scheduler_config.json",
                "vocab.json",
                "merges.txt"
            )
        )

        val resolved = com.example.ai.OnnxModelManager.resolveFolderSlotMatches(candidates, siblings)
        assertEquals(4, resolved.size)
        assertEquals(0, resolved[com.example.ai.DiffusionModelSlot.TEXT_ENCODER])
        assertEquals(1, resolved[com.example.ai.DiffusionModelSlot.UNET])
        assertEquals(2, resolved[com.example.ai.DiffusionModelSlot.VAE_DECODER])
        assertEquals(3, resolved[com.example.ai.DiffusionModelSlot.VAE_ENCODER])

        // Verify weights.pb is ONLY copied for the 1.09 MB model.onnx shell that references it,
        // and NEVER duplicated into 492 MB text_encoder, 198 MB vae_decoder, or 136 MB vae_encoder!
        val tmpShell = java.io.File.createTempFile("unet_shell_", ".onnx")
        try {
            tmpShell.writeBytes("onnx_proto_header_location_weights.pb_tensor_raw".toByteArray())
            assertTrue(
                com.example.ai.OnnxModelManager.shouldCopyCompanionWeight(
                    onnxFileName = "model.onnx",
                    onnxSizeBytes = 1_140_000L,
                    companionFileName = "weights.pb",
                    onnxFileForRefScan = tmpShell
                )
            )
        } finally {
            tmpShell.delete()
        }

        assertFalse(
            com.example.ai.OnnxModelManager.shouldCopyCompanionWeight(
                onnxFileName = "text_encoder.onnx",
                onnxSizeBytes = 492_540_000L,
                companionFileName = "weights.pb"
            )
        )
        assertFalse(
            com.example.ai.OnnxModelManager.shouldCopyCompanionWeight(
                onnxFileName = "vae_decoder.onnx",
                onnxSizeBytes = 198_080_000L,
                companionFileName = "weights.pb"
            )
        )
        assertFalse(
            com.example.ai.OnnxModelManager.shouldCopyCompanionWeight(
                onnxFileName = "vae_encoder.onnx",
                onnxSizeBytes = 136_760_000L,
                companionFileName = "weights.pb"
            )
        )
        assertFalse(
            com.example.ai.OnnxModelManager.shouldCopyCompanionWeight(
                onnxFileName = "det_10g.onnx",
                onnxSizeBytes = 16_900_000L,
                companionFileName = "weights.pb"
            )
        )
    }

    @Test
    fun protobufExternalDataScanner_detectsWeightsPbAndModelOnnxDataAndMissingStatus() {
        val dir = java.nio.file.Files.createTempDirectory("onnx_ext_test_").toFile()
        try {
            val unetOnnx = java.io.File(dir, "model.onnx")
            val locationBytes = "weights.pb".toByteArray(Charsets.UTF_8)
            val protoPayload = java.io.ByteArrayOutputStream().apply {
                write("onnx_header_padding_".toByteArray(Charsets.UTF_8))
                write(byteArrayOf(0x0A, 0x08))
                write("location".toByteArray(Charsets.UTF_8))
                write(byteArrayOf(0x12, locationBytes.size.toByte()))
                write(locationBytes)
                write(ByteArray(2048))
            }.toByteArray()
            unetOnnx.writeBytes(protoPayload)

            val extracted = com.example.ai.OnnxModelManager.extractExternalDataLocations(unetOnnx)
            assertEquals(listOf("weights.pb"), extracted)

            // Before weights.pb is added -> reported as missing!
            assertEquals(
                listOf("weights.pb"),
                com.example.ai.OnnxModelManager.findMissingCompanionFiles(unetOnnx)
            )

            // Add valid companion weights.pb (>= 64 KB) -> no longer missing!
            val weightsFile = java.io.File(dir, "weights.pb")
            weightsFile.writeBytes(ByteArray(80 * 1024) { 1 })
            assertTrue(com.example.ai.OnnxModelManager.findMissingCompanionFiles(unetOnnx).isEmpty())

            // Even if an .onnx graph is > 15 MB (e.g. 20 MB quantized graph), explicit protobuf external_data is honored
            assertTrue(
                com.example.ai.OnnxModelManager.shouldCopyCompanionWeight(
                    onnxFileName = "unet.onnx",
                    onnxSizeBytes = 20_000_000L,
                    companionFileName = "weights (1).pb",
                    onnxFileForRefScan = unetOnnx
                )
            )
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun browserDuplicateNormalizerAndDownloadCatalog_haveValidEntries() {
        assertEquals(
            "weights.pb",
            com.example.ai.OnnxModelManager.normalizeBrowserDuplicateName("weights (1).pb")
        )
        assertEquals(
            "model.onnx.data",
            com.example.ai.OnnxModelManager.normalizeBrowserDuplicateName("model.onnx (1).data")
        )
        assertEquals(
            "model.onnx.data",
            com.example.ai.OnnxModelManager.normalizeBrowserDuplicateName("model (2).onnx.data")
        )
        assertEquals(
            "unet.onnx",
            com.example.ai.OnnxModelManager.normalizeBrowserDuplicateName("unet (1).onnx")
        )

        val translated = OfflineTamilPromptTranslator.normalizeForClip("அழகான பட்டுப்புடவை அணிந்த பெண்")
        assertTrue(translated.contains("wearing"))
        assertTrue(translated.contains("traditional silk saree"))

        val allLinks = com.example.ai.OnnxModelDownloadCatalog.groups.flatMap { it.links }
        assertTrue(allLinks.size >= 8)
        assertTrue(allLinks.all { it.url.startsWith("https://huggingface.co/") })
        assertTrue(allLinks.any { it.url.endsWith("/unet/model_int8.onnx.data") })
        assertTrue(allLinks.any { it.url.endsWith("/unet/weights.pb") })
    }

    @Test
    fun dualUnetFolder_prioritizesFastLcmByDefaultAndRespectsStandardOverride() {
        val parent = "image generator"
        val candidates = listOf(
            com.example.ai.OnnxModelManager.CandidateModelFile(0, "text_encoder.onnx", parent, 492_540_000L),
            com.example.ai.OnnxModelManager.CandidateModelFile(1, "unet.onnx", parent, 1_750_000L),
            com.example.ai.OnnxModelManager.CandidateModelFile(2, "model.onnx", parent, 2_150_000L),
            com.example.ai.OnnxModelManager.CandidateModelFile(3, "vae_decoder.onnx", parent, 198_080_000L),
            com.example.ai.OnnxModelManager.CandidateModelFile(4, "vae_encoder.onnx", parent, 136_760_000L)
        )
        val siblings = mapOf(
            parent to listOf(
                "text_encoder.onnx",
                "unet.onnx",
                "weights.pb",
                "model.onnx",
                "model.onnx.data",
                "vae_decoder.onnx",
                "vae_encoder.onnx",
                "scheduler_config.json"
            )
        )
        val companionSizes = mapOf(
            parent to mapOf(
                "weights.pb" to 3_438_000_000L,
                "model.onnx.data" to 859_800_000L
            )
        )

        assertTrue(com.example.ai.OnnxModelManager.hasMultipleUnetCandidates(candidates, siblings))

        // Default AUTO_FAST_MOBILE must select ~860 MB Fast LCM (index 2: model.onnx) over 3.44 GB unet.onnx!
        val fastResolved = com.example.ai.OnnxModelManager.resolveFolderSlotMatches(
            candidates = candidates,
            allFileNamesByParent = siblings,
            preferredUnetMode = com.example.ai.PreferredUnetImportMode.AUTO_FAST_MOBILE,
            companionSizesByParent = companionSizes
        )
        assertEquals(2, fastResolved[com.example.ai.DiffusionModelSlot.UNET])
        assertEquals(0, fastResolved[com.example.ai.DiffusionModelSlot.TEXT_ENCODER])
        assertEquals(3, fastResolved[com.example.ai.DiffusionModelSlot.VAE_DECODER])
        assertEquals(4, fastResolved[com.example.ai.DiffusionModelSlot.VAE_ENCODER])

        // Explicit FORCE_STANDARD_SD selects 3.44 GB Standard SD (index 1: unet.onnx)
        val standardResolved = com.example.ai.OnnxModelManager.resolveFolderSlotMatches(
            candidates = candidates,
            allFileNamesByParent = siblings,
            preferredUnetMode = com.example.ai.PreferredUnetImportMode.FORCE_STANDARD_SD,
            companionSizesByParent = companionSizes
        )
        assertEquals(1, standardResolved[com.example.ai.DiffusionModelSlot.UNET])
    }

    @Test
    fun lcmDetectorAndThreadTuner_detectsFastLcmDespiteLeftoverPndmSchedulerAndUsesFourBigCores() {
        assertEquals(4, OnnxDiffusionEngine.computePerformanceThreadCount(8))
        assertEquals(4, OnnxDiffusionEngine.computePerformanceThreadCount(6))
        assertEquals(2, OnnxDiffusionEngine.computePerformanceThreadCount(2))

        // ~860 MB model.onnx + model.onnx.data is recognized as LCM-capable even if PNDMScheduler JSON existed
        assertTrue(
            OnnxDiffusionEngine.isLikelyLcmByMetadataAndSize(
                sourceMeta = "model.onnx + model.onnx.data",
                totalUnetBytes = 862_000_000L,
                schedulerConfigText = """{"_class_name": "PNDMScheduler"}"""
            )
        )

        // 3.44 GB unet.onnx + weights.pb is recognized as Standard SD 1.5 (not LCM)
        assertFalse(
            OnnxDiffusionEngine.isLikelyLcmByMetadataAndSize(
                sourceMeta = "unet.onnx + weights.pb",
                totalUnetBytes = 3_440_000_000L,
                schedulerConfigText = """{"_class_name": "PNDMScheduler"}"""
            )
        )
    }

    @Test
    fun flatFolderSizeBasedFallbackAndWightsPbTypo_resolvesAllSlotsCleanly() {
        val parent = "ig"
        val candidates = listOf(
            com.example.ai.OnnxModelManager.CandidateModelFile(0, "model.onnx", parent, 492_540_000L), // Text Encoder by size
            com.example.ai.OnnxModelManager.CandidateModelFile(1, "model.onnx", parent, 198_080_000L), // VAE Decoder by size
            com.example.ai.OnnxModelManager.CandidateModelFile(2, "model.onnx", parent, 136_760_000L), // VAE Encoder by size
            com.example.ai.OnnxModelManager.CandidateModelFile(3, "model.onnx", parent, 3_050_000L)    // UNet shell + model_int8.onnx.data
        )
        val siblings = mapOf(
            parent to listOf(
                "model.onnx",
                "model_int8.onnx.data"
            )
        )
        val resolved = com.example.ai.OnnxModelManager.resolveFolderSlotMatches(candidates, siblings)
        assertEquals(0, resolved[com.example.ai.DiffusionModelSlot.TEXT_ENCODER])
        assertEquals(1, resolved[com.example.ai.DiffusionModelSlot.VAE_DECODER])
        assertEquals(2, resolved[com.example.ai.DiffusionModelSlot.VAE_ENCODER])
        assertEquals(3, resolved[com.example.ai.DiffusionModelSlot.UNET])

        assertTrue(com.example.ai.OnnxModelManager.isCompanionWeightFileName("wights.pb"))
        assertTrue(com.example.ai.OnnxModelManager.companionMatchesExpectedLocation("wights (1).pb", "weights.pb"))
        assertTrue(com.example.ai.OnnxModelManager.companionMatchesExpectedLocation("model_int8.onnx.data", "model_int8.onnx.data"))

        val allLinks = com.example.ai.OnnxModelDownloadCatalog.groups.flatMap { it.links }
        assertTrue(allLinks.any { it.url.contains("TheyCallMeHex/LCM-Dreamshaper-V7-ONNX") })
        assertTrue(allLinks.any { it.url.endsWith("/unet/model_int8.onnx.data") })
    }
}


