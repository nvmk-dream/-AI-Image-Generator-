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
}
