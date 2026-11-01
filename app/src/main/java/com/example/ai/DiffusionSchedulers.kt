package com.example.ai

import java.util.Random
import kotlin.math.atan
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

enum class SamplerType(val displayLabel: String, val defaultSteps: Int, val defaultCfg: Float) {
    LCM_FAST("Fast LCM (4–8 Steps)", 6, 1.5f),
    EULER_STANDARD("Standard Euler (15–20 Steps)", 16, 7.0f)
}

/**
 * Provides both:
 * 1. LcmDiscreteScheduler — Fast 4–8 step Latent Consistency Model scheduler
 * 2. EulerDiscreteScheduler — Classic 12–25 step Stable Diffusion 1.5 scheduler
 */
object DiffusionSchedulers {

    private const val NUM_TRAIN_TIMESTEPS = 1000
    private const val BETA_START = 0.00085
    private const val BETA_END = 0.0120
    private const val SIGMA_DATA = 0.5

    // Scaled linear beta schedule used by Stable Diffusion 1.5 & LCM-Dreamshaper
    val alphasCumprod: DoubleArray by lazy {
        val start = sqrt(BETA_START)
        val end = sqrt(BETA_END)
        val alphas = DoubleArray(NUM_TRAIN_TIMESTEPS)
        var prod = 1.0
        for (i in 0 until NUM_TRAIN_TIMESTEPS) {
            val beta = (start + (end - start) * i.toDouble() / (NUM_TRAIN_TIMESTEPS - 1)).pow(2.0)
            val alpha = 1.0 - beta
            prod *= alpha
            alphas[i] = prod
        }
        alphas
    }

    fun createRandomLatents(size: Int, seed: Long, initSigma: Float = 1.0f): FloatArray {
        val rng = Random(seed)
        return FloatArray(size) {
            (rng.nextGaussian() * initSigma).toFloat()
        }
    }

    // =========================================================================
    // 1. LCM Discrete Scheduler (4–8 steps fast inference)
    // =========================================================================
    class LcmSession(numInferenceSteps: Int) {
        val timesteps: IntArray = buildLcmTimesteps(numInferenceSteps)

        private fun buildLcmTimesteps(steps: Int): IntArray {
            val originalSteps = 50
            val k = NUM_TRAIN_TIMESTEPS / originalSteps // 20
            val lcmOriginTimesteps = IntArray(originalSteps) { i -> (i + 1) * k - 1 }
            val stepRatio = lcmOriginTimesteps.size.toDouble() / steps.toDouble()
            return IntArray(steps) { idx ->
                val pos = ((lcmOriginTimesteps.size - 1) - (idx * stepRatio).toInt())
                    .coerceIn(0, lcmOriginTimesteps.lastIndex)
                lcmOriginTimesteps[pos]
            }
        }

        fun step(
            modelOutput: FloatArray,
            stepIndex: Int,
            sample: FloatArray,
            seed: Long
        ): FloatArray {
            val t = timesteps[stepIndex]
            val prevT = if (stepIndex < timesteps.lastIndex) timesteps[stepIndex + 1] else t

            val alphaProdT = alphasCumprod[t]
            val betaProdT = 1.0 - alphaProdT
            val sqrtAlphaT = sqrt(alphaProdT)
            val sqrtBetaT = sqrt(betaProdT)

            // LCM boundary condition scalings (epsilon prediction)
            val scaledTimestep = t.toDouble() * 10.0
            val cSkip = (SIGMA_DATA * SIGMA_DATA) /
                (scaledTimestep * scaledTimestep + SIGMA_DATA * SIGMA_DATA)
            val cOut = scaledTimestep /
                sqrt(scaledTimestep * scaledTimestep + SIGMA_DATA * SIGMA_DATA)

            val denoised = FloatArray(sample.size)
            for (i in sample.indices) {
                val x0Pred = (sample[i].toDouble() - sqrtBetaT * modelOutput[i].toDouble()) / sqrtAlphaT
                denoised[i] = (cOut * x0Pred + cSkip * sample[i].toDouble()).toFloat()
            }

            if (stepIndex == timesteps.lastIndex) {
                return denoised
            }

            val alphaProdPrev = alphasCumprod[prevT]
            val betaProdPrev = 1.0 - alphaProdPrev
            val sqrtAlphaPrev = sqrt(alphaProdPrev)
            val sqrtBetaPrev = sqrt(betaProdPrev)

            val stepRng = Random(seed + stepIndex * 31L + 7L)
            val prevSample = FloatArray(sample.size)
            for (i in sample.indices) {
                val noise = stepRng.nextGaussian()
                prevSample[i] = (sqrtAlphaPrev * denoised[i].toDouble() + sqrtBetaPrev * noise).toFloat()
            }
            return prevSample
        }
    }

    // =========================================================================
    // 2. Euler Discrete Scheduler (Standard SD 1.5 12–25 steps)
    // =========================================================================
    class EulerSession(numInferenceSteps: Int) {
        val timesteps: IntArray
        val sigmas: DoubleArray

        init {
            val stepRatio = (NUM_TRAIN_TIMESTEPS - 1).toDouble() / (numInferenceSteps - 1).coerceAtLeast(1)
            timesteps = IntArray(numInferenceSteps) { i ->
                ((NUM_TRAIN_TIMESTEPS - 1) - (i * stepRatio)).toInt().coerceIn(0, NUM_TRAIN_TIMESTEPS - 1)
            }
            sigmas = DoubleArray(numInferenceSteps + 1) { i ->
                if (i == numInferenceSteps) {
                    0.0
                } else {
                    val alpha = alphasCumprod[timesteps[i]]
                    sqrt((1.0 - alpha) / alpha)
                }
            }
        }

        val initNoiseSigma: Float
            get() = sigmas.firstOrNull()?.toFloat() ?: 1.0f

        fun scaleModelInput(sample: FloatArray, stepIndex: Int): FloatArray {
            val sigma = sigmas[stepIndex]
            val factor = (1.0 / sqrt(sigma * sigma + 1.0)).toFloat()
            return FloatArray(sample.size) { i -> sample[i] * factor }
        }

        fun step(
            modelOutput: FloatArray,
            stepIndex: Int,
            sample: FloatArray
        ): FloatArray {
            val sigma = sigmas[stepIndex]
            val sigmaNext = sigmas[stepIndex + 1]
            val dt = sigmaNext - sigma
            return FloatArray(sample.size) { i ->
                val predOrig = sample[i].toDouble() - sigma * modelOutput[i].toDouble()
                val derivative = (sample[i].toDouble() - predOrig) / sigma.coerceAtLeast(1e-6)
                (sample[i].toDouble() + derivative * dt).toFloat()
            }
        }
    }

    /**
     * Guidance scale Fourier/sinusoidal embedding `timestep_cond` required by LCM UNet models.
     */
    fun buildGuidanceScaleEmbedding(w: Float, embeddingDim: Int = 256): FloatArray {
        val halfDim = embeddingDim / 2
        val wScaled = w * 1000.0
        val logMax = kotlin.math.ln(10000.0) / (halfDim - 1).coerceAtLeast(1)
        val out = FloatArray(embeddingDim)
        for (i in 0 until halfDim) {
            val freq = kotlin.math.exp(-i * logMax)
            val arg = wScaled * freq
            out[i] = sin(arg).toFloat()
            out[halfDim + i] = cos(arg).toFloat()
        }
        return out
    }
}
