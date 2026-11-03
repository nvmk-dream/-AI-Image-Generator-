package com.example.ui.screens

import android.net.Uri
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.AddPhotoAlternate
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Cancel
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.FolderSpecial
import androidx.compose.material.icons.filled.Security
import androidx.compose.material.icons.filled.Shuffle
import androidx.compose.material.icons.filled.Star
import androidx.compose.material.icons.outlined.StarOutline
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.minimumInteractiveComponentSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import com.example.ai.GenerationProgress
import com.example.ai.RealDeviceAndModelMetrics
import com.example.ai.SamplerType
import com.example.data.ActiveDiffusionEngine
import com.example.data.AppPreferencesState
import com.example.data.GeneratedCreation
import com.example.ui.components.ActiveEngineSwitchCard
import com.example.ui.components.CircularGlowIconButton
import com.example.ui.components.FrostedGlassCard
import com.example.ui.theme.DeepPlumText
import com.example.ui.theme.NeonReadyGreen
import com.example.ui.theme.PinkGlowEnd
import com.example.ui.theme.PinkGlowStart
import com.example.ui.theme.PurpleActionEnd
import com.example.ui.theme.PurpleActionStart
import com.example.ui.theme.VibrantPinkPill
import java.io.File
import kotlin.math.roundToInt
import kotlin.random.Random

private enum class CompareViewMode {
    AFTER,
    BEFORE,
    SIDE_BY_SIDE
}

@Composable
fun ImageToImageScreen(
    preferences: AppPreferencesState,
    metrics: RealDeviceAndModelMetrics,
    selectedPhotoUri: Uri?,
    userPrompt: String,
    isGenerating: Boolean,
    progress: GenerationProgress?,
    latestImg2ImgCreation: GeneratedCreation?,
    latestSourcePreviewPath: String?,
    errorMessage: String?,
    onSelectEngine: (ActiveDiffusionEngine) -> Unit = {},
    onPickReferencePhoto: () -> Unit,
    onUserPromptChange: (String) -> Unit,
    onGenerateImg2Img: (
        sampler: SamplerType,
        strength: Float,
        steps: Int,
        guidanceScale: Float,
        seed: Long,
        negativePrompt: String
    ) -> Unit,
    onCancelGeneration: () -> Unit,
    onExportToGallery: (GeneratedCreation) -> Unit,
    onToggleFavorite: (String) -> Unit,
    onOpenModelsTab: () -> Unit,
    onOpenContentSafetySettings: () -> Unit,
    onBack: () -> Unit,
    modifier: Modifier = Modifier
) {
    BackHandler { onBack() }

    val isTamil = preferences.isTamil
    val darkTheme = preferences.darkTheme
    val textColor = if (darkTheme) Color.White else DeepPlumText
    val subColor = if (darkTheme) Color(0xFFE1C9F7) else Color(0xFF4F376E)

    var selectedSampler by remember {
        mutableStateOf(
            if (preferences.activeEngine == ActiveDiffusionEngine.FAST_LCM) SamplerType.LCM_FAST
            else SamplerType.EULER_STANDARD
        )
    }
    var strength by remember { mutableFloatStateOf(0.60f) }
    var steps by remember {
        mutableIntStateOf(if (preferences.activeEngine == ActiveDiffusionEngine.FAST_LCM) 8 else 18)
    }
    var guidanceScale by remember {
        mutableFloatStateOf(if (preferences.activeEngine == ActiveDiffusionEngine.FAST_LCM) 1.5f else 7.0f)
    }
    var seed by remember { mutableLongStateOf(Random.nextLong(100_000L, 999_999_999L)) }
    var negativePrompt by remember {
        mutableStateOf("blurry, low quality, deformed hands, distorted face, extra limbs")
    }
    var showSafetyCombinedPreview by remember { mutableStateOf(false) }
    var compareMode by remember { mutableStateOf(CompareViewMode.AFTER) }
    var liveElapsedSec by remember { androidx.compose.runtime.mutableIntStateOf(0) }

    androidx.compose.runtime.LaunchedEffect(preferences.activeEngine) {
        if (preferences.activeEngine == ActiveDiffusionEngine.FAST_LCM) {
            selectedSampler = SamplerType.LCM_FAST
            steps = 8
            guidanceScale = 1.5f
        } else {
            selectedSampler = SamplerType.EULER_STANDARD
            steps = 18
            guidanceScale = 7.0f
        }
    }

    androidx.compose.runtime.LaunchedEffect(isGenerating) {
        if (isGenerating) {
            val startedAt = System.currentTimeMillis()
            liveElapsedSec = 0
            while (true) {
                kotlinx.coroutines.delay(1000L)
                liveElapsedSec = ((System.currentTimeMillis() - startedAt) / 1000L).toInt()
            }
        } else {
            liveElapsedSec = 0
        }
    }

    val finalCombinedPrompt = preferences.buildFinalGenerationInput(userPrompt)

    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = modifier
            .fillMaxSize()
            .windowInsetsPadding(WindowInsets.statusBars)
            .windowInsetsPadding(WindowInsets.navigationBars)
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp)
    ) {
        Column(
            verticalArrangement = Arrangement.spacedBy(12.dp),
            modifier = Modifier
                .fillMaxWidth()
                .widthIn(max = 600.dp)
                .padding(vertical = 10.dp)
        ) {
            // Top Bar
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                CircularGlowIconButton(
                    onClick = onBack,
                    contentDescription = "Back",
                    testTag = "img2img_back_button"
                ) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                        contentDescription = "Back",
                        tint = Color.White
                    )
                }
                Spacer(modifier = Modifier.width(12.dp))
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "Image to Image",
                        color = textColor,
                        style = MaterialTheme.typography.headlineMedium.copy(fontWeight = FontWeight.ExtraBold)
                    )
                    Text(
                        text = "${preferences.resolution} • ${metrics.activeUnetBadgeLabel(isTamil, preferences.activeEngine)}",
                        color = subColor,
                        style = MaterialTheme.typography.bodySmall
                    )
                }
                CircularGlowIconButton(
                    onClick = onOpenContentSafetySettings,
                    contentDescription = "Content Safety Prompt Settings",
                    testTag = "img2img_open_safety_button"
                ) {
                    Icon(
                        imageVector = Icons.Filled.Security,
                        contentDescription = "Safety Prompt",
                        tint = Color.White
                    )
                }
            }

            // Top 1-Touch Active Engine Switch (Fast LCM vs Standard SD)
            ActiveEngineSwitchCard(
                activeEngine = preferences.activeEngine,
                metrics = metrics,
                isTamil = isTamil,
                darkTheme = darkTheme,
                enabled = !isGenerating,
                tagPrefix = "img2img",
                onSelectEngine = onSelectEngine
            )

            // 1. Reference Photo Card (Auto 1:1 Center-Crop Preview)
            FrostedGlassCard(darkTheme = darkTheme) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        text = if (isTamil) "1. Reference புகைப்படம்" else "1. Reference Photo",
                        color = textColor,
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                        modifier = Modifier.weight(1f)
                    )
                    Text(
                        text = if (selectedPhotoUri != null) "Auto Crop ✓" else if (isTamil) "தேர்வு செய்க" else "Required",
                        color = if (selectedPhotoUri != null) NeonReadyGreen else VibrantPinkPill,
                        style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                    )
                }

                Spacer(modifier = Modifier.height(8.dp))

                if (selectedPhotoUri != null) {
                    AsyncImage(
                        model = selectedPhotoUri,
                        contentDescription = "Selected Reference Photo",
                        contentScale = ContentScale.Crop,
                        modifier = Modifier
                            .fillMaxWidth()
                            .aspectRatio(1f)
                            .clip(RoundedCornerShape(18.dp))
                            .border(1.3.dp, Color.White, RoundedCornerShape(18.dp))
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                } else {
                    Box(
                        contentAlignment = Alignment.Center,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(140.dp)
                            .clip(RoundedCornerShape(16.dp))
                            .background(Color.White.copy(alpha = if (darkTheme) 0.08f else 0.48f))
                            .border(1.dp, Color.White.copy(alpha = 0.7f), RoundedCornerShape(16.dp))
                            .clickable(enabled = !isGenerating, onClick = onPickReferencePhoto)
                            .padding(14.dp)
                    ) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            Icon(
                                imageVector = Icons.Filled.AddPhotoAlternate,
                                contentDescription = null,
                                tint = PurpleActionEnd,
                                modifier = Modifier.size(34.dp)
                            )
                            Spacer(modifier = Modifier.height(6.dp))
                            Text(
                                text = if (isTamil) {
                                    "மாற்ற வேண்டிய புகைப்படத்தைத் தேர்வு செய்யவும் (தானாக 512×512 Center-Crop செய்யப்படும்)"
                                } else {
                                    "Tap to pick a photo from Gallery (Auto Center-Cropped without face stretching)"
                                },
                                color = subColor,
                                style = MaterialTheme.typography.bodyMedium,
                                textAlign = TextAlign.Center
                            )
                        }
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                }

                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.Center,
                    modifier = Modifier
                        .fillMaxWidth()
                        .minimumInteractiveComponentSize()
                        .clip(RoundedCornerShape(16.dp))
                        .background(Brush.horizontalGradient(listOf(PurpleActionStart, PurpleActionEnd)))
                        .border(1.dp, Color.White.copy(alpha = 0.85f), RoundedCornerShape(16.dp))
                        .clickable(enabled = !isGenerating, onClick = onPickReferencePhoto)
                        .padding(vertical = 11.dp)
                        .testTag("pick_img2img_photo_button")
                ) {
                    Icon(
                        imageVector = Icons.Filled.AddPhotoAlternate,
                        contentDescription = null,
                        tint = Color.White,
                        modifier = Modifier.size(20.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = if (selectedPhotoUri == null) {
                            if (isTamil) "Gallery-யில் படம் தேர்வு செய்க" else "Choose Reference Photo"
                        } else {
                            if (isTamil) "வேறு படம் மாற்றுக" else "Change Reference Photo"
                        },
                        color = Color.White,
                        style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                    )
                }
            }

            // 2. Creative User Prompt Card + Combined Safety Prompt Preview
            FrostedGlassCard(darkTheme = darkTheme) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        text = if (isTamil) "2. மாற்றத்திற்கான Prompt (தமிழ் / English)" else "2. Transformation Prompt",
                        color = textColor,
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                        modifier = Modifier.weight(1f)
                    )
                    Text(
                        text = if (showSafetyCombinedPreview) "Hide Combined" else "Safety Prompt ✓",
                        color = VibrantPinkPill,
                        style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold),
                        modifier = Modifier
                            .clickable { showSafetyCombinedPreview = !showSafetyCombinedPreview }
                            .padding(4.dp)
                    )
                }

                Spacer(modifier = Modifier.height(6.dp))

                val sampleChips = listOf(
                    "இளஞ்சிவப்பு பட்டுப்புடவை மற்றும் தங்க நகைகள், இயற்கை ஒளி, தெளிவான புகைப்படம்",
                    "Royal silk saree portrait, golden hour warm sunset bokeh, photorealistic studio quality",
                    "Artistic digital painting portrait, soft pastel lighting, intricate details"
                )
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    modifier = Modifier
                        .fillMaxWidth()
                        .horizontalScroll(rememberScrollState())
                ) {
                    sampleChips.forEachIndexed { idx, chipPrompt ->
                        Box(
                            modifier = Modifier
                                .clip(RoundedCornerShape(50))
                                .background(Color.White.copy(alpha = if (darkTheme) 0.14f else 0.65f))
                                .border(1.dp, Color.White.copy(alpha = 0.7f), RoundedCornerShape(50))
                                .clickable(enabled = !isGenerating) { onUserPromptChange(chipPrompt) }
                                .padding(horizontal = 12.dp, vertical = 6.dp)
                                .testTag("img2img_sample_chip_$idx")
                        ) {
                            Text(
                                text = if (idx == 0) "தமிழ் Style" else "Style ${idx + 1}",
                                color = textColor,
                                style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold)
                            )
                        }
                    }
                }

                Spacer(modifier = Modifier.height(8.dp))

                OutlinedTextField(
                    value = userPrompt,
                    onValueChange = onUserPromptChange,
                    enabled = !isGenerating,
                    placeholder = {
                        Text(
                            if (isTamil) {
                                "படத்தை எப்படி மாற்ற வேண்டும் என தமிழ் அல்லது ஆங்கிலத்தில் எழுதுங்கள்..."
                            } else {
                                "Describe the outfit, background, or artistic style..."
                            }
                        )
                    },
                    minLines = 3,
                    maxLines = 5,
                    modifier = Modifier
                        .fillMaxWidth()
                        .testTag("img2img_prompt_input")
                )

                if (showSafetyCombinedPreview) {
                    Spacer(modifier = Modifier.height(8.dp))
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(12.dp))
                            .background(Color.White.copy(alpha = if (darkTheme) 0.08f else 0.55f))
                            .padding(10.dp)
                    ) {
                        Column {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Text(
                                    text = "Final Generation Input:",
                                    color = VibrantPinkPill,
                                    style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                                    modifier = Modifier.weight(1f)
                                )
                                Text(
                                    text = "Edit Safety Prompt >",
                                    color = PurpleActionEnd,
                                    style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                                    modifier = Modifier.clickable(onClick = onOpenContentSafetySettings)
                                )
                            }
                            Spacer(modifier = Modifier.height(4.dp))
                            Text(
                                text = finalCombinedPrompt.ifBlank { "(Empty)" },
                                color = subColor,
                                style = MaterialTheme.typography.bodyMedium
                            )
                        }
                    }
                }
            }

            // 3. Strength Slider + Sampler + Steps + Generate/Cancel Card
            FrostedGlassCard(darkTheme = darkTheme) {
                val strengthPct = (strength * 100f).roundToInt()
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        text = if (isTamil) "மாற்றத்தின் அளவு (Strength)" else "Transformation Strength",
                        color = textColor,
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                        modifier = Modifier.weight(1f)
                    )
                    Text(
                        text = "$strengthPct% (${String.format("%.2f", strength)})",
                        color = VibrantPinkPill,
                        style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.ExtraBold)
                    )
                }

                Slider(
                    value = strength,
                    onValueChange = { strength = (it * 20f).roundToInt() / 20f },
                    enabled = !isGenerating,
                    valueRange = 0.20f..0.90f,
                    modifier = Modifier.testTag("img2img_strength_slider")
                )

                Row(
                    horizontalArrangement = Arrangement.SpaceBetween,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        text = if (isTamil) "20% (சிறிய மாற்றம்)" else "20% (Keep Original)",
                        color = subColor,
                        style = MaterialTheme.typography.labelSmall
                    )
                    Text(
                        text = if (isTamil) "90% (முழுப் புதிய ஸ்டைல்)" else "90% (Strong Restyle)",
                        color = subColor,
                        style = MaterialTheme.typography.labelSmall
                    )
                }

                Spacer(modifier = Modifier.height(10.dp))

                // Fast LCM vs Standard SD
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    SamplerType.entries.forEach { sampler ->
                        val selected = selectedSampler == sampler
                        val shape = RoundedCornerShape(14.dp)
                        Box(
                            contentAlignment = Alignment.Center,
                            modifier = Modifier
                                .weight(1f)
                                .minimumInteractiveComponentSize()
                                .clip(shape)
                                .background(
                                    if (selected) {
                                        Brush.horizontalGradient(listOf(PinkGlowStart, PinkGlowEnd))
                                    } else {
                                        Brush.horizontalGradient(
                                            listOf(
                                                Color.White.copy(alpha = if (darkTheme) 0.14f else 0.55f),
                                                Color.White.copy(alpha = if (darkTheme) 0.08f else 0.45f)
                                            )
                                        )
                                    }
                                )
                                .border(1.dp, Color.White.copy(alpha = 0.7f), shape)
                                .clickable(enabled = !isGenerating) {
                                    selectedSampler = sampler
                                    steps = if (sampler == SamplerType.LCM_FAST) 8 else 18
                                    guidanceScale = sampler.defaultCfg
                                    val targetEngine = if (sampler == SamplerType.LCM_FAST) {
                                        ActiveDiffusionEngine.FAST_LCM
                                    } else {
                                        ActiveDiffusionEngine.STANDARD_SD
                                    }
                                    if (preferences.activeEngine != targetEngine) {
                                        onSelectEngine(targetEngine)
                                    }
                                }
                                .padding(vertical = 8.dp, horizontal = 6.dp)
                                .testTag("img2img_sampler_${sampler.name.lowercase()}")
                        ) {
                            Text(
                                text = if (sampler == SamplerType.LCM_FAST) "⚡ Fast LCM" else "🎨 Standard SD",
                                color = if (selected) Color.White else textColor,
                                style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                            )
                        }
                    }
                }

                Spacer(modifier = Modifier.height(8.dp))

                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        text = "Steps: $steps",
                        color = textColor,
                        style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold),
                        modifier = Modifier.weight(1f)
                    )
                    Text(
                        text = "Seed: $seed",
                        color = subColor,
                        style = MaterialTheme.typography.labelLarge
                    )
                    IconButton(
                        onClick = { seed = Random.nextLong(100_000L, 999_999_999L) },
                        enabled = !isGenerating
                    ) {
                        Icon(
                            imageVector = Icons.Filled.Shuffle,
                            contentDescription = "Randomize Seed",
                            tint = PurpleActionEnd
                        )
                    }
                }

                Slider(
                    value = steps.toFloat(),
                    onValueChange = { steps = it.toInt() },
                    enabled = !isGenerating,
                    valueRange = if (selectedSampler == SamplerType.LCM_FAST) 4f..12f else 10f..25f
                )

                val allFourModelsReady = metrics.isImg2ImgReadyFor(preferences.activeEngine)
                val btnShape = RoundedCornerShape(22.dp)

                if (isGenerating) {
                    Column(modifier = Modifier.fillMaxWidth()) {
                        val pct = (progress?.percent ?: 5).coerceIn(1, 100)
                        val displaySec = maxOf(liveElapsedSec, ((progress?.elapsedMs ?: 0L) / 1000L).toInt())
                        val timerText = java.lang.String.format(java.util.Locale.US, "%02d:%02ds", displaySec / 60, displaySec % 60)
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text(
                                text = progress?.stageTitle ?: "Transforming Image...",
                                color = textColor,
                                style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.SemiBold),
                                modifier = Modifier.weight(1f)
                            )
                            Text(
                                text = "⏱ $timerText • $pct%",
                                color = VibrantPinkPill,
                                style = MaterialTheme.typography.titleSmall.copy(fontWeight = FontWeight.ExtraBold)
                            )
                        }
                        Spacer(modifier = Modifier.height(6.dp))
                        LinearProgressIndicator(
                            progress = { pct / 100f },
                            color = VibrantPinkPill,
                            trackColor = Color.White.copy(alpha = 0.45f),
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(9.dp)
                                .clip(RoundedCornerShape(50))
                        )
                        Spacer(modifier = Modifier.height(10.dp))
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.Center,
                            modifier = Modifier
                                .fillMaxWidth()
                                .minimumInteractiveComponentSize()
                                .clip(btnShape)
                                .background(Color(0xFFD32F2F))
                                .border(1.2.dp, Color.White, btnShape)
                                .clickable(onClick = onCancelGeneration)
                                .padding(vertical = 12.dp)
                                .testTag("img2img_cancel_button")
                        ) {
                            Icon(
                                imageVector = Icons.Filled.Cancel,
                                contentDescription = null,
                                tint = Color.White
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = if (isTamil) "Cancel (நிறுத்து)" else "Cancel Transformation",
                                color = Color.White,
                                style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                            )
                        }
                    }
                } else {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.Center,
                        modifier = Modifier
                            .fillMaxWidth()
                            .minimumInteractiveComponentSize()
                            .shadow(10.dp, btnShape, spotColor = PurpleActionEnd)
                            .clip(btnShape)
                            .background(
                                Brush.horizontalGradient(
                                    if (allFourModelsReady) {
                                        listOf(PinkGlowStart, PinkGlowEnd)
                                    } else {
                                        listOf(PurpleActionStart, PurpleActionEnd)
                                    }
                                )
                            )
                            .border(1.3.dp, Color.White, btnShape)
                            .clickable {
                                if (!allFourModelsReady) {
                                    onOpenModelsTab()
                                } else {
                                    onGenerateImg2Img(
                                        selectedSampler,
                                        strength,
                                        steps,
                                        guidanceScale,
                                        seed,
                                        negativePrompt
                                    )
                                }
                            }
                            .padding(vertical = 13.dp, horizontal = 16.dp)
                            .testTag("generate_img2img_button")
                    ) {
                        Icon(
                            imageVector = if (allFourModelsReady) Icons.Filled.AutoAwesome else Icons.Filled.FolderSpecial,
                            contentDescription = null,
                            tint = Color.White
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = if (allFourModelsReady) {
                                if (preferences.activeEngine == ActiveDiffusionEngine.FAST_LCM) {
                                    "Transform Image (⚡ Fast LCM)"
                                } else {
                                    "Transform Image (🎨 Standard SD)"
                                }
                            } else {
                                metrics.missingSummaryFor(
                                    engine = preferences.activeEngine,
                                    requireImg2Img = true,
                                    isTamil = isTamil
                                )
                            },
                            color = Color.White,
                            style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                        )
                    }
                }

                if (!errorMessage.isNullOrBlank()) {
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = errorMessage,
                        color = Color(0xFFD32F2F),
                        style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.SemiBold)
                    )
                }
            }

            // 4. Before / After Output Comparison & Save Card
            if (latestImg2ImgCreation != null && File(latestImg2ImgCreation.filePath).exists()) {
                val hasBeforePreview = !latestSourcePreviewPath.isNullOrBlank() &&
                    File(latestSourcePreviewPath).exists()

                FrostedGlassCard(darkTheme = darkTheme) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(
                            text = if (isTamil) "புதிய படம் & ஒப்பீடு" else "Transformed Result",
                            color = textColor,
                            style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                            modifier = Modifier.weight(1f)
                        )
                        Text(
                            text = "${latestImg2ImgCreation.durationMs / 1000.0}s • ${latestImg2ImgCreation.resolution}",
                            color = NeonReadyGreen,
                            style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                        )
                    }

                    if (hasBeforePreview) {
                        Spacer(modifier = Modifier.height(8.dp))
                        // Before / After / Side-by-Side Comparison Toggle
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(6.dp),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            val tabs = listOf(
                                CompareViewMode.AFTER to (if (isTamil) "புதியது (After)" else "After"),
                                CompareViewMode.BEFORE to (if (isTamil) "அசல் (Before)" else "Before"),
                                CompareViewMode.SIDE_BY_SIDE to (if (isTamil) "ஒப்பீடு (Split)" else "Compare")
                            )
                            tabs.forEach { (mode, label) ->
                                val active = compareMode == mode
                                val chipShape = RoundedCornerShape(12.dp)
                                Box(
                                    contentAlignment = Alignment.Center,
                                    modifier = Modifier
                                        .weight(1f)
                                        .minimumInteractiveComponentSize()
                                        .clip(chipShape)
                                        .background(
                                            if (active) {
                                                Brush.horizontalGradient(listOf(PinkGlowStart, PinkGlowEnd))
                                            } else {
                                                Brush.horizontalGradient(
                                                    listOf(
                                                        Color.White.copy(alpha = if (darkTheme) 0.14f else 0.58f),
                                                        Color.White.copy(alpha = if (darkTheme) 0.08f else 0.48f)
                                                    )
                                                )
                                            }
                                        )
                                        .border(1.dp, Color.White.copy(alpha = 0.75f), chipShape)
                                        .clickable { compareMode = mode }
                                        .padding(vertical = 7.dp, horizontal = 4.dp)
                                        .testTag("compare_mode_${mode.name.lowercase()}")
                                ) {
                                    Text(
                                        text = label,
                                        color = if (active) Color.White else textColor,
                                        style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold)
                                    )
                                }
                            }
                        }
                    }

                    Spacer(modifier = Modifier.height(10.dp))

                    when {
                        hasBeforePreview && compareMode == CompareViewMode.BEFORE -> {
                            AsyncImage(
                                model = File(latestSourcePreviewPath!!),
                                contentDescription = "Original Cropped Reference",
                                contentScale = ContentScale.Crop,
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .aspectRatio(1f)
                                    .clip(RoundedCornerShape(18.dp))
                                    .border(1.4.dp, Color.White, RoundedCornerShape(18.dp))
                            )
                        }

                        hasBeforePreview && compareMode == CompareViewMode.SIDE_BY_SIDE -> {
                            Row(
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Column(
                                    horizontalAlignment = Alignment.CenterHorizontally,
                                    modifier = Modifier.weight(1f)
                                ) {
                                    AsyncImage(
                                        model = File(latestSourcePreviewPath!!),
                                        contentDescription = "Before Photo",
                                        contentScale = ContentScale.Crop,
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .aspectRatio(1f)
                                            .clip(RoundedCornerShape(14.dp))
                                            .border(1.2.dp, Color.White, RoundedCornerShape(14.dp))
                                    )
                                    Spacer(modifier = Modifier.height(4.dp))
                                    Text(
                                        text = if (isTamil) "அசல் (Before)" else "Before",
                                        color = subColor,
                                        style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold)
                                    )
                                }
                                Column(
                                    horizontalAlignment = Alignment.CenterHorizontally,
                                    modifier = Modifier.weight(1f)
                                ) {
                                    AsyncImage(
                                        model = File(latestImg2ImgCreation.filePath),
                                        contentDescription = "After Photo",
                                        contentScale = ContentScale.Crop,
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .aspectRatio(1f)
                                            .clip(RoundedCornerShape(14.dp))
                                            .border(1.2.dp, Color.White, RoundedCornerShape(14.dp))
                                    )
                                    Spacer(modifier = Modifier.height(4.dp))
                                    Text(
                                        text = if (isTamil) "புதியது (After)" else "After",
                                        color = VibrantPinkPill,
                                        style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold)
                                    )
                                }
                            }
                        }

                        else -> {
                            AsyncImage(
                                model = File(latestImg2ImgCreation.filePath),
                                contentDescription = latestImg2ImgCreation.userPrompt,
                                contentScale = ContentScale.Crop,
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .aspectRatio(1f)
                                    .clip(RoundedCornerShape(18.dp))
                                    .border(1.4.dp, Color.White, RoundedCornerShape(18.dp))
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(10.dp))

                    Row(
                        horizontalArrangement = Arrangement.spacedBy(10.dp),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.Center,
                            modifier = Modifier
                                .weight(1f)
                                .minimumInteractiveComponentSize()
                                .clip(RoundedCornerShape(16.dp))
                                .background(Brush.horizontalGradient(listOf(PurpleActionStart, PurpleActionEnd)))
                                .border(1.dp, Color.White, RoundedCornerShape(16.dp))
                                .clickable { onExportToGallery(latestImg2ImgCreation) }
                                .padding(vertical = 10.dp)
                                .testTag("img2img_save_gallery_button")
                        ) {
                            Icon(
                                imageVector = Icons.Filled.Download,
                                contentDescription = null,
                                tint = Color.White,
                                modifier = Modifier.size(18.dp)
                            )
                            Spacer(modifier = Modifier.width(6.dp))
                            Text(
                                text = if (isTamil) "Gallery-ல் சேமி" else "Save to Gallery",
                                color = Color.White,
                                style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                            )
                        }

                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.Center,
                            modifier = Modifier
                                .minimumInteractiveComponentSize()
                                .clip(RoundedCornerShape(16.dp))
                                .background(
                                    Brush.horizontalGradient(
                                        if (latestImg2ImgCreation.isFavorite) {
                                            listOf(PinkGlowStart, VibrantPinkPill)
                                        } else {
                                            listOf(Color(0xFF7B5299), Color(0xFF5A3775))
                                        }
                                    )
                                )
                                .border(1.dp, Color.White, RoundedCornerShape(16.dp))
                                .clickable { onToggleFavorite(latestImg2ImgCreation.id) }
                                .padding(horizontal = 16.dp, vertical = 10.dp)
                                .testTag("img2img_favorite_button")
                        ) {
                            Icon(
                                imageVector = if (latestImg2ImgCreation.isFavorite) Icons.Filled.Star else Icons.Outlined.StarOutline,
                                contentDescription = "Favorite",
                                tint = Color.White,
                                modifier = Modifier.size(20.dp)
                            )
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))
        }
    }
}
