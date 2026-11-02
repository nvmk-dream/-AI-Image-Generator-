package com.example.ui.screens

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
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Cancel
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.FolderSpecial
import androidx.compose.material.icons.filled.Security
import androidx.compose.material.icons.filled.Shuffle
import androidx.compose.material.icons.filled.Star
import androidx.compose.material.icons.outlined.StarOutline
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.minimumInteractiveComponentSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
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
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import com.example.ai.GenerationProgress
import com.example.ai.RealDeviceAndModelMetrics
import com.example.ai.SamplerType
import com.example.ai.UnetVariant
import com.example.data.AppPreferencesState
import com.example.data.GeneratedCreation
import com.example.ui.components.CircularGlowIconButton
import com.example.ui.components.FrostedGlassCard
import com.example.ui.theme.DeepPlumText
import com.example.ui.theme.NeonReadyGreen
import com.example.ui.theme.PinkGlowEnd
import com.example.ui.theme.PinkGlowStart
import com.example.ui.theme.PurpleActionEnd
import com.example.ui.theme.PurpleActionStart
import com.example.ui.theme.VibrantPinkPill
import kotlinx.coroutines.delay
import java.io.File
import java.util.Locale
import kotlin.random.Random

@Composable
fun TextToImageScreen(
    preferences: AppPreferencesState,
    metrics: RealDeviceAndModelMetrics,
    userPrompt: String,
    isGenerating: Boolean,
    progress: GenerationProgress?,
    latestCreation: GeneratedCreation?,
    errorMessage: String?,
    isImportingModels: Boolean = false,
    importProgressText: String? = null,
    onQuickSwitchToFastLcmFolder: () -> Unit = {},
    onResolutionChange: (String) -> Unit = {},
    onUserPromptChange: (String) -> Unit,
    onGenerate: (sampler: SamplerType, steps: Int, guidanceScale: Float, seed: Long, negativePrompt: String) -> Unit,
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

    var selectedSampler by remember { mutableStateOf(SamplerType.LCM_FAST) }
    var steps by remember { mutableIntStateOf(6) }
    var guidanceScale by remember { mutableFloatStateOf(1.5f) }
    var seed by remember { mutableLongStateOf(Random.nextLong(100_000L, 999_999_999L)) }
    var negativePrompt by remember {
        mutableStateOf("blurry, low quality, deformed hands, distorted face, extra limbs")
    }
    var showSafetyCombinedPreview by remember { mutableStateOf(false) }
    var liveElapsedSec by remember { mutableIntStateOf(0) }

    LaunchedEffect(isGenerating) {
        if (isGenerating) {
            val startedAt = System.currentTimeMillis()
            liveElapsedSec = 0
            while (true) {
                delay(1000L)
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
            // Top Header
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                CircularGlowIconButton(
                    onClick = onBack,
                    contentDescription = "Back",
                    testTag = "txt2img_back_button"
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
                        text = "Text to Image",
                        color = textColor,
                        style = MaterialTheme.typography.headlineMedium.copy(fontWeight = FontWeight.ExtraBold)
                    )
                    Text(
                        text = "${preferences.resolution} • ${preferences.inferenceMode} • Batch 1",
                        color = subColor,
                        style = MaterialTheme.typography.bodyMedium
                    )
                }
                CircularGlowIconButton(
                    onClick = onOpenContentSafetySettings,
                    contentDescription = "Content Safety Prompt Settings",
                    testTag = "txt2img_open_safety_button"
                ) {
                    Icon(
                        imageVector = Icons.Filled.Security,
                        contentDescription = "Safety Prompt",
                        tint = Color.White
                    )
                }
            }

            // Active UNet Status + Quick Resolution Strip (384×384 Fast vs 512×512 HD)
            FrostedGlassCard(darkTheme = darkTheme) {
                val isSlowFp32 = metrics.unetVariant == UnetVariant.HEAVY_FP32
                Text(
                    text = metrics.activeUnetBadgeLabel(isTamil),
                    color = if (isSlowFp32 && !darkTheme) Color(0xFFE65100) else textColor,
                    style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.ExtraBold)
                )

                if (isSlowFp32) {
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = if (isTamil) {
                            "⚠️ தற்போது 3.44 GB (weights.pb) மாடல் உள்ளது — CPU-வில் 1 Step முடிய ~3 நிமிடம் ஆகும்! உங்கள் Folder-ல் உள்ள வேகமான ~860 MB Fast LCM மாடலுக்கு மாற்ற கீழே தட்டவும்:"
                        } else {
                            "⚠️ Slow 3.44 GB FP32 UNet is loaded (~3 min/step on CPU). Switch to your ~860 MB Fast LCM model in 1 tap:"
                        },
                        color = subColor,
                        style = MaterialTheme.typography.labelSmall
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.Center,
                        modifier = Modifier
                            .fillMaxWidth()
                            .minimumInteractiveComponentSize()
                            .clip(RoundedCornerShape(14.dp))
                            .background(Brush.horizontalGradient(listOf(PinkGlowStart, VibrantPinkPill)))
                            .border(1.dp, Color.White.copy(alpha = 0.85f), RoundedCornerShape(14.dp))
                            .clickable(enabled = !isGenerating && !isImportingModels) {
                                onQuickSwitchToFastLcmFolder()
                            }
                            .padding(vertical = 9.dp, horizontal = 12.dp)
                            .testTag("txt2img_switch_to_fast_lcm_button")
                    ) {
                        if (isImportingModels) {
                            CircularProgressIndicator(
                                color = Color.White,
                                strokeWidth = 2.2.dp,
                                modifier = Modifier.size(18.dp)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                        }
                        Text(
                            text = if (isImportingModels) {
                                importProgressText ?: "Importing Fast LCM..."
                            } else if (isTamil) {
                                "⚡ Fast LCM (~860 MB) மாடலுக்கு மாற்றுக (Folder)"
                            } else {
                                "⚡ Switch to Fast LCM (~860 MB) from Folder"
                            },
                            color = Color.White,
                            style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                        )
                    }
                }

                Spacer(modifier = Modifier.height(8.dp))
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        text = if (isTamil) "வேகம் / அளவு:" else "Speed / Size:",
                        color = subColor,
                        style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.SemiBold)
                    )
                    listOf("384 × 384" to "⚡ 384×384 Fast", "512 × 512" to "✨ 512×512 HD").forEach { (resVal, resLabel) ->
                        val activeRes = preferences.resolution == resVal
                        Box(
                            contentAlignment = Alignment.Center,
                            modifier = Modifier
                                .weight(1f)
                                .minimumInteractiveComponentSize()
                                .clip(RoundedCornerShape(12.dp))
                                .background(
                                    if (activeRes) {
                                        Brush.horizontalGradient(listOf(PurpleActionStart, PurpleActionEnd))
                                    } else {
                                        Brush.horizontalGradient(
                                            listOf(
                                                Color.White.copy(alpha = if (darkTheme) 0.14f else 0.6f),
                                                Color.White.copy(alpha = if (darkTheme) 0.08f else 0.5f)
                                            )
                                        )
                                    }
                                )
                                .border(1.dp, Color.White.copy(alpha = 0.7f), RoundedCornerShape(12.dp))
                                .clickable(enabled = !isGenerating) { onResolutionChange(resVal) }
                                .padding(vertical = 6.dp, horizontal = 8.dp)
                                .testTag("quick_res_${resVal.substringBefore(" ")}")
                        ) {
                            Text(
                                text = resLabel,
                                color = if (activeRes) Color.White else textColor,
                                style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold)
                            )
                        }
                    }
                }
            }

            // 1. User Image Prompt Card (saved separately from Content Safety Prompt!)
            FrostedGlassCard(darkTheme = darkTheme) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        text = if (isTamil) "User Image Prompt (தமிழ் / English)" else "User Image Prompt",
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

                // Quick Tamil & English Sample Prompts
                val sampleChips = listOf(
                    "அழகான பட்டுப்புடவை அணிந்த பெண், இயற்கை ஒளி, சிரிக்கும் முகம்",
                    "Graceful Indian woman in rose pink silk saree, golden sunset bokeh, photorealistic",
                    "Modern fashion studio portrait, royal purple evening gown, soft rim lighting"
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
                                .testTag("sample_prompt_chip_$idx")
                        ) {
                            Text(
                                text = if (idx == 0) "தமிழ் Sample" else "Sample ${idx + 1}",
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
                                "உங்கள் கற்பனைப் படத்தை தமிழ் அல்லது ஆங்கிலத்தில் எழுதுங்கள்..."
                            } else {
                                "Describe your image in Tamil or English..."
                            }
                        )
                    },
                    minLines = 3,
                    maxLines = 5,
                    modifier = Modifier
                        .fillMaxWidth()
                        .testTag("user_image_prompt_input")
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

            // 2. Fast LCM / Standard Euler Controls Card
            FrostedGlassCard(darkTheme = darkTheme) {
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
                                    steps = sampler.defaultSteps
                                    guidanceScale = sampler.defaultCfg
                                }
                                .padding(vertical = 8.dp, horizontal = 6.dp)
                                .testTag("sampler_${sampler.name.lowercase()}")
                        ) {
                            Text(
                                text = if (sampler == SamplerType.LCM_FAST) "⚡ Fast LCM" else "🎨 Standard SD",
                                color = if (selected) Color.White else textColor,
                                style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                            )
                        }
                    }
                }

                Spacer(modifier = Modifier.height(10.dp))

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
                    valueRange = if (selectedSampler == SamplerType.LCM_FAST) 4f..10f else 10f..25f
                )

                // Generate or Cancel Button
                val modelsReady = metrics.isTxt2ImgReady
                val btnShape = RoundedCornerShape(22.dp)

                if (isGenerating) {
                    Column(modifier = Modifier.fillMaxWidth()) {
                        val pct = (progress?.percent ?: 5).coerceIn(1, 100)
                        val displaySec = maxOf(liveElapsedSec, ((progress?.elapsedMs ?: 0L) / 1000L).toInt())
                        val timerText = String.format(Locale.US, "%02d:%02ds", displaySec / 60, displaySec % 60)
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text(
                                text = progress?.stageTitle ?: "Generating...",
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
                                .testTag("cancel_generation_button")
                        ) {
                            Icon(
                                imageVector = Icons.Filled.Cancel,
                                contentDescription = null,
                                tint = Color.White
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = if (isTamil) "Cancel (நிறுத்து)" else "Cancel Generation",
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
                                    if (modelsReady) {
                                        listOf(PinkGlowStart, PinkGlowEnd)
                                    } else {
                                        listOf(PurpleActionStart, PurpleActionEnd)
                                    }
                                )
                            )
                            .border(1.3.dp, Color.White, btnShape)
                            .clickable {
                                if (!modelsReady) {
                                    onOpenModelsTab()
                                } else {
                                    onGenerate(
                                        selectedSampler,
                                        steps,
                                        guidanceScale,
                                        seed,
                                        negativePrompt
                                    )
                                }
                            }
                            .padding(vertical = 13.dp, horizontal = 16.dp)
                            .testTag("generate_image_button")
                    ) {
                        Icon(
                            imageVector = if (modelsReady) Icons.Filled.AutoAwesome else Icons.Filled.FolderSpecial,
                            contentDescription = null,
                            tint = Color.White
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = when {
                                modelsReady -> "Generate Offline Image"
                                metrics.incompleteSlots.isNotEmpty() -> {
                                    val missing = metrics.incompleteSlots.first().missingCompanionFiles.firstOrNull() ?: "weights.pb"
                                    if (isTamil) {
                                        "UNet $missing இணைக்கவும் (${metrics.installedCount}/4)"
                                    } else {
                                        "Attach Missing $missing (${metrics.installedCount}/4)"
                                    }
                                }
                                isTamil -> "முதலில் ONNX Folder ஏற்றவும் (${metrics.installedCount}/4)"
                                else -> "Upload ONNX Folder First (${metrics.installedCount}/4)"
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

            // 3. Generated Output Preview & Save Card
            if (latestCreation != null && File(latestCreation.filePath).exists()) {
                FrostedGlassCard(darkTheme = darkTheme) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(
                            text = if (isTamil) "உருவாக்கப்பட்ட படம்" else "Generated Result",
                            color = textColor,
                            style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                            modifier = Modifier.weight(1f)
                        )
                        Text(
                            text = "${latestCreation.durationMs / 1000.0}s • ${latestCreation.resolution}",
                            color = NeonReadyGreen,
                            style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                        )
                    }
                    Spacer(modifier = Modifier.height(10.dp))

                    AsyncImage(
                        model = File(latestCreation.filePath),
                        contentDescription = latestCreation.userPrompt,
                        contentScale = ContentScale.Crop,
                        modifier = Modifier
                            .fillMaxWidth()
                            .aspectRatio(1f)
                            .clip(RoundedCornerShape(18.dp))
                            .border(1.4.dp, Color.White, RoundedCornerShape(18.dp))
                    )

                    Spacer(modifier = Modifier.height(10.dp))

                    Row(
                        horizontalArrangement = Arrangement.spacedBy(10.dp),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        // Save to Phone Gallery
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.Center,
                            modifier = Modifier
                                .weight(1f)
                                .minimumInteractiveComponentSize()
                                .clip(RoundedCornerShape(16.dp))
                                .background(Brush.horizontalGradient(listOf(PurpleActionStart, PurpleActionEnd)))
                                .border(1.dp, Color.White, RoundedCornerShape(16.dp))
                                .clickable { onExportToGallery(latestCreation) }
                                .padding(vertical = 10.dp)
                                .testTag("save_to_gallery_button")
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

                        // Favorite Toggle
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.Center,
                            modifier = Modifier
                                .minimumInteractiveComponentSize()
                                .clip(RoundedCornerShape(16.dp))
                                .background(
                                    Brush.horizontalGradient(
                                        if (latestCreation.isFavorite) {
                                            listOf(PinkGlowStart, VibrantPinkPill)
                                        } else {
                                            listOf(Color(0xFF7B5299), Color(0xFF5A3775))
                                        }
                                    )
                                )
                                .border(1.dp, Color.White, RoundedCornerShape(16.dp))
                                .clickable { onToggleFavorite(latestCreation.id) }
                                .padding(horizontal = 16.dp, vertical = 10.dp)
                                .testTag("favorite_result_button")
                        ) {
                            Icon(
                                imageVector = if (latestCreation.isFavorite) Icons.Filled.Star else Icons.Outlined.StarOutline,
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
