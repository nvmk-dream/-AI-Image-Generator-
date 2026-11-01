package com.example.ui.screens

import android.net.Uri
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
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
import androidx.compose.material.icons.filled.FaceRetouchingNatural
import androidx.compose.material.icons.filled.FolderSpecial
import androidx.compose.material.icons.filled.Star
import androidx.compose.material.icons.outlined.StarOutline
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.minimumInteractiveComponentSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
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
import java.io.File
import kotlin.math.roundToInt

private enum class FaceSwapCompareMode {
    SWAPPED,
    ORIGINAL_TARGET,
    SIDE_BY_SIDE
}

@Composable
fun FaceSwapScreen(
    preferences: AppPreferencesState,
    metrics: RealDeviceAndModelMetrics,
    sourceFaceUri: Uri?,
    targetImageUri: Uri?,
    targetFallbackCreation: GeneratedCreation?,
    useLatestCreationAsTarget: Boolean,
    isSwapping: Boolean,
    progress: GenerationProgress?,
    latestFaceSwapCreation: GeneratedCreation?,
    latestFaceSwapTargetPreviewPath: String?,
    errorMessage: String?,
    onPickSourceFace: () -> Unit,
    onPickTargetPhoto: () -> Unit,
    onUseLatestCreationAsTarget: () -> Unit,
    onRunFaceSwap: (enableSkinToneMatch: Boolean, blendStrength: Float) -> Unit,
    onCancelSwap: () -> Unit,
    onExportToGallery: (GeneratedCreation) -> Unit,
    onToggleFavorite: (String) -> Unit,
    onOpenModelsTab: () -> Unit,
    onBack: () -> Unit,
    modifier: Modifier = Modifier
) {
    BackHandler { onBack() }

    val isTamil = preferences.isTamil
    val darkTheme = preferences.darkTheme
    val textColor = if (darkTheme) Color.White else DeepPlumText
    val subColor = if (darkTheme) Color(0xFFE1C9F7) else Color(0xFF4F376E)

    var enableSkinToneMatch by remember { mutableStateOf(true) }
    var blendStrength by remember { mutableFloatStateOf(0.92f) }
    var compareMode by remember { mutableStateOf(FaceSwapCompareMode.SWAPPED) }

    val activeTargetModel: Any? = when {
        useLatestCreationAsTarget && targetFallbackCreation != null -> File(targetFallbackCreation.filePath)
        targetImageUri != null -> targetImageUri
        else -> null
    }

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
                    testTag = "faceswap_back_button"
                ) {
                    Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back", tint = Color.White)
                }
                Spacer(modifier = Modifier.width(12.dp))
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "Face Swap",
                        color = textColor,
                        style = MaterialTheme.typography.headlineMedium.copy(fontWeight = FontWeight.ExtraBold)
                    )
                    Text(
                        text = if (isTamil) "Offline ONNX • InSwapper / HyperSwap" else "Offline ONNX • InSwapper & HyperSwap",
                        color = subColor,
                        style = MaterialTheme.typography.bodyMedium
                    )
                }
            }

            // 1. Source Face + Target Photo Side-by-Side Card
            FrostedGlassCard(darkTheme = darkTheme) {
                Text(
                    text = if (isTamil) "1. முகம் & இலக்கு புகைப்படம்" else "1. Source Face & Target Photo",
                    color = textColor,
                    style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                )
                Spacer(modifier = Modifier.height(8.dp))

                Row(
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    // Left: Source Face
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        modifier = Modifier.weight(1f)
                    ) {
                        Box(
                            contentAlignment = Alignment.Center,
                            modifier = Modifier
                                .fillMaxWidth()
                                .aspectRatio(1f)
                                .clip(RoundedCornerShape(16.dp))
                                .background(Color.White.copy(alpha = if (darkTheme) 0.10f else 0.50f))
                                .border(1.2.dp, Color.White.copy(alpha = 0.85f), RoundedCornerShape(16.dp))
                                .clickable(enabled = !isSwapping, onClick = onPickSourceFace)
                                .testTag("faceswap_pick_source_button")
                        ) {
                            if (sourceFaceUri != null) {
                                AsyncImage(
                                    model = sourceFaceUri,
                                    contentDescription = "Source Face",
                                    contentScale = ContentScale.Crop,
                                    modifier = Modifier.fillMaxSize()
                                )
                            } else {
                                Column(
                                    horizontalAlignment = Alignment.CenterHorizontally,
                                    modifier = Modifier.padding(8.dp)
                                ) {
                                    Icon(Icons.Filled.AddPhotoAlternate, contentDescription = null, tint = PurpleActionEnd, modifier = Modifier.size(30.dp))
                                    Spacer(modifier = Modifier.height(4.dp))
                                    Text(
                                        text = if (isTamil) "Source முகம் தேர்வு" else "Pick Source Face",
                                        color = textColor,
                                        style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold),
                                        textAlign = TextAlign.Center
                                    )
                                }
                            }
                        }
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = if (isTamil) "Source Face" else "Source Face",
                            color = subColor,
                            style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold)
                        )
                    }

                    // Right: Target Photo
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        modifier = Modifier.weight(1f)
                    ) {
                        Box(
                            contentAlignment = Alignment.Center,
                            modifier = Modifier
                                .fillMaxWidth()
                                .aspectRatio(1f)
                                .clip(RoundedCornerShape(16.dp))
                                .background(Color.White.copy(alpha = if (darkTheme) 0.10f else 0.50f))
                                .border(1.2.dp, Color.White.copy(alpha = 0.85f), RoundedCornerShape(16.dp))
                                .clickable(enabled = !isSwapping, onClick = onPickTargetPhoto)
                                .testTag("faceswap_pick_target_button")
                        ) {
                            if (activeTargetModel != null) {
                                AsyncImage(
                                    model = activeTargetModel,
                                    contentDescription = "Target Photo",
                                    contentScale = ContentScale.Crop,
                                    modifier = Modifier.fillMaxSize()
                                )
                            } else {
                                Column(
                                    horizontalAlignment = Alignment.CenterHorizontally,
                                    modifier = Modifier.padding(8.dp)
                                ) {
                                    Icon(Icons.Filled.AddPhotoAlternate, contentDescription = null, tint = VibrantPinkPill, modifier = Modifier.size(30.dp))
                                    Spacer(modifier = Modifier.height(4.dp))
                                    Text(
                                        text = if (isTamil) "Target படம் தேர்வு" else "Pick Target Photo",
                                        color = textColor,
                                        style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold),
                                        textAlign = TextAlign.Center
                                    )
                                }
                            }
                        }
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = if (isTamil) "Target Photo" else "Target Photo",
                            color = subColor,
                            style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold)
                        )
                    }
                }

                if (targetFallbackCreation != null && File(targetFallbackCreation.filePath).exists()) {
                    Spacer(modifier = Modifier.height(10.dp))
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.Center,
                        modifier = Modifier
                            .fillMaxWidth()
                            .minimumInteractiveComponentSize()
                            .clip(RoundedCornerShape(14.dp))
                            .background(Brush.horizontalGradient(listOf(PurpleActionStart, PurpleActionEnd)))
                            .border(1.dp, Color.White.copy(alpha = 0.8f), RoundedCornerShape(14.dp))
                            .clickable(enabled = !isSwapping, onClick = onUseLatestCreationAsTarget)
                            .padding(vertical = 9.dp, horizontal = 12.dp)
                            .testTag("faceswap_use_latest_creation_chip")
                    ) {
                        Icon(Icons.Filled.AutoAwesome, contentDescription = null, tint = Color.White, modifier = Modifier.size(18.dp))
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = if (isTamil) "கடைசியாக உருவாக்கிய AI படத்தை Target ஆக்கு" else "Use Latest AI Creation as Target",
                            color = Color.White,
                            style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                        )
                    }
                }
            }

            // 2. Face Swap Options & Execute Card
            FrostedGlassCard(darkTheme = darkTheme) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = if (isTamil) "நிறப் பொருத்தம் (Skin Tone Match)" else "Auto Skin Tone Match",
                            color = textColor,
                            style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                        )
                        Text(
                            text = if (isTamil) "Target படத்தின் வெளிச்சத்திற்கு ஏற்ப முகம் பொருந்தும்" else "Matches target lighting & skin tone naturally",
                            color = subColor,
                            style = MaterialTheme.typography.labelSmall
                        )
                    }
                    Switch(
                        checked = enableSkinToneMatch,
                        onCheckedChange = { enableSkinToneMatch = it },
                        enabled = !isSwapping,
                        modifier = Modifier.testTag("faceswap_skin_match_switch")
                    )
                }

                Spacer(modifier = Modifier.height(8.dp))

                Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
                    Text(
                        text = if (isTamil) "இணைப்பு அளவு (Blend)" else "Face Blend Strength",
                        color = textColor,
                        style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold),
                        modifier = Modifier.weight(1f)
                    )
                    Text(
                        text = "${(blendStrength * 100).roundToInt()}%",
                        color = VibrantPinkPill,
                        style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.ExtraBold)
                    )
                }
                Slider(
                    value = blendStrength,
                    onValueChange = { blendStrength = it },
                    enabled = !isSwapping,
                    valueRange = 0.55f..1.0f,
                    modifier = Modifier.testTag("faceswap_blend_slider")
                )

                Text(
                    text = "Face Swap Models: ${metrics.faceSwapInstalledCount}/3 (det_10g • w600k_r50 • inswapper/hyperswap)",
                    color = if (metrics.isFaceSwapReady) NeonReadyGreen else subColor,
                    style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.SemiBold)
                )

                Spacer(modifier = Modifier.height(10.dp))

                val swapReady = metrics.isFaceSwapReady
                val btnShape = RoundedCornerShape(22.dp)

                if (isSwapping) {
                    val pct = (progress?.percent ?: 10).coerceIn(1, 100)
                    Text(
                        text = "${progress?.stageTitle ?: "Swapping Face Offline..."} ($pct%)",
                        color = textColor,
                        style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.SemiBold)
                    )
                    Spacer(modifier = Modifier.height(6.dp))
                    LinearProgressIndicator(
                        progress = { pct / 100f },
                        color = VibrantPinkPill,
                        modifier = Modifier.fillMaxWidth().height(8.dp).clip(RoundedCornerShape(50))
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.Center,
                        modifier = Modifier
                            .fillMaxWidth()
                            .minimumInteractiveComponentSize()
                            .clip(btnShape)
                            .background(Color(0xFFD32F2F))
                            .clickable(onClick = onCancelSwap)
                            .padding(vertical = 12.dp)
                            .testTag("faceswap_cancel_button")
                    ) {
                        Icon(Icons.Filled.Cancel, contentDescription = null, tint = Color.White)
                        Spacer(modifier = Modifier.width(6.dp))
                        Text("Cancel", color = Color.White, style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold))
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
                                    if (swapReady) listOf(PinkGlowStart, PinkGlowEnd)
                                    else listOf(PurpleActionStart, PurpleActionEnd)
                                )
                            )
                            .border(1.3.dp, Color.White, btnShape)
                            .clickable {
                                if (!swapReady) {
                                    onOpenModelsTab()
                                } else {
                                    onRunFaceSwap(enableSkinToneMatch, blendStrength)
                                }
                            }
                            .padding(vertical = 13.dp, horizontal = 14.dp)
                            .testTag("faceswap_execute_button")
                    ) {
                        Icon(
                            imageVector = if (swapReady) Icons.Filled.FaceRetouchingNatural else Icons.Filled.FolderSpecial,
                            contentDescription = null,
                            tint = Color.White
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = if (swapReady) {
                                if (isTamil) "Swap Face Offline (முகம் மாற்று)" else "Swap Face Offline"
                            } else {
                                if (isTamil) "Face Swap ONNX மாடல்களை ஏற்றவும் (${metrics.faceSwapInstalledCount}/3)"
                                else "Upload Face Swap Models (${metrics.faceSwapInstalledCount}/3)"
                            },
                            color = Color.White,
                            style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                        )
                    }
                }

                if (!errorMessage.isNullOrBlank()) {
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(errorMessage, color = Color(0xFFD32F2F), style = MaterialTheme.typography.bodyMedium)
                }
            }

            // 3. Swapped Result + Before/After Compare
            if (latestFaceSwapCreation != null && File(latestFaceSwapCreation.filePath).exists()) {
                val hasTargetBefore = !latestFaceSwapTargetPreviewPath.isNullOrBlank() &&
                    File(latestFaceSwapTargetPreviewPath).exists()

                FrostedGlassCard(darkTheme = darkTheme) {
                    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
                        Text(
                            text = if (isTamil) "Face Swap முடிவு" else "Face Swap Result",
                            color = textColor,
                            style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                            modifier = Modifier.weight(1f)
                        )
                        Text(
                            text = "${latestFaceSwapCreation.durationMs / 1000.0}s",
                            color = NeonReadyGreen,
                            style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                        )
                    }

                    if (hasTargetBefore) {
                        Spacer(modifier = Modifier.height(8.dp))
                        Row(horizontalArrangement = Arrangement.spacedBy(6.dp), modifier = Modifier.fillMaxWidth()) {
                            val modes = listOf(
                                FaceSwapCompareMode.SWAPPED to "After",
                                FaceSwapCompareMode.ORIGINAL_TARGET to "Before",
                                FaceSwapCompareMode.SIDE_BY_SIDE to "Compare"
                            )
                            modes.forEach { (mode, label) ->
                                val active = compareMode == mode
                                Box(
                                    contentAlignment = Alignment.Center,
                                    modifier = Modifier
                                        .weight(1f)
                                        .minimumInteractiveComponentSize()
                                        .clip(RoundedCornerShape(12.dp))
                                        .background(
                                            if (active) Brush.horizontalGradient(listOf(PinkGlowStart, PinkGlowEnd))
                                            else Brush.horizontalGradient(listOf(Color.White.copy(alpha = 0.45f), Color.White.copy(alpha = 0.35f)))
                                        )
                                        .border(1.dp, Color.White.copy(alpha = 0.75f), RoundedCornerShape(12.dp))
                                        .clickable { compareMode = mode }
                                        .padding(vertical = 7.dp)
                                ) {
                                    Text(label, color = if (active) Color.White else textColor, style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold))
                                }
                            }
                        }
                    }

                    Spacer(modifier = Modifier.height(10.dp))

                    when {
                        hasTargetBefore && compareMode == FaceSwapCompareMode.ORIGINAL_TARGET -> {
                            AsyncImage(
                                model = File(latestFaceSwapTargetPreviewPath!!),
                                contentDescription = "Original Target",
                                contentScale = ContentScale.Crop,
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .aspectRatio(1f)
                                    .clip(RoundedCornerShape(18.dp))
                                    .border(1.4.dp, Color.White, RoundedCornerShape(18.dp))
                            )
                        }
                        hasTargetBefore && compareMode == FaceSwapCompareMode.SIDE_BY_SIDE -> {
                            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                                AsyncImage(
                                    model = File(latestFaceSwapTargetPreviewPath!!),
                                    contentDescription = "Before",
                                    contentScale = ContentScale.Crop,
                                    modifier = Modifier
                                        .weight(1f)
                                        .aspectRatio(1f)
                                        .clip(RoundedCornerShape(14.dp))
                                        .border(1.2.dp, Color.White, RoundedCornerShape(14.dp))
                                )
                                AsyncImage(
                                    model = File(latestFaceSwapCreation.filePath),
                                    contentDescription = "After",
                                    contentScale = ContentScale.Crop,
                                    modifier = Modifier
                                        .weight(1f)
                                        .aspectRatio(1f)
                                        .clip(RoundedCornerShape(14.dp))
                                        .border(1.2.dp, Color.White, RoundedCornerShape(14.dp))
                                )
                            }
                        }
                        else -> {
                            AsyncImage(
                                model = File(latestFaceSwapCreation.filePath),
                                contentDescription = "Swapped Result",
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

                    Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.fillMaxWidth()) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.Center,
                            modifier = Modifier
                                .weight(1f)
                                .minimumInteractiveComponentSize()
                                .clip(RoundedCornerShape(16.dp))
                                .background(Brush.horizontalGradient(listOf(PurpleActionStart, PurpleActionEnd)))
                                .border(1.dp, Color.White, RoundedCornerShape(16.dp))
                                .clickable { onExportToGallery(latestFaceSwapCreation) }
                                .padding(vertical = 10.dp)
                                .testTag("faceswap_save_gallery_button")
                        ) {
                            Icon(Icons.Filled.Download, contentDescription = null, tint = Color.White, modifier = Modifier.size(18.dp))
                            Spacer(modifier = Modifier.width(6.dp))
                            Text(
                                text = if (isTamil) "Gallery-ல் சேமி" else "Save to Gallery",
                                color = Color.White,
                                style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.Bold)
                            )
                        }
                        IconButton(
                            onClick = { onToggleFavorite(latestFaceSwapCreation.id) },
                            modifier = Modifier.testTag("faceswap_favorite_button")
                        ) {
                            Icon(
                                imageVector = if (latestFaceSwapCreation.isFavorite) Icons.Filled.Star else Icons.Outlined.StarOutline,
                                contentDescription = "Favorite",
                                tint = VibrantPinkPill
                            )
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))
        }
    }
}
