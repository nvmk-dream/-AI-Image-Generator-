package com.example.ui.components

import android.graphics.Bitmap
import android.net.Uri
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Undo
import androidx.compose.material.icons.filled.AddPhotoAlternate
import androidx.compose.material.icons.filled.AutoFixHigh
import androidx.compose.material.icons.filled.Brush
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Compare
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Layers
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.RemoveCircleOutline
import androidx.compose.material.icons.filled.Star
import androidx.compose.material.icons.filled.Visibility
import androidx.compose.material.icons.outlined.StarOutline
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.minimumInteractiveComponentSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.example.ai.AvatarVisionAnalysis
import com.example.ai.MaskBrushStroke
import com.example.ai.RealisticAvatarPreset
import com.example.data.GeneratedCreation
import com.example.ui.theme.DeepPlumText
import com.example.ui.theme.NeonReadyGreen
import com.example.ui.theme.PinkGlowEnd
import com.example.ui.theme.PinkGlowStart
import com.example.ui.theme.PurpleActionEnd
import com.example.ui.theme.PurpleActionStart
import com.example.ui.theme.SoftAmberStatus
import com.example.ui.theme.VibrantPinkPill
import java.io.File
import kotlin.math.min
import kotlin.math.roundToInt

enum class MaskTouchTool {
    VIEW,
    BRUSH_ADD,
    BRUSH_ERASE
}

@Composable
fun AvatarLabeledSlider(
    title: String,
    valueBadge: String,
    value: Float,
    valueRange: ClosedFloatingPointRange<Float>,
    steps: Int = 0,
    enabled: Boolean = true,
    darkTheme: Boolean,
    testTag: String,
    onValueChange: (Float) -> Unit,
    modifier: Modifier = Modifier
) {
    val textColor = if (darkTheme) Color.White else DeepPlumText
    Column(modifier = modifier.fillMaxWidth()) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(
                text = title,
                color = textColor,
                style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.SemiBold),
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.weight(1f)
            )
            Spacer(modifier = Modifier.width(8.dp))
            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(50))
                    .background(
                        Brush.horizontalGradient(
                            listOf(
                                PurpleActionStart.copy(alpha = if (darkTheme) 0.55f else 0.16f),
                                VibrantPinkPill.copy(alpha = if (darkTheme) 0.55f else 0.18f)
                            )
                        )
                    )
                    .border(
                        width = 1.dp,
                        color = VibrantPinkPill.copy(alpha = 0.65f),
                        shape = RoundedCornerShape(50)
                    )
                    .padding(horizontal = 9.dp, vertical = 2.dp)
            ) {
                Text(
                    text = valueBadge,
                    color = if (darkTheme) Color.White else DeepPlumText,
                    style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.ExtraBold)
                )
            }
        }
        Slider(
            value = value,
            onValueChange = onValueChange,
            valueRange = valueRange,
            steps = steps,
            enabled = enabled,
            colors = SliderDefaults.colors(
                thumbColor = VibrantPinkPill,
                activeTrackColor = PurpleActionStart,
                inactiveTrackColor = Color.White.copy(alpha = if (darkTheme) 0.22f else 0.65f)
            ),
            modifier = Modifier
                .fillMaxWidth()
                .testTag(testTag)
        )
    }
}

@Composable
fun RealisticPhotoPresetRow(
    presets: List<RealisticAvatarPreset>,
    selectedPresetId: String,
    isTamil: Boolean,
    darkTheme: Boolean,
    enabled: Boolean,
    onSelectPreset: (RealisticAvatarPreset) -> Unit,
    modifier: Modifier = Modifier
) {
    val textColor = if (darkTheme) Color.White else DeepPlumText
    Column(
        verticalArrangement = Arrangement.spacedBy(10.dp),
        modifier = modifier
            .fillMaxWidth()
            .padding(vertical = 2.dp)
            .testTag("avatar_preset_grid")
    ) {
        presets.chunked(2).forEach { rowItems ->
            Row(
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                rowItems.forEach { preset ->
                    val isSelected = preset.id == selectedPresetId
                    val cardShape = RoundedCornerShape(18.dp)
                    val title = if (isTamil) preset.titleTa else preset.titleEn
                    val badge = if (isTamil) preset.badgeTa else preset.badgeEn

                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        modifier = Modifier
                            .weight(1f)
                            .shadow(
                                elevation = if (isSelected) 10.dp else 4.dp,
                                shape = cardShape,
                                spotColor = if (isSelected) VibrantPinkPill else PurpleActionEnd
                            )
                            .clip(cardShape)
                            .background(
                                if (isSelected) {
                                    Brush.verticalGradient(listOf(PurpleActionStart, PinkGlowEnd))
                                } else {
                                    Brush.verticalGradient(
                                        listOf(
                                            Color.White.copy(alpha = if (darkTheme) 0.14f else 0.78f),
                                            Color.White.copy(alpha = if (darkTheme) 0.08f else 0.60f)
                                        )
                                    )
                                }
                            )
                            .border(
                                width = if (isSelected) 2.2.dp else 1.dp,
                                color = if (isSelected) Color.White else Color.White.copy(alpha = 0.65f),
                                shape = cardShape
                            )
                            .clickable(enabled = enabled) { onSelectPreset(preset) }
                            .padding(6.dp)
                            .testTag("avatar_preset_${preset.id}")
                    ) {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(114.dp)
                                .clip(RoundedCornerShape(13.dp))
                        ) {
                            Image(
                                painter = painterResource(id = preset.photoResId),
                                contentDescription = title,
                                contentScale = ContentScale.Crop,
                                modifier = Modifier.fillMaxSize()
                            )
                            // Top-left DSLR Photo Badge
                            Box(
                                modifier = Modifier
                                    .align(Alignment.TopStart)
                                    .padding(5.dp)
                                    .clip(RoundedCornerShape(50))
                                    .background(Color(0xCC120724))
                                    .border(0.8.dp, Color.White.copy(alpha = 0.7f), RoundedCornerShape(50))
                                    .padding(horizontal = 6.dp, vertical = 2.dp)
                            ) {
                                Text(
                                    text = badge,
                                    color = Color.White,
                                    style = MaterialTheme.typography.labelSmall.copy(
                                        fontSize = 9.5.sp,
                                        fontWeight = FontWeight.Bold
                                    )
                                )
                            }
                            if (isSelected) {
                                Icon(
                                    imageVector = Icons.Filled.CheckCircle,
                                    contentDescription = "Selected Preset",
                                    tint = NeonReadyGreen,
                                    modifier = Modifier
                                        .align(Alignment.BottomEnd)
                                        .padding(5.dp)
                                        .size(20.dp)
                                        .background(Color(0xCC0D061A), CircleShape)
                                )
                            }
                        }
                        Spacer(modifier = Modifier.height(5.dp))
                        Text(
                            text = title,
                            color = if (isSelected) Color.White else textColor,
                            style = MaterialTheme.typography.labelMedium.copy(
                                fontWeight = FontWeight.ExtraBold,
                                fontSize = 11.5.sp
                            ),
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                            textAlign = TextAlign.Center
                        )
                    }
                }
                if (rowItems.size == 1) {
                    Spacer(modifier = Modifier.weight(1f))
                }
            }
        }
    }
}

@Composable
fun AvatarInteractivePreviewCard(
    sourcePhotoUri: Uri?,
    hasLatestAppCreation: Boolean,
    visionAnalysis: AvatarVisionAnalysis?,
    maskOverlayBitmap: Bitmap?,
    userStrokesCount: Int,
    latestAvatarCreation: GeneratedCreation?,
    latestAvatarSourcePath: String?,
    isGenerating: Boolean,
    isTamil: Boolean,
    darkTheme: Boolean,
    onPickPhoto: () -> Unit,
    onUseLatestAppCreation: () -> Unit,
    onAddBrushStroke: (MaskBrushStroke) -> Unit,
    onUndoBrushStroke: () -> Unit,
    onResetBrushStrokes: () -> Unit,
    onUseResultAsNextSource: () -> Unit,
    onToggleFavorite: (String) -> Unit,
    onExportToGallery: (GeneratedCreation) -> Unit,
    modifier: Modifier = Modifier
) {
    val textColor = if (darkTheme) Color.White else DeepPlumText
    val subColor = if (darkTheme) Color(0xFFE1C9F7) else Color(0xFF4F376E)

    var showMaskOverlay by remember { mutableStateOf(true) }
    var touchTool by remember { mutableStateOf(MaskTouchTool.VIEW) }
    var brushRadiusNorm by remember { mutableFloatStateOf(0.055f) }
    var showBeforeAfterResult by remember(latestAvatarCreation?.id) {
        mutableStateOf(latestAvatarCreation != null)
    }
    var splitFraction by remember(latestAvatarCreation?.id) { mutableFloatStateOf(0.52f) }

    val activeDragPoints = remember { mutableStateListOf<Pair<Float, Float>>() }
    var canvasPixelSize by remember { mutableStateOf(IntSize.Zero) }

    FrostedGlassCard(darkTheme = darkTheme, modifier = modifier) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(
                text = if (isTamil) {
                    "1. உண்மையான புகைப்படம் & AI Mask"
                } else {
                    "1. Realistic Photo & AI Mask Canvas"
                },
                color = textColor,
                style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.ExtraBold),
                modifier = Modifier.weight(1f)
            )
            if (latestAvatarCreation != null) {
                Surface(
                    shape = RoundedCornerShape(50),
                    color = if (showBeforeAfterResult) VibrantPinkPill else Color.White.copy(alpha = 0.2f),
                    border = BorderStroke(1.dp, Color.White.copy(alpha = 0.75f)),
                    modifier = Modifier
                        .minimumInteractiveComponentSize()
                        .clickable { showBeforeAfterResult = !showBeforeAfterResult }
                        .testTag("avatar_toggle_result_compare")
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Filled.Compare,
                            contentDescription = "Toggle Before After Compare",
                            tint = Color.White,
                            modifier = Modifier.size(16.dp)
                        )
                        Spacer(modifier = Modifier.width(4.dp))
                        Text(
                            text = if (showBeforeAfterResult) {
                                if (isTamil) "Mask எடிட்" else "Edit Mask"
                            } else {
                                if (isTamil) "Before/After" else "Before/After"
                            },
                            color = Color.White,
                            style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold)
                        )
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        // Photo Picker + Use Latest Creation Action Row
        Row(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            val btnShape = RoundedCornerShape(16.dp)
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.Center,
                modifier = Modifier
                    .weight(1f)
                    .minimumInteractiveComponentSize()
                    .clip(btnShape)
                    .background(Brush.horizontalGradient(listOf(PinkGlowStart, PurpleActionEnd)))
                    .border(1.2.dp, Color.White.copy(alpha = 0.85f), btnShape)
                    .clickable(enabled = !isGenerating, onClick = onPickPhoto)
                    .padding(vertical = 10.dp, horizontal = 10.dp)
                    .testTag("avatar_pick_photo_button")
            ) {
                Icon(
                    imageVector = Icons.Filled.AddPhotoAlternate,
                    contentDescription = "Select Realistic Photo from Gallery",
                    tint = Color.White,
                    modifier = Modifier.size(19.dp)
                )
                Spacer(modifier = Modifier.width(6.dp))
                Text(
                    text = if (isTamil) "கேலரி படம் தேர்வு" else "Pick Gallery Photo",
                    color = Color.White,
                    style = MaterialTheme.typography.labelLarge.copy(fontWeight = FontWeight.ExtraBold),
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
            }

            if (hasLatestAppCreation) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.Center,
                    modifier = Modifier
                        .minimumInteractiveComponentSize()
                        .clip(btnShape)
                        .background(Color.White.copy(alpha = if (darkTheme) 0.14f else 0.72f))
                        .border(1.dp, PurpleActionStart.copy(alpha = 0.75f), btnShape)
                        .clickable(enabled = !isGenerating, onClick = onUseLatestAppCreation)
                        .padding(vertical = 10.dp, horizontal = 10.dp)
                        .testTag("avatar_use_latest_button")
                ) {
                    Icon(
                        imageVector = Icons.Filled.History,
                        contentDescription = "Use Latest App Photo",
                        tint = if (darkTheme) Color.White else PurpleActionStart,
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(
                        text = if (isTamil) "சமீபத்திய படம்" else "Latest Photo",
                        color = textColor,
                        style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold)
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(10.dp))

        // Main Square 1:1 Interactive Canvas
        val frameShape = RoundedCornerShape(20.dp)
        if (showBeforeAfterResult && latestAvatarCreation != null && File(latestAvatarCreation.filePath).exists()) {
            // Draggable Before ↔ After Split View
            val density = LocalDensity.current
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .aspectRatio(1f)
                    .clip(frameShape)
                    .border(1.5.dp, Color.White.copy(alpha = 0.85f), frameShape)
                    .onSizeChanged { canvasPixelSize = it }
                    .pointerInput(latestAvatarCreation.id) {
                        detectDragGestures { change, _ ->
                            change.consume()
                            if (size.width > 0) {
                                splitFraction = (change.position.x / size.width.toFloat()).coerceIn(0.05f, 0.95f)
                            }
                        }
                    }
                    .testTag("avatar_before_after_canvas")
            ) {
                // Full "After" generated DSLR photograph underneath
                AsyncImage(
                    model = File(latestAvatarCreation.filePath),
                    contentDescription = "Customized Realistic Output Photo",
                    contentScale = ContentScale.Crop,
                    modifier = Modifier.fillMaxSize()
                )

                // Left-clipped "Before" original photograph
                val beforeModel: Any? = latestAvatarSourcePath
                    ?.let { File(it) }
                    ?.takeIf { it.exists() }
                    ?: sourcePhotoUri
                if (beforeModel != null) {
                    AsyncImage(
                        model = beforeModel,
                        contentDescription = "Original Source Photo",
                        contentScale = ContentScale.Crop,
                        modifier = Modifier
                            .fillMaxSize()
                            .drawWithContent {
                                clipRect(right = size.width * splitFraction) {
                                    this@drawWithContent.drawContent()
                                }
                            }
                    )
                }

                // Vertical split divider + circular thumb
                val dividerX = (canvasPixelSize.width * splitFraction).roundToInt()
                if (canvasPixelSize.width > 0) {
                    val xDp = with(density) { dividerX.toDp() }
                    Box(
                        modifier = Modifier
                            .offset(x = xDp - 1.dp)
                            .fillMaxHeight()
                            .width(2.5.dp)
                            .background(Color.White)
                    )
                    Box(
                        contentAlignment = Alignment.Center,
                        modifier = Modifier
                            .align(Alignment.CenterStart)
                            .offset { IntOffset(dividerX - 18.dp.roundToPx(), 0) }
                            .size(36.dp)
                            .shadow(6.dp, CircleShape)
                            .clip(CircleShape)
                            .background(Brush.linearGradient(listOf(PurpleActionStart, VibrantPinkPill)))
                            .border(1.5.dp, Color.White, CircleShape)
                    ) {
                        Icon(
                            imageVector = Icons.Filled.Compare,
                            contentDescription = null,
                            tint = Color.White,
                            modifier = Modifier.size(18.dp)
                        )
                    }
                }

                // Before / After corner labels
                Box(
                    modifier = Modifier
                        .align(Alignment.TopStart)
                        .padding(8.dp)
                        .clip(RoundedCornerShape(50))
                        .background(Color(0xB80E061C))
                        .padding(horizontal = 8.dp, vertical = 3.dp)
                ) {
                    Text(
                        text = if (isTamil) "அசல் (Before)" else "BEFORE",
                        color = Color.White,
                        style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold)
                    )
                }
                Box(
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .padding(8.dp)
                        .clip(RoundedCornerShape(50))
                        .background(Color(0xCC4A148C))
                        .padding(horizontal = 8.dp, vertical = 3.dp)
                ) {
                    Text(
                        text = if (isTamil) "மாற்றம் (After)" else "AFTER • DSLR",
                        color = NeonReadyGreen,
                        style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.ExtraBold)
                    )
                }
            }

            Spacer(modifier = Modifier.height(6.dp))
            Slider(
                value = splitFraction,
                onValueChange = { splitFraction = it },
                valueRange = 0.05f..0.95f,
                modifier = Modifier
                    .fillMaxWidth()
                    .testTag("avatar_before_after_slider")
            )

            // Result Action Row: Chain Next Edit + Favorite + Save to Gallery
            Row(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.Center,
                    modifier = Modifier
                        .weight(1f)
                        .minimumInteractiveComponentSize()
                        .clip(RoundedCornerShape(14.dp))
                        .background(Brush.horizontalGradient(listOf(PurpleActionStart, PurpleActionEnd)))
                        .border(1.dp, Color.White.copy(alpha = 0.8f), RoundedCornerShape(14.dp))
                        .clickable(enabled = !isGenerating) {
                            showBeforeAfterResult = false
                            onUseResultAsNextSource()
                        }
                        .padding(vertical = 9.dp, horizontal = 8.dp)
                        .testTag("avatar_chain_next_edit_button")
                ) {
                    Icon(
                        imageVector = Icons.Filled.AutoFixHigh,
                        contentDescription = null,
                        tint = Color.White,
                        modifier = Modifier.size(17.dp)
                    )
                    Spacer(modifier = Modifier.width(5.dp))
                    Text(
                        text = if (isTamil) "இதன் மேல் அடுத்த மாற்றம்" else "Chain Next Attribute",
                        color = Color.White,
                        style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                }

                IconButton(
                    onClick = { onToggleFavorite(latestAvatarCreation.id) },
                    modifier = Modifier
                        .minimumInteractiveComponentSize()
                        .clip(RoundedCornerShape(14.dp))
                        .background(Color.White.copy(alpha = if (darkTheme) 0.14f else 0.72f))
                        .testTag("avatar_favorite_button")
                ) {
                    Icon(
                        imageVector = if (latestAvatarCreation.isFavorite) Icons.Filled.Star else Icons.Outlined.StarOutline,
                        contentDescription = "Favorite Creation",
                        tint = if (latestAvatarCreation.isFavorite) Color(0xFFFFD54F) else textColor
                    )
                }

                IconButton(
                    onClick = { onExportToGallery(latestAvatarCreation) },
                    modifier = Modifier
                        .minimumInteractiveComponentSize()
                        .clip(RoundedCornerShape(14.dp))
                        .background(Brush.horizontalGradient(listOf(PinkGlowStart, VibrantPinkPill)))
                        .testTag("avatar_save_gallery_button")
                ) {
                    Icon(
                        imageVector = Icons.Filled.Download,
                        contentDescription = "Save Realistic Photo to Gallery",
                        tint = Color.White
                    )
                }
            }
        } else if (sourcePhotoUri != null) {
            // Source Photo + Live Glowing AI Mask Overlay + Finger Brush/Eraser Touch Canvas
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .aspectRatio(1f)
                    .clip(frameShape)
                    .border(1.5.dp, Color.White.copy(alpha = 0.85f), frameShape)
                    .onSizeChanged { canvasPixelSize = it }
                    .then(
                        if (touchTool != MaskTouchTool.VIEW && !isGenerating) {
                            Modifier.pointerInput(touchTool, brushRadiusNorm) {
                                detectDragGestures(
                                    onDragStart = { startOffset ->
                                        activeDragPoints.clear()
                                        if (size.width > 0 && size.height > 0) {
                                            activeDragPoints.add(
                                                (startOffset.x / size.width).coerceIn(0f, 1f) to
                                                    (startOffset.y / size.height).coerceIn(0f, 1f)
                                            )
                                        }
                                    },
                                    onDrag = { change, _ ->
                                        change.consume()
                                        if (size.width > 0 && size.height > 0) {
                                            activeDragPoints.add(
                                                (change.position.x / size.width).coerceIn(0f, 1f) to
                                                    (change.position.y / size.height).coerceIn(0f, 1f)
                                            )
                                        }
                                    },
                                    onDragEnd = {
                                        if (activeDragPoints.isNotEmpty()) {
                                            onAddBrushStroke(
                                                MaskBrushStroke(
                                                    normalizedPoints = activeDragPoints.toList(),
                                                    normalizedRadius = brushRadiusNorm,
                                                    isErase = touchTool == MaskTouchTool.BRUSH_ERASE
                                                )
                                            )
                                            activeDragPoints.clear()
                                        }
                                    },
                                    onDragCancel = {
                                        activeDragPoints.clear()
                                    }
                                )
                            }
                        } else {
                            Modifier
                        }
                    )
                    .testTag("avatar_mask_canvas")
            ) {
                AsyncImage(
                    model = sourcePhotoUri,
                    contentDescription = "Selected Realistic Source Photo",
                    contentScale = ContentScale.Crop,
                    modifier = Modifier.fillMaxSize()
                )

                if (showMaskOverlay && maskOverlayBitmap != null && !maskOverlayBitmap.isRecycled) {
                    Image(
                        bitmap = maskOverlayBitmap.asImageBitmap(),
                        contentDescription = "AI Detected Edit Mask Overlay",
                        contentScale = ContentScale.FillBounds,
                        modifier = Modifier.fillMaxSize()
                    )
                }

                // Live finger stroke feedback while dragging
                if (activeDragPoints.isNotEmpty()) {
                    Canvas(modifier = Modifier.fillMaxSize()) {
                        val rPx = brushRadiusNorm * min(size.width, size.height)
                        val strokeColor = if (touchTool == MaskTouchTool.BRUSH_ERASE) {
                            Color(0xAAFF5252)
                        } else {
                            Color(0xAA00E676)
                        }
                        if (activeDragPoints.size == 1) {
                            val p = activeDragPoints.first()
                            drawCircle(
                                color = strokeColor,
                                radius = rPx,
                                center = Offset(p.first * size.width, p.second * size.height)
                            )
                        } else {
                            val path = Path()
                            activeDragPoints.forEachIndexed { idx, pt ->
                                val x = pt.first * size.width
                                val y = pt.second * size.height
                                if (idx == 0) path.moveTo(x, y) else path.lineTo(x, y)
                            }
                            drawPath(
                                path = path,
                                color = strokeColor,
                                style = Stroke(
                                    width = rPx * 2f,
                                    cap = StrokeCap.Round,
                                    join = StrokeJoin.Round
                                )
                            )
                        }
                    }
                }

                // Detection Badges Overlay (Top-Start)
                if (visionAnalysis != null) {
                    val badgeLabel = buildString {
                        append(if (visionAnalysis.faceDetected) "Face ✓" else "Portrait")
                        if (visionAnalysis.poseDetected) append(" • Pose ✓")
                        if (visionAnalysis.usedMediaPipeSegmenter) append(" • Seg ✓")
                    }
                    Box(
                        modifier = Modifier
                            .align(Alignment.TopStart)
                            .padding(8.dp)
                            .clip(RoundedCornerShape(50))
                            .background(Color(0xC8100622))
                            .border(0.8.dp, NeonReadyGreen.copy(alpha = 0.75f), RoundedCornerShape(50))
                            .padding(horizontal = 9.dp, vertical = 4.dp)
                    ) {
                        Text(
                            text = badgeLabel,
                            color = NeonReadyGreen,
                            style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold)
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Interactive Mask Overlay & Finger Brush / Eraser Toolbar
            Row(
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState())
            ) {
                MaskToolChip(
                    label = if (showMaskOverlay) {
                        if (isTamil) "Mask காட்டு: ON" else "Mask: ON"
                    } else {
                        if (isTamil) "Mask காட்டு: OFF" else "Mask: OFF"
                    },
                    icon = Icons.Filled.Layers,
                    selected = showMaskOverlay,
                    darkTheme = darkTheme,
                    testTag = "avatar_toggle_mask_overlay",
                    onClick = { showMaskOverlay = !showMaskOverlay }
                )
                MaskToolChip(
                    label = if (isTamil) "பார்வை" else "View",
                    icon = Icons.Filled.Visibility,
                    selected = touchTool == MaskTouchTool.VIEW,
                    darkTheme = darkTheme,
                    testTag = "avatar_touch_mode_view",
                    onClick = { touchTool = MaskTouchTool.VIEW }
                )
                MaskToolChip(
                    label = if (isTamil) "Brush (+)" else "Brush (+)",
                    icon = Icons.Filled.Brush,
                    selected = touchTool == MaskTouchTool.BRUSH_ADD,
                    darkTheme = darkTheme,
                    testTag = "avatar_touch_mode_paint",
                    onClick = {
                        showMaskOverlay = true
                        touchTool = MaskTouchTool.BRUSH_ADD
                    }
                )
                MaskToolChip(
                    label = if (isTamil) "Eraser (-)" else "Eraser (-)",
                    icon = Icons.Filled.RemoveCircleOutline,
                    selected = touchTool == MaskTouchTool.BRUSH_ERASE,
                    darkTheme = darkTheme,
                    testTag = "avatar_touch_mode_erase",
                    onClick = {
                        showMaskOverlay = true
                        touchTool = MaskTouchTool.BRUSH_ERASE
                    }
                )
                if (userStrokesCount > 0) {
                    IconButton(
                        onClick = onUndoBrushStroke,
                        modifier = Modifier
                            .minimumInteractiveComponentSize()
                            .testTag("avatar_undo_stroke_button")
                    ) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.Undo,
                            contentDescription = "Undo Mask Stroke",
                            tint = textColor
                        )
                    }
                    IconButton(
                        onClick = onResetBrushStrokes,
                        modifier = Modifier
                            .minimumInteractiveComponentSize()
                            .testTag("avatar_reset_mask_button")
                    ) {
                        Icon(
                            imageVector = Icons.Filled.Refresh,
                            contentDescription = "Reset Auto AI Mask",
                            tint = SoftAmberStatus
                        )
                    }
                }
            }

            if (touchTool != MaskTouchTool.VIEW) {
                Spacer(modifier = Modifier.height(4.dp))
                AvatarLabeledSlider(
                    title = if (isTamil) "Brush அளவு (விரலால் Mask திருத்த)" else "Touch Brush Size (Draw on Photo)",
                    valueBadge = "${(brushRadiusNorm * 500).roundToInt()} px",
                    value = brushRadiusNorm,
                    valueRange = 0.02f..0.14f,
                    enabled = !isGenerating,
                    darkTheme = darkTheme,
                    testTag = "avatar_slider_brush_size",
                    onValueChange = { brushRadiusNorm = it }
                )
            }
        } else {
            // Empty Realistic Photo Prompt Placeholder
            Box(
                contentAlignment = Alignment.Center,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(190.dp)
                    .clip(frameShape)
                    .background(Color.White.copy(alpha = if (darkTheme) 0.08f else 0.55f))
                    .border(1.3.dp, VibrantPinkPill.copy(alpha = 0.55f), frameShape)
                    .clickable(enabled = !isGenerating, onClick = onPickPhoto)
                    .padding(16.dp)
                    .testTag("avatar_empty_photo_placeholder")
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Icon(
                        imageVector = Icons.Filled.AddPhotoAlternate,
                        contentDescription = null,
                        tint = VibrantPinkPill,
                        modifier = Modifier.size(42.dp)
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = if (isTamil) {
                            "உங்கள் மொபைல் கேலரியில் இருந்து ஒரு இயற்கையான புகைப்படத்தைத் (Realistic Photo) தேர்ந்தெடுக்கவும்"
                        } else {
                            "Tap to select a Natural Realistic Photograph from your phone gallery"
                        },
                        color = textColor,
                        style = MaterialTheme.typography.bodyMedium.copy(fontWeight = FontWeight.Bold),
                        textAlign = TextAlign.Center
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = if (isTamil) {
                            "100% ஆஃப்லைன் • அசல் முகம் & அடையாளம் பாதுகாக்கப்படும்"
                        } else {
                            "100% Offline • Face & Identity Locked Automatically"
                        },
                        color = subColor,
                        style = MaterialTheme.typography.labelSmall,
                        textAlign = TextAlign.Center
                    )
                }
            }
        }
    }
}

@Composable
private fun MaskToolChip(
    label: String,
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    selected: Boolean,
    darkTheme: Boolean,
    testTag: String,
    onClick: () -> Unit
) {
    val shape = RoundedCornerShape(50)
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .minimumInteractiveComponentSize()
            .clip(shape)
            .background(
                if (selected) {
                    Brush.horizontalGradient(listOf(PurpleActionStart, VibrantPinkPill))
                } else {
                    Brush.horizontalGradient(
                        listOf(
                            Color.White.copy(alpha = if (darkTheme) 0.14f else 0.72f),
                            Color.White.copy(alpha = if (darkTheme) 0.10f else 0.62f)
                        )
                    )
                }
            )
            .border(
                width = 1.dp,
                color = if (selected) Color.White else Color.White.copy(alpha = 0.6f),
                shape = shape
            )
            .clickable(onClick = onClick)
            .padding(horizontal = 11.dp, vertical = 7.dp)
            .testTag(testTag)
    ) {
        Icon(
            imageVector = icon,
            contentDescription = null,
            tint = if (selected) Color.White else if (darkTheme) Color.White else DeepPlumText,
            modifier = Modifier.size(15.dp)
        )
        Spacer(modifier = Modifier.width(5.dp))
        Text(
            text = label,
            color = if (selected) Color.White else if (darkTheme) Color.White else DeepPlumText,
            style = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold)
        )
    }
}
