package com.example.ui.screens

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoFixHigh
import androidx.compose.material.icons.filled.Autorenew
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Collections
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.material.icons.filled.Image
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.SwapHoriz
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.R
import com.example.ai.RealDeviceAndModelMetrics
import com.example.data.AppPreferencesState
import com.example.ui.ActiveScreen
import com.example.ui.components.CircularGlowIconButton
import com.example.ui.components.FeatureGlowCard
import com.example.ui.components.PaletteAppLogo
import com.example.ui.theme.AmberGlowEnd
import com.example.ui.theme.AmberGlowStart
import com.example.ui.theme.BlueGlowEnd
import com.example.ui.theme.BlueGlowStart
import com.example.ui.theme.DancingScriptFontFamily
import com.example.ui.theme.GreenGlowEnd
import com.example.ui.theme.GreenGlowStart
import com.example.ui.theme.MagentaGlowEnd
import com.example.ui.theme.MagentaGlowStart
import com.example.ui.theme.NeonReadyGreen
import com.example.ui.theme.PinkGlowEnd
import com.example.ui.theme.PinkGlowStart
import com.example.ui.theme.PurpleActionEnd
import com.example.ui.theme.RoyalIndigoTitle
import com.example.ui.theme.SoftAmberStatus
import com.example.ui.theme.TealGlowEnd
import com.example.ui.theme.TealGlowStart

@Composable
fun HomeScreen(
    preferences: AppPreferencesState,
    metrics: RealDeviceAndModelMetrics,
    onOpenSettings: () -> Unit,
    onSelectFeature: (ActiveScreen) -> Unit,
    onOpenModelsTab: () -> Unit,
    modifier: Modifier = Modifier
) {
    val isTamil = preferences.isTamil
    val darkTheme = preferences.darkTheme
    val titleColor = if (darkTheme) Color.White else RoyalIndigoTitle
    val subtitleColor = if (darkTheme) Color(0xFFEBD4FF) else Color(0xFF331D5C)

    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = modifier
            .fillMaxSize()
            .windowInsetsPadding(WindowInsets.statusBars)
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 16.dp)
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier
                .fillMaxWidth()
                .widthIn(max = 600.dp)
        ) {
            Spacer(modifier = Modifier.height(8.dp))

            // Top Header: App Logo + Title + Settings Icon
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                PaletteAppLogo(size = 54.dp)

                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    modifier = Modifier
                        .weight(1f)
                        .padding(horizontal = 8.dp)
                ) {
                    Text(
                        text = "Image Generator",
                        color = titleColor,
                        style = MaterialTheme.typography.headlineMedium.copy(
                            fontWeight = FontWeight.ExtraBold,
                            fontSize = 24.sp
                        ),
                        textAlign = TextAlign.Center
                    )
                    Text(
                        text = if (isTamil) {
                            "Offline AI • உருவாக்கு • திருத்து • மகிழ்"
                        } else {
                            "Offline AI • Create • Edit • Enjoy"
                        },
                        color = subtitleColor,
                        style = MaterialTheme.typography.bodyMedium.copy(
                            fontWeight = FontWeight.SemiBold,
                            fontSize = 12.5.sp
                        ),
                        textAlign = TextAlign.Center
                    )
                }

                CircularGlowIconButton(
                    onClick = onOpenSettings,
                    contentDescription = "Open Settings",
                    testTag = "open_settings_button"
                ) {
                    Icon(
                        imageVector = Icons.Filled.Settings,
                        contentDescription = "Settings",
                        tint = Color.White,
                        modifier = Modifier.size(24.dp)
                    )
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            // Hero Welcome Banner
            HeroWelcomeSection(
                darkTheme = darkTheme,
                isTamil = isTamil
            )

            Spacer(modifier = Modifier.height(14.dp))

            // 2x3 Feature Cards Grid
            Column(
                verticalArrangement = Arrangement.spacedBy(12.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                // Row 1: Text to Image | Image to Image
                Row(
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    FeatureGlowCard(
                        title = if (isTamil) "Text to\nImage" else "Text to\nImage",
                        gradientStart = PinkGlowStart,
                        gradientEnd = PinkGlowEnd,
                        testTag = "card_text_to_image",
                        onClick = { onSelectFeature(ActiveScreen.TEXT_TO_IMAGE) },
                        modifier = Modifier.weight(1f)
                    ) {
                        FramedIconBadge(
                            bgStart = Color(0xFF4FC3F7),
                            bgEnd = Color(0xFF1976D2),
                            mainIcon = Icons.Filled.Image
                        )
                    }

                    FeatureGlowCard(
                        title = if (isTamil) "Image to\nImage" else "Image to\nImage",
                        gradientStart = BlueGlowStart,
                        gradientEnd = BlueGlowEnd,
                        testTag = "card_image_to_image",
                        onClick = { onSelectFeature(ActiveScreen.IMAGE_TO_IMAGE) },
                        modifier = Modifier.weight(1f)
                    ) {
                        Box(contentAlignment = Alignment.BottomEnd) {
                            FramedIconBadge(
                                bgStart = Color(0xFF29B6F6),
                                bgEnd = Color(0xFF1565C0),
                                mainIcon = Icons.Filled.Image
                            )
                            Box(
                                contentAlignment = Alignment.Center,
                                modifier = Modifier
                                    .size(22.dp)
                                    .clip(CircleShape)
                                    .background(Color(0xFF0D47A1))
                                    .border(1.dp, Color.White, CircleShape)
                            ) {
                                Icon(
                                    imageVector = Icons.Filled.Autorenew,
                                    contentDescription = null,
                                    tint = Color.White,
                                    modifier = Modifier.size(13.dp)
                                )
                            }
                        }
                    }
                }

                // Row 2: Character Studio | Face Swap
                Row(
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    FeatureGlowCard(
                        title = if (isTamil) "Character\nStudio" else "Character\nStudio",
                        gradientStart = TealGlowStart,
                        gradientEnd = TealGlowEnd,
                        testTag = "card_character_studio",
                        onClick = { onSelectFeature(ActiveScreen.CHARACTER_STUDIO) },
                        modifier = Modifier.weight(1f)
                    ) {
                        Image(
                            painter = painterResource(id = R.drawable.img_character_avatar),
                            contentDescription = "Character Studio",
                            contentScale = ContentScale.Crop,
                            modifier = Modifier
                                .size(50.dp)
                                .clip(RoundedCornerShape(14.dp))
                                .border(1.2.dp, Color.White.copy(alpha = 0.85f), RoundedCornerShape(14.dp))
                        )
                    }

                    FeatureGlowCard(
                        title = if (isTamil) "Face\nSwap" else "Face\nSwap",
                        gradientStart = MagentaGlowStart,
                        gradientEnd = MagentaGlowEnd,
                        testTag = "card_face_swap",
                        onClick = { onSelectFeature(ActiveScreen.FACE_SWAP) },
                        modifier = Modifier.weight(1f)
                    ) {
                        Box(contentAlignment = Alignment.Center, modifier = Modifier.size(52.dp)) {
                            Image(
                                painter = painterResource(id = R.drawable.img_character_avatar),
                                contentDescription = null,
                                contentScale = ContentScale.Crop,
                                modifier = Modifier
                                    .size(34.dp)
                                    .offset(x = (-8).dp)
                                    .clip(CircleShape)
                                    .border(1.dp, Color.White, CircleShape)
                            )
                            Image(
                                painter = painterResource(id = R.drawable.img_hero_portrait),
                                contentDescription = null,
                                contentScale = ContentScale.Crop,
                                modifier = Modifier
                                    .size(34.dp)
                                    .offset(x = 9.dp)
                                    .clip(CircleShape)
                                    .border(1.dp, Color.White, CircleShape)
                            )
                            Box(
                                contentAlignment = Alignment.Center,
                                modifier = Modifier
                                    .size(20.dp)
                                    .clip(CircleShape)
                                    .background(PurpleActionEnd)
                                    .border(1.dp, Color.White, CircleShape)
                            ) {
                                Icon(
                                    imageVector = Icons.Filled.SwapHoriz,
                                    contentDescription = null,
                                    tint = Color.White,
                                    modifier = Modifier.size(13.dp)
                                )
                            }
                        }
                    }
                }

                // Row 3: Image Editor | My Gallery
                Row(
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    FeatureGlowCard(
                        title = if (isTamil) "Image\nEditor" else "Image\nEditor",
                        gradientStart = AmberGlowStart,
                        gradientEnd = AmberGlowEnd,
                        testTag = "card_image_editor",
                        onClick = { onSelectFeature(ActiveScreen.IMAGE_EDITOR) },
                        modifier = Modifier.weight(1f)
                    ) {
                        FramedIconBadge(
                            bgStart = Color(0xFFFFB300),
                            bgEnd = Color(0xFFEF6C00),
                            mainIcon = Icons.Filled.AutoFixHigh
                        )
                    }

                    FeatureGlowCard(
                        title = if (isTamil) "My\nGallery" else "My\nGallery",
                        gradientStart = GreenGlowStart,
                        gradientEnd = GreenGlowEnd,
                        testTag = "card_my_gallery",
                        onClick = { onSelectFeature(ActiveScreen.MY_GALLERY) },
                        modifier = Modifier.weight(1f)
                    ) {
                        FramedIconBadge(
                            bgStart = Color(0xFF66BB6A),
                            bgEnd = Color(0xFF2E7D32),
                            mainIcon = Icons.Filled.Collections
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(14.dp))

            // ONNX Runtime & Offline Mode Status Pill
            OnnxStatusPill(
                metrics = metrics,
                offlineMode = preferences.offlineMode,
                isTamil = isTamil,
                onClick = onOpenModelsTab
            )

            Spacer(modifier = Modifier.height(16.dp))
        }
    }
}

@Composable
private fun HeroWelcomeSection(
    darkTheme: Boolean,
    isTamil: Boolean
) {
    val shape = RoundedCornerShape(26.dp)
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(215.dp)
            .shadow(12.dp, shape, spotColor = Color(0x55C2185B))
            .clip(shape)
            .border(
                width = 1.5.dp,
                brush = Brush.verticalGradient(
                    listOf(Color.White.copy(alpha = 0.9f), Color.White.copy(alpha = 0.35f))
                ),
                shape = shape
            )
    ) {
        Image(
            painter = painterResource(id = R.drawable.img_hero_portrait),
            contentDescription = "Creative Studio Portrait Banner",
            contentScale = ContentScale.Crop,
            modifier = Modifier.fillMaxSize()
        )
        // Soft gradient scrim on the right so the script tagline reads crisply
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(
                    Brush.horizontalGradient(
                        colors = listOf(
                            Color.Transparent,
                            if (darkTheme) Color(0xAA230D38) else Color(0x99FFE6F2)
                        ),
                        startX = 280f
                    )
                )
        )
        Column(
            horizontalAlignment = Alignment.End,
            verticalArrangement = Arrangement.Center,
            modifier = Modifier
                .align(Alignment.CenterEnd)
                .padding(end = 16.dp, start = 140.dp)
        ) {
            Text(
                text = if (isTamil) {
                    "உங்கள் கற்பனையை\nநிஜமாக்குங்கள்"
                } else {
                    "Turn Your\nImagination\ninto Reality"
                },
                fontFamily = if (isTamil) MaterialTheme.typography.headlineMedium.fontFamily else DancingScriptFontFamily,
                fontWeight = FontWeight.Bold,
                fontSize = if (isTamil) 18.sp else 26.sp,
                lineHeight = if (isTamil) 24.sp else 30.sp,
                color = if (darkTheme) Color.White else Color(0xFF391563),
                textAlign = TextAlign.Center
            )
            Spacer(modifier = Modifier.height(6.dp))
            Icon(
                imageVector = Icons.Filled.Favorite,
                contentDescription = null,
                tint = Color(0xFF9C27B0),
                modifier = Modifier.size(22.dp)
            )
        }
    }
}

@Composable
private fun FramedIconBadge(
    bgStart: Color,
    bgEnd: Color,
    mainIcon: androidx.compose.ui.graphics.vector.ImageVector
) {
    val shape = RoundedCornerShape(14.dp)
    Box(
        contentAlignment = Alignment.Center,
        modifier = Modifier
            .size(48.dp)
            .clip(shape)
            .background(Brush.linearGradient(listOf(bgStart, bgEnd)))
            .border(1.4.dp, Color.White.copy(alpha = 0.9f), shape)
    ) {
        Icon(
            imageVector = mainIcon,
            contentDescription = null,
            tint = Color.White,
            modifier = Modifier.size(27.dp)
        )
    }
}

@Composable
private fun OnnxStatusPill(
    metrics: RealDeviceAndModelMetrics,
    offlineMode: Boolean,
    isTamil: Boolean,
    onClick: () -> Unit
) {
    val pillShape = RoundedCornerShape(50)
    val onnxReady = metrics.ortRuntimeAvailable && metrics.isTxt2ImgReady
    val onnxDotColor = if (onnxReady) {
        NeonReadyGreen
    } else if (metrics.ortRuntimeAvailable) {
        SoftAmberStatus
    } else {
        Color(0xFFFF5252)
    }

    val onnxStatusText = when {
        onnxReady -> "ONNX Runtime : Ready (${metrics.installedCount}/4)"
        metrics.ortRuntimeAvailable -> if (isTamil) {
            "ONNX v${metrics.ortVersion} : மாடல் சேர்க்கவும் (${metrics.installedCount}/4)"
        } else {
            "ONNX v${metrics.ortVersion} : Models (${metrics.installedCount}/4)"
        }
        else -> "ONNX Runtime : Unavailable"
    }

    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.Center,
        modifier = Modifier
            .clip(pillShape)
            .background(Color(0xAD3D224C))
            .border(1.dp, Color.White.copy(alpha = 0.55f), pillShape)
            .clickable(onClick = onClick)
            .padding(horizontal = 14.dp, vertical = 8.dp)
            .testTag("onnx_status_pill")
    ) {
        Box(
            modifier = Modifier
                .size(11.dp)
                .clip(CircleShape)
                .background(onnxDotColor)
        )
        Spacer(modifier = Modifier.width(6.dp))
        Text(
            text = onnxStatusText,
            color = Color.White,
            style = MaterialTheme.typography.labelLarge.copy(fontSize = 12.sp)
        )
        Spacer(modifier = Modifier.width(8.dp))
        Text(
            text = "|",
            color = Color.White.copy(alpha = 0.6f),
            style = MaterialTheme.typography.labelLarge
        )
        Spacer(modifier = Modifier.width(8.dp))
        Icon(
            imageVector = Icons.Filled.CheckCircle,
            contentDescription = null,
            tint = if (offlineMode) NeonReadyGreen else SoftAmberStatus,
            modifier = Modifier.size(15.dp)
        )
        Spacer(modifier = Modifier.width(4.dp))
        Text(
            text = if (offlineMode) {
                if (isTamil) "Offline Mode" else "Offline Mode"
            } else {
                "Online / Hybrid"
            },
            color = Color.White,
            style = MaterialTheme.typography.labelLarge.copy(fontSize = 12.sp)
        )
    }
}
