package com.example.ui.components

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowRight
import androidx.compose.material.icons.filled.CloudDownload
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Star
import androidx.compose.material.icons.outlined.CloudDownload
import androidx.compose.material.icons.outlined.FavoriteBorder
import androidx.compose.material.icons.outlined.Home
import androidx.compose.material.icons.outlined.StarOutline
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.minimumInteractiveComponentSize
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
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.R
import com.example.ui.BottomNavTab
import com.example.ui.theme.DarkGlassBorder
import com.example.ui.theme.DarkGlassSurface
import com.example.ui.theme.DeepVelvetBackground
import com.example.ui.theme.LightGlassBorder
import com.example.ui.theme.LightGlassSurface
import com.example.ui.theme.PinkGlowEnd
import com.example.ui.theme.PinkGlowStart
import com.example.ui.theme.PurpleActionEnd
import com.example.ui.theme.PurpleActionStart

@Composable
fun DreamyHeartBackground(
    darkTheme: Boolean,
    modifier: Modifier = Modifier,
    content: @Composable BoxScope.() -> Unit
) {
    Box(modifier = modifier.fillMaxSize()) {
        Image(
            painter = painterResource(id = R.drawable.img_pink_hearts_bg),
            contentDescription = null,
            contentScale = ContentScale.Crop,
            modifier = Modifier.fillMaxSize()
        )
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(
                    if (darkTheme) {
                        Brush.verticalGradient(
                            colors = listOf(
                                DeepVelvetBackground.copy(alpha = 0.72f),
                                Color(0xB328113E),
                                DeepVelvetBackground.copy(alpha = 0.84f)
                            )
                        )
                    } else {
                        Brush.verticalGradient(
                            colors = listOf(
                                Color(0x33FFE8F4),
                                Color(0x1FFFD6EC),
                                Color(0x40F7C8E6)
                            )
                        )
                    }
                )
        )
        content()
    }
}

@Composable
fun FrostedGlassCard(
    darkTheme: Boolean,
    modifier: Modifier = Modifier,
    cornerRadius: Dp = 22.dp,
    contentPadding: Dp = 14.dp,
    content: @Composable ColumnScope.() -> Unit
) {
    val shape = RoundedCornerShape(cornerRadius)
    val bgBrush = if (darkTheme) {
        Brush.verticalGradient(
            listOf(
                DarkGlassSurface,
                Color(0xB31E0E30)
            )
        )
    } else {
        Brush.verticalGradient(
            listOf(
                Color(0x9EFFFFFF),
                LightGlassSurface
            )
        )
    }
    val borderBrush = Brush.linearGradient(
        if (darkTheme) {
            listOf(Color(0x99FF9CE3), DarkGlassBorder)
        } else {
            listOf(LightGlassBorder, Color(0x99FFD4EC))
        }
    )

    Column(
        modifier = modifier
            .shadow(
                elevation = 10.dp,
                shape = shape,
                ambientColor = Color(0x45C2409E),
                spotColor = Color(0x458E2DE2)
            )
            .clip(shape)
            .background(bgBrush)
            .border(width = 1.3.dp, brush = borderBrush, shape = shape)
            .padding(contentPadding),
        content = content
    )
}

@Composable
fun PaletteAppLogo(
    size: Dp = 52.dp,
    modifier: Modifier = Modifier
) {
    Surface(
        modifier = modifier
            .size(size)
            .shadow(8.dp, CircleShape, spotColor = PurpleActionEnd),
        shape = CircleShape,
        border = BorderStroke(1.8.dp, Color.White.copy(alpha = 0.9f)),
        color = Color(0xFF2A1045)
    ) {
        Image(
            painter = painterResource(id = R.drawable.img_app_icon_palette),
            contentDescription = "Image Generator Palette Logo",
            contentScale = ContentScale.Crop,
            modifier = Modifier.fillMaxSize()
        )
    }
}

@Composable
fun CircularGlowIconButton(
    onClick: () -> Unit,
    contentDescription: String,
    testTag: String,
    modifier: Modifier = Modifier,
    icon: @Composable () -> Unit
) {
    Box(
        contentAlignment = Alignment.Center,
        modifier = modifier
            .minimumInteractiveComponentSize()
            .size(48.dp)
            .shadow(8.dp, CircleShape, spotColor = PurpleActionEnd)
            .clip(CircleShape)
            .background(
                Brush.linearGradient(
                    colors = listOf(PurpleActionStart, PurpleActionEnd)
                )
            )
            .border(1.4.dp, Color.White.copy(alpha = 0.8f), CircleShape)
            .clickable(onClick = onClick)
            .testTag(testTag)
    ) {
        icon()
    }
}

@Composable
fun FeatureGlowCard(
    title: String,
    gradientStart: Color,
    gradientEnd: Color,
    testTag: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    leadingArtwork: @Composable () -> Unit
) {
    val shape = RoundedCornerShape(22.dp)
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = modifier
            .height(92.dp)
            .shadow(
                elevation = 10.dp,
                shape = shape,
                ambientColor = gradientEnd.copy(alpha = 0.55f),
                spotColor = gradientEnd.copy(alpha = 0.65f)
            )
            .clip(shape)
            .background(
                Brush.linearGradient(
                    colors = listOf(
                        gradientStart.copy(alpha = 0.92f),
                        gradientEnd.copy(alpha = 0.88f)
                    )
                )
            )
            .border(
                width = 1.5.dp,
                brush = Brush.verticalGradient(
                    listOf(Color.White.copy(alpha = 0.9f), Color.White.copy(alpha = 0.35f))
                ),
                shape = shape
            )
            .clickable(onClick = onClick)
            .padding(horizontal = 10.dp, vertical = 8.dp)
            .testTag(testTag)
    ) {
        Box(
            contentAlignment = Alignment.Center,
            modifier = Modifier.size(54.dp)
        ) {
            leadingArtwork()
        }
        Spacer(modifier = Modifier.width(8.dp))
        Text(
            text = title,
            color = Color.White,
            style = MaterialTheme.typography.titleMedium.copy(
                fontWeight = FontWeight.Bold,
                fontSize = 15.sp,
                lineHeight = 19.sp
            ),
            maxLines = 2,
            overflow = TextOverflow.Ellipsis,
            modifier = Modifier.weight(1f)
        )
        Spacer(modifier = Modifier.width(4.dp))
        Box(
            contentAlignment = Alignment.Center,
            modifier = Modifier
                .size(24.dp)
                .clip(CircleShape)
                .background(Color.Black.copy(alpha = 0.22f))
                .border(1.dp, Color.White.copy(alpha = 0.85f), CircleShape)
        ) {
            Icon(
                imageVector = Icons.AutoMirrored.Filled.KeyboardArrowRight,
                contentDescription = null,
                tint = Color.White,
                modifier = Modifier.size(16.dp)
            )
        }
    }
}

@Composable
fun GlassBottomNavBar(
    selectedTab: BottomNavTab,
    isTamil: Boolean,
    onSelectTab: (BottomNavTab) -> Unit,
    modifier: Modifier = Modifier
) {
    val barShape = RoundedCornerShape(28.dp)
    Box(
        modifier = modifier
            .fillMaxWidth()
            .windowInsetsPadding(WindowInsets.navigationBars)
            .padding(horizontal = 14.dp, vertical = 8.dp)
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceEvenly,
            modifier = Modifier
                .fillMaxWidth()
                .shadow(12.dp, barShape, spotColor = Color(0x664A154B))
                .clip(barShape)
                .background(
                    Brush.horizontalGradient(
                        listOf(
                            Color(0xCC5A305E),
                            Color(0xCC4B2854),
                            Color(0xCC5E3360)
                        )
                    )
                )
                .border(
                    width = 1.3.dp,
                    brush = Brush.verticalGradient(
                        listOf(Color.White.copy(alpha = 0.78f), Color.White.copy(alpha = 0.25f))
                    ),
                    shape = barShape
                )
                .padding(horizontal = 6.dp, vertical = 6.dp)
        ) {
            val items = listOf(
                Triple(
                    BottomNavTab.HOME,
                    if (isTamil) "முகப்பு" else "Home",
                    Icons.Filled.Home to Icons.Outlined.Home
                ),
                Triple(
                    BottomNavTab.FAVORITES,
                    if (isTamil) "பிடித்தவை" else "Favorites",
                    Icons.Filled.Star to Icons.Outlined.StarOutline
                ),
                Triple(
                    BottomNavTab.MY_CREATIONS,
                    if (isTamil) "படைப்புகள்" else "My Creations",
                    Icons.Filled.Favorite to Icons.Outlined.FavoriteBorder
                ),
                Triple(
                    BottomNavTab.MODELS,
                    if (isTamil) "மாடல்கள்" else "Models",
                    Icons.Filled.CloudDownload to Icons.Outlined.CloudDownload
                )
            )

            items.forEach { (tab, label, iconPair) ->
                val isSelected = selectedTab == tab
                val pillShape = RoundedCornerShape(20.dp)
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center,
                    modifier = Modifier
                        .weight(1f)
                        .minimumInteractiveComponentSize()
                        .clip(pillShape)
                        .then(
                            if (isSelected) {
                                Modifier
                                    .background(
                                        Brush.verticalGradient(
                                            listOf(
                                                PinkGlowStart.copy(alpha = 0.85f),
                                                PinkGlowEnd.copy(alpha = 0.9f)
                                            )
                                        )
                                    )
                                    .border(
                                        1.2.dp,
                                        Color.White.copy(alpha = 0.85f),
                                        pillShape
                                    )
                            } else {
                                Modifier
                            }
                        )
                        .clickable { onSelectTab(tab) }
                        .padding(vertical = 6.dp, horizontal = 4.dp)
                        .testTag("nav_tab_${tab.name.lowercase()}")
                ) {
                    Icon(
                        imageVector = if (isSelected) iconPair.first else iconPair.second,
                        contentDescription = label,
                        tint = Color.White,
                        modifier = Modifier.size(22.dp)
                    )
                    Spacer(modifier = Modifier.height(2.dp))
                    Text(
                        text = label,
                        color = Color.White,
                        style = MaterialTheme.typography.labelSmall.copy(
                            fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Medium,
                            fontSize = 11.sp
                        ),
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                }
            }
        }
    }
}
