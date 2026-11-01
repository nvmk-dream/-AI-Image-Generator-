package com.example.ui.theme

import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp

private val LightGlassColorScheme = lightColorScheme(
    primary = PurpleActionEnd,
    onPrimary = Color.White,
    primaryContainer = LightGlassHighlight,
    onPrimaryContainer = RoyalIndigoTitle,
    secondary = VibrantPinkPill,
    onSecondary = Color.White,
    tertiary = BlueGlowEnd,
    background = SoftRoseBackground,
    onBackground = DeepPlumText,
    surface = LightGlassSurface,
    onSurface = DeepPlumText,
    surfaceVariant = LightGlassBorder,
    onSurfaceVariant = SoftPlumSubtext
)

private val DarkGlassColorScheme = darkColorScheme(
    primary = PinkGlowStart,
    onPrimary = DeepVelvetBackground,
    primaryContainer = DarkGlassHighlight,
    onPrimaryContainer = Color.White,
    secondary = VibrantPinkPill,
    onSecondary = Color.White,
    tertiary = CyanVersionAccent,
    background = DeepVelvetBackground,
    onBackground = Color.White,
    surface = DarkGlassSurface,
    onSurface = Color.White,
    surfaceVariant = DarkGlassBorder,
    onSurfaceVariant = Color(0xFFE5CCF7)
)

val AppShapes = Shapes(
    small = RoundedCornerShape(12.dp),
    medium = RoundedCornerShape(20.dp),
    large = RoundedCornerShape(26.dp),
    extraLarge = RoundedCornerShape(32.dp)
)

@Composable
fun MyApplicationTheme(
    darkTheme: Boolean = false,
    content: @Composable () -> Unit
) {
    val colorScheme = if (darkTheme) DarkGlassColorScheme else LightGlassColorScheme
    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography,
        shapes = AppShapes,
        content = content
    )
}
