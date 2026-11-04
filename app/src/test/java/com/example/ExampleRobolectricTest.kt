package com.example

import android.content.Context
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.test.core.app.ApplicationProvider
import com.example.ai.DiffusionModelSlot
import com.example.ai.OnnxModelManager
import com.example.ai.RealDeviceAndModelMetrics
import com.example.data.AppPreferencesState
import com.example.ui.screens.SettingsScreen
import com.example.ui.theme.MyApplicationTheme
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.ByteArrayOutputStream
import java.io.File

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [36])
class ExampleRobolectricTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    @Test
    fun `read string from context`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val appName = context.getString(R.string.app_name)
        assertEquals("Image Generator", appName)
    }

    @Test
    fun `incomplete unet shell without weights pb is marked not ready until companion attached`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val manager = OnnxModelManager(context)
        manager.deleteAllModels()

        val unetDir = File(manager.modelsRootDir, DiffusionModelSlot.UNET.folderHint).apply { mkdirs() }
        val unetGraph = File(unetDir, "model.onnx")
        val locBytes = "model.onnx.data".toByteArray(Charsets.UTF_8)
        val shellPayload = ByteArrayOutputStream().apply {
            write("onnx_graph_shell_".toByteArray(Charsets.UTF_8))
            write(byteArrayOf(0x0A, 0x08))
            write("location".toByteArray(Charsets.UTF_8))
            write(byteArrayOf(0x12, locBytes.size.toByte()))
            write(locBytes)
            write(ByteArray(4096))
        }.toByteArray()
        unetGraph.writeBytes(shellPayload)

        val metricsBefore = manager.inspectCurrentMetrics()
        val unetBefore = metricsBefore.slots.first { it.slot == DiffusionModelSlot.UNET }
        assertFalse(unetBefore.isInstalled)
        assertTrue(unetBefore.isIncompleteExternalGraph)
        assertEquals(listOf("model.onnx.data"), unetBefore.missingCompanionFiles)

        // Attach companion weights -> slot becomes Ready!
        File(unetDir, "model.onnx.data").writeBytes(ByteArray(80 * 1024) { 2 })
        val metricsAfter = manager.inspectCurrentMetrics()
        val unetAfter = metricsAfter.slots.first { it.slot == DiffusionModelSlot.UNET }
        assertTrue(unetAfter.isInstalled)
        assertFalse(unetAfter.isIncompleteExternalGraph)
        assertTrue(unetAfter.missingCompanionFiles.isEmpty())

        manager.deleteAllModels()
    }

    @Test
    fun `settings screen displays onnx model download links card`() {
        composeTestRule.setContent {
            MyApplicationTheme(darkTheme = false) {
                SettingsScreen(
                    preferences = AppPreferencesState(),
                    metrics = RealDeviceAndModelMetrics(),
                    onBack = {},
                    onOpenModelsManager = {},
                    onOpenContentSafetyPage = {},
                    onResolutionChange = {},
                    onInferenceModeChange = {},
                    onOfflineModeChange = {},
                    onLanguageChange = {},
                    onDarkThemeChange = {},
                    onNotificationsChange = {},
                    onContentFilterToggle = {},
                    onClearCache = {},
                    onUnloadRam = {}
                )
            }
        }
        composeTestRule.waitForIdle()

        composeTestRule.onNodeWithTag("settings_onnx_download_card")
            .performScrollTo()
            .assertIsDisplayed()
        composeTestRule.onNodeWithTag("settings_download_open_lcm_unet_data")
            .performScrollTo()
            .assertIsDisplayed()
        composeTestRule.onNodeWithTag("settings_toggle_extra_links")
            .performScrollTo()
            .performClick()
        composeTestRule.waitForIdle()
        composeTestRule.onNodeWithTag("settings_download_open_sd_compvis_weights_pb")
            .performScrollTo()
            .assertIsDisplayed()
    }

    @Test
    fun `home screen displays 6 realistic cards without face swap and opens avatar customization`() {
        var navigatedScreen: com.example.ui.ActiveScreen? = null
        composeTestRule.setContent {
            MyApplicationTheme(darkTheme = false) {
                com.example.ui.screens.HomeScreen(
                    preferences = AppPreferencesState(),
                    metrics = RealDeviceAndModelMetrics(),
                    onOpenSettings = {},
                    onSelectFeature = { navigatedScreen = it },
                    onOpenModelsTab = {}
                )
            }
        }
        composeTestRule.waitForIdle()

        // Face Swap card must be completely removed
        composeTestRule.onNodeWithTag("card_face_swap").assertDoesNotExist()

        // All 6 realistic feature cards must exist and render
        val expectedTags = listOf(
            "card_text_to_image",
            "card_image_to_image",
            "card_character_studio",
            "card_avatar_customization",
            "card_image_editor",
            "card_my_gallery"
        )
        for (tag in expectedTags) {
            composeTestRule.onNodeWithTag(tag)
                .performScrollTo()
                .assertIsDisplayed()
        }

        // Tapping Avatar Customization opens ActiveScreen.AVATAR_CUSTOMIZATION
        composeTestRule.onNodeWithTag("card_avatar_customization")
            .performScrollTo()
            .performClick()
        composeTestRule.waitForIdle()
        assertEquals(com.example.ui.ActiveScreen.AVATAR_CUSTOMIZATION, navigatedScreen)

        // Tapping Text to Image navigates normally
        composeTestRule.onNodeWithTag("card_text_to_image")
            .performScrollTo()
            .performClick()
        composeTestRule.waitForIdle()
        assertEquals(com.example.ui.ActiveScreen.TEXT_TO_IMAGE, navigatedScreen)
    }

    @Test
    fun `avatar customization screen renders 4 tabs realistic presets sliders and slot 7 in models tab`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val manager = OnnxModelManager(context)
        val realMetrics = manager.inspectCurrentMetrics()

        composeTestRule.setContent {
            var currentParams = androidx.compose.runtime.remember {
                androidx.compose.runtime.mutableStateOf(com.example.ai.AvatarCustomizationParams())
            }
            MyApplicationTheme(darkTheme = false) {
                com.example.ui.screens.AvatarCustomizationScreen(
                    preferences = AppPreferencesState(),
                    metrics = realMetrics,
                    sourcePhotoUri = null,
                    hasLatestAppCreation = false,
                    params = currentParams.value,
                    userBrushStrokes = emptyList(),
                    visionAnalysis = null,
                    maskOverlayBitmap = null,
                    latestAvatarCreation = null,
                    latestAvatarSourcePath = null,
                    isGenerating = false,
                    progress = null,
                    errorMessage = null,
                    isDownloadingMediaPipe = false,
                    mediaPipeDownloadStatus = null,
                    onSelectEngine = {},
                    onPickSourcePhoto = {},
                    onUseLatestAppCreation = {},
                    onParamsChange = { currentParams.value = it },
                    onAddBrushStroke = {},
                    onUndoBrushStroke = {},
                    onResetBrushStrokes = {},
                    onDownloadMediaPipeTasks = {},
                    onGenerateAvatar = {},
                    onCancelGeneration = {},
                    onUseResultAsNextSource = {},
                    onToggleFavorite = {},
                    onExportToGallery = {},
                    onOpenModelsTab = {},
                    onOpenContentSafetySettings = {},
                    onBack = {}
                )
            }
        }
        composeTestRule.waitForIdle()

        // Default Tab 1: Age Progression presets + sliders + shared sliders
        composeTestRule.onNodeWithTag("avatar_preset_age_senior_72").performScrollTo().assertIsDisplayed()
        composeTestRule.onNodeWithTag("avatar_slider_target_age").performScrollTo().assertIsDisplayed()
        composeTestRule.onNodeWithTag("avatar_slider_transform_strength").performScrollTo().assertIsDisplayed()
        composeTestRule.onNodeWithTag("avatar_slider_identity_lock").performScrollTo().assertIsDisplayed()

        // Switch to Tab 2: Body / Abs
        composeTestRule.onNodeWithTag("avatar_tab_body_abs").performScrollTo().performClick()
        composeTestRule.waitForIdle()
        composeTestRule.onNodeWithTag("avatar_preset_body_six_pack_abs").performScrollTo().assertIsDisplayed()
        composeTestRule.onNodeWithTag("avatar_slider_abs_definition").performScrollTo().assertIsDisplayed()

        // Switch to Tab 3: Hairstyle
        composeTestRule.onNodeWithTag("avatar_tab_hairstyle").performScrollTo().performClick()
        composeTestRule.waitForIdle()
        composeTestRule.onNodeWithTag("avatar_preset_hair_layered_waves").performScrollTo().assertIsDisplayed()
        composeTestRule.onNodeWithTag("avatar_slider_hair_volume").performScrollTo().assertIsDisplayed()

        // Switch to Tab 4: Outfit / Clothes
        composeTestRule.onNodeWithTag("avatar_tab_outfit").performScrollTo().performClick()
        composeTestRule.waitForIdle()
        composeTestRule.onNodeWithTag("avatar_preset_outfit_silk_saree").performScrollTo().assertIsDisplayed()
        composeTestRule.onNodeWithTag("avatar_slider_cloth_draping").performScrollTo().assertIsDisplayed()
    }

    @Test
    fun `models tab screen displays optional slot 7 aging engine and mediapipe vision card`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val realMetrics = OnnxModelManager(context).inspectCurrentMetrics()

        composeTestRule.setContent {
            MyApplicationTheme(darkTheme = false) {
                com.example.ui.screens.ModelsTabScreen(
                    preferences = AppPreferencesState(),
                    metrics = realMetrics,
                    isImporting = false,
                    importProgressText = null,
                    verificationReport = null,
                    onOneTouchFolderUpload = {},
                    onVerifyPipeline = {}
                )
            }
        }
        composeTestRule.waitForIdle()

        composeTestRule.onNodeWithTag("upload_slot_7_face_aging")
            .performScrollTo()
            .assertIsDisplayed()
        composeTestRule.onNodeWithTag("models_download_mediapipe_button")
            .performScrollTo()
            .assertIsDisplayed()
    }
}
