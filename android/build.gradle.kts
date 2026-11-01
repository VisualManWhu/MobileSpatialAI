plugins {
    alias(libs.plugins.android.application) apply false
    // AGP 9 has built-in Kotlin support and only pulls in an older Kotlin Gradle plugin.
    // Declaring kotlin-android here (never applied) pins KGP — and thus the Kotlin compiler —
    // to libs.versions.kotlin, which the Compose and serialization compiler plugins must match.
    alias(libs.plugins.kotlin.android) apply false
    alias(libs.plugins.kotlin.compose) apply false
    alias(libs.plugins.kotlin.serialization) apply false
}
