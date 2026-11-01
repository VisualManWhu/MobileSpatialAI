package dev.recon3d.capture

import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.hypot

enum class Stage { SPIN, SWEEP, ROAM, DONE }
enum class Arrow { LEFT, RIGHT, UP, DOWN, FORWARD_LEFT, FORWARD_RIGHT, NONE }

data class GuideUi(
    val stage: Stage,
    val stageProgress: Float,
    val arrow: Arrow,
    val hint: String,
    val subHint: String,
    val warning: String?,
    val canFinish: Boolean,
)

/** Three-stage capture guide (SPIN -> SWEEP -> ROAM -> DONE). Pure Kotlin, no Android deps. */
class CaptureGuide {
    private companion object {
        const val SPIN_BINS = 36
        const val SPIN_NEED = 33
        const val SPIN_PITCH_LIMIT = 25f
        const val SWEEP_PITCH = 45f
        const val ROAM_SECTORS = 8
        const val ROAM_NEED = 24
        const val ROAM_PATH_M = 1.5f
        const val CELL_M = 0.5f
        const val MAX_DEG_PER_S = 60.0
    }

    var stage: Stage = Stage.SPIN
        private set

    // SPIN
    private val spinBins = BooleanArray(SPIN_BINS)
    private var anchorSumX = 0.0
    private var anchorSumZ = 0.0
    private var anchorN = 0

    // SWEEP
    private var reachedUp = false
    private var reachedDown = false

    // ROAM
    private var anchorX = 0f
    private var anchorZ = 0f
    private val roamCombos = HashSet<Triple<Int, Int, Int>>()
    private val sectorCounts = IntArray(ROAM_SECTORS)
    private var pathLen = 0f

    // frame tracking
    private var prevT = 0L
    private var prevYaw = 0f
    private var prevPitch = 0f
    private var prevX = 0f
    private var prevZ = 0f
    private var havePrev = false
    private var curYaw = 0f

    var last: GuideUi = build(null)
        private set

    fun onFrame(tNs: Long, poseGl: FloatArray, tracking: Boolean): GuideUi {
        if (!tracking) {
            havePrev = false // avoid spurious velocity/path jump on resume
            last = build("跟丢了，请回到刚才的位置", lost = true)
            return last
        }
        val (yaw, pitch) = PoseMath.yawPitchDeg(poseGl)
        val x = poseGl[12]
        val z = poseGl[14]
        curYaw = yaw

        var warning: String? = null
        if (havePrev) {
            val dt = (tNs - prevT) / 1e9
            if (dt > 0) {
                val dYaw = wrap180(yaw - prevYaw)
                val dPitch = pitch - prevPitch
                if (hypot(dYaw.toDouble(), dPitch.toDouble()) / dt > MAX_DEG_PER_S) warning = "太快了，慢一点"
            }
        }

        when (stage) {
            Stage.SPIN -> {
                anchorSumX += x; anchorSumZ += z; anchorN++
                if (abs(pitch) <= SPIN_PITCH_LIMIT) spinBins[bin(yaw, SPIN_BINS)] = true
                if (spinBins.count { it } >= SPIN_NEED) {
                    anchorX = (anchorSumX / anchorN).toFloat()
                    anchorZ = (anchorSumZ / anchorN).toFloat()
                    stage = Stage.SWEEP
                }
            }
            Stage.SWEEP -> {
                if (pitch >= SWEEP_PITCH) reachedUp = true
                if (pitch <= -SWEEP_PITCH) reachedDown = true
                if (reachedUp && reachedDown) {
                    stage = Stage.ROAM
                    // path length starts counting from the next ROAM frame
                    prevT = tNs; prevYaw = yaw; prevPitch = pitch; prevX = x; prevZ = z; havePrev = true
                    last = build(warning)
                    return last
                }
            }
            Stage.ROAM -> {
                if (havePrev) pathLen += hypot((x - prevX).toDouble(), (z - prevZ).toDouble()).toFloat()
                val cx = floor((x - anchorX) / CELL_M).toInt()
                val cz = floor((z - anchorZ) / CELL_M).toInt()
                if (!(cx == 0 && cz == 0)) {
                    val s = bin(yaw, ROAM_SECTORS)
                    if (roamCombos.add(Triple(cx, cz, s))) sectorCounts[s]++
                }
                if (roamCombos.size >= ROAM_NEED && pathLen >= ROAM_PATH_M) stage = Stage.DONE
            }
            Stage.DONE -> {}
        }

        prevT = tNs; prevYaw = yaw; prevPitch = pitch; prevX = x; prevZ = z; havePrev = true
        last = build(warning)
        return last
    }

    fun skipStage() {
        when (stage) {
            Stage.SWEEP -> { stage = Stage.ROAM; havePrev = false }
            Stage.ROAM -> stage = Stage.DONE
            else -> return
        }
        last = build(null)
    }

    private fun build(warning: String?, lost: Boolean = false): GuideUi {
        val canFinish = stage != Stage.SPIN
        return when (stage) {
            Stage.SPIN -> {
                val covered = spinBins.count { it }
                GuideUi(
                    Stage.SPIN, (covered.toFloat() / SPIN_NEED).coerceAtMost(1f),
                    if (lost) Arrow.NONE else spinArrow(),
                    "先完成原地转圈：站在房间中央，慢慢向右转", "保持手机与胸口同高", warning, false,
                )
            }
            Stage.SWEEP -> {
                val p = (if (reachedUp) 0.5f else 0f) + (if (reachedDown) 0.5f else 0f)
                val arrow = if (lost) Arrow.NONE else if (!reachedUp) Arrow.UP else Arrow.DOWN
                val hint = if (reachedUp) "然后向下扫到地面" else "慢慢抬起，拍天花板"
                val sub = if (reachedUp) "" else "然后向下扫到地面"
                GuideUi(Stage.SWEEP, p, arrow, hint, sub, warning, canFinish)
            }
            Stage.ROAM -> {
                val p = minOf(roamCombos.size / ROAM_NEED.toFloat(), pathLen / ROAM_PATH_M).coerceIn(0f, 1f)
                GuideUi(Stage.ROAM, p, if (lost) Arrow.NONE else roamArrow(), "向箭头方向走两步", "还有角落没拍到", warning, canFinish)
            }
            Stage.DONE -> GuideUi(Stage.DONE, 1f, Arrow.NONE, "拍摄完成，可以上传了", "", warning, true)
        }
    }

    private fun spinArrow(): Arrow {
        if (spinBins.all { it }) return Arrow.RIGHT
        val cur = bin(curYaw, SPIN_BINS)
        var right = Int.MAX_VALUE
        var left = Int.MAX_VALUE
        for (k in 1 until SPIN_BINS) {
            if (right == Int.MAX_VALUE && !spinBins[(cur + k) % SPIN_BINS]) right = k
            if (left == Int.MAX_VALUE && !spinBins[((cur - k) % SPIN_BINS + SPIN_BINS) % SPIN_BINS]) left = k
        }
        return if (left < right) Arrow.LEFT else Arrow.RIGHT
    }

    private fun roamArrow(): Arrow {
        val sector = 360f / ROAM_SECTORS
        var bestDelta = 0f
        var bestCount = Int.MAX_VALUE
        var bestAbs = Float.MAX_VALUE
        for (s in 0 until ROAM_SECTORS) {
            val center = -180f + sector * s + sector / 2
            val delta = wrap180(center - curYaw)
            val c = sectorCounts[s]
            if (c < bestCount || (c == bestCount && abs(delta) < bestAbs)) {
                bestCount = c; bestAbs = abs(delta); bestDelta = delta
            }
        }
        return when {
            abs(bestDelta) <= 45f -> if (bestDelta < 0) Arrow.FORWARD_LEFT else Arrow.FORWARD_RIGHT
            bestDelta < 0 -> Arrow.LEFT
            else -> Arrow.RIGHT
        }
    }

    private fun bin(yawDeg: Float, n: Int): Int {
        val w = (((yawDeg + 180f) % 360f) + 360f) % 360f
        return floor(w / (360f / n)).toInt().coerceIn(0, n - 1)
    }

    private fun wrap180(d: Float): Float {
        var r = ((d + 180f) % 360f + 360f) % 360f - 180f
        if (r == -180f) r = 180f
        return r
    }
}
