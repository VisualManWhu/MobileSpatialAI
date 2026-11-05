package dev.recon3d.rooms

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class DeleteDialogTest {
    @Test fun choicesAndLabels() {
        assertEquals(listOf("彻底删除", "仅删手机文件"), captureDeleteChoices(listOf(DeleteMode.CLOUD, DeleteMode.LOCAL)).map { it.label })
        assertEquals(listOf("删除"), captureDeleteChoices(listOf(DeleteMode.LOCAL)).map { it.label })
        assertEquals(listOf("彻底删除"), captureDeleteChoices(listOf(DeleteMode.CLOUD)).map { it.label })
        assertEquals(listOf("彻底删除", "仅删手机"), roomDeleteChoices(listOf(DeleteMode.CLOUD, DeleteMode.LOCAL)).map { it.label })
        assertEquals(listOf("删除"), roomDeleteChoices(listOf(DeleteMode.LOCAL)).map { it.label })
        assertTrue(captureDeleteChoices(listOf(DeleteMode.CLOUD)).single().destructive)
        assertFalse(captureDeleteChoices(listOf(DeleteMode.CLOUD, DeleteMode.LOCAL))[1].destructive)
    }

    @Test fun bodiesFollowSpec() {
        val rec = CaptureRecord(id = "cap-1", zipPath = "")
        assertEquals("删除「cap-1」", captureDeleteTitle(rec))
        assertEquals("删除手机上的拍摄包，不可恢复。", captureDeleteBody(listOf(DeleteMode.LOCAL)))
        assertEquals("手机和计算端上的拍摄包与模型都删掉，不可恢复。", captureDeleteBody(listOf(DeleteMode.CLOUD)))
        assertEquals(
            "彻底删除：手机和计算端上的拍摄包与模型都删掉，不可恢复。\n仅删手机文件：释放手机空间，保留记录，之后可从计算端重新下载模型。",
            captureDeleteBody(listOf(DeleteMode.CLOUD, DeleteMode.LOCAL)),
        )
        val room = Room("r", "客厅", 0, captures = listOf(rec, rec.copy(id = "cap-2")))
        assertEquals("删除「客厅」", roomDeleteTitle(room))
        assertEquals(
            "彻底删除：这个房间的 2 条拍摄在手机和计算端都删掉，不可恢复。\n仅删手机：删掉手机上的房间和文件，计算端的任务保留。",
            roomDeleteBody(room, listOf(DeleteMode.CLOUD, DeleteMode.LOCAL)),
        )
        assertEquals("删除手机上的这个房间，不可恢复。", roomDeleteBody(Room("r", "空", 0), listOf(DeleteMode.LOCAL)))
    }
}
