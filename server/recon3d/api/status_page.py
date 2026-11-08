"""The `/` status page: a static HTML shell that polls GET /api/v1/jobs every 5 s."""

HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Recon3D 任务</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 24px; background: #f6f6f6; color: #222; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  #meta { color: #666; font-size: 13px; margin-bottom: 16px; }
  table { border-collapse: collapse; width: 100%; background: #fff; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #e5e5e5; font-size: 14px; vertical-align: top; }
  th { background: #fafafa; font-weight: 600; }
  .bar { height: 8px; background: #e5e5e5; border-radius: 4px; overflow: hidden; width: 160px; }
  .bar i { display: block; height: 100%; background: #2e7dff; }
  .uploading .bar i { background: #f0a020; }
  .done .bar i { background: #3cb371; }
  .failed .bar i { background: #e33; }
  .failed td.msg { color: #c00; }
  .objfail { color: #c00; }
  code { font-size: 13px; }
  #empty { color: #666; padding: 16px 0; }
</style>
</head>
<body>
<h1>Recon3D 任务</h1>
<div id="meta">加载中…</div>
<table id="jobs" hidden>
  <thead><tr><th>任务 ID</th><th>拍摄</th><th>状态</th><th>模式</th><th>进度</th><th>说明</th><th>创建时间</th><th>拍摄包</th><th>结果</th><th>物体</th></tr></thead>
  <tbody></tbody>
</table>
<div id="empty" hidden>还没有任务。手机上传拍摄包后会出现在这里。</div>
<script>
const STATE = { uploading: "上传中", queued: "排队中", training: "训练中", exporting: "导出中", done: "完成", failed: "失败" };
function fmtBytes(n) {
  if (n >= 1024 ** 3) return (n / 1024 ** 3).toFixed(2) + " GB";
  if (n >= 1024 ** 2) return (n / 1024 ** 2).toFixed(1) + " MB";
  return Math.round(n / 1024) + " KB";
}
function fmtTime(t) {
  const d = new Date(t * 1000);
  return d.toLocaleString("zh-CN", { hour12: false });
}
function objectsCell(j) {
  const o = j.objects || { state: "none" };
  if (o.state === "none") return "无";
  if (o.state === "queued") return "排队";
  if (o.state === "running") return `识别中 ${Math.round(o.progress * 100)}%`;
  if (o.state === "done") return `<a href="/api/v1/jobs/${j.job_id}/objects.zip">${o.count} 个</a>`;
  return `<span class="objfail">失败</span>`;
}
function esc(s) {
  return String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
async function refresh() {
  let data;
  try {
    const [h, j] = await Promise.all([fetch("/api/v1/health"), fetch("/api/v1/jobs")]);
    const health = await h.json();
    data = await j.json();
    document.getElementById("meta").textContent =
      `GPU：${health.gpu} · 档位：${health.profile} · 版本 ${health.version} · 每 5 秒刷新，最近 ${new Date().toLocaleTimeString("zh-CN", { hour12: false })}`;
  } catch (e) {
    document.getElementById("meta").textContent = "连不上服务端，稍后重试…";
    return;
  }
  const rows = data.jobs.map((j) => {
    const pct = Math.round(j.progress * 100);
    const result = j.has_scene
      ? `<a href="/api/v1/jobs/${j.job_id}/scene.splat">scene.splat</a>`
      : "";
    return `<tr class="${esc(j.state)}">
      <td><code>${esc(j.job_id)}</code></td>
      <td><code>${esc(j.capture_id ?? "")}</code></td>
      <td>${esc(STATE[j.state] || j.state)}</td>
      <td>${j.mode === "precise" ? "高精" : "快速"}</td>
      <td><div class="bar"><i style="width:${pct}%"></i></div>${pct}%</td>
      <td class="msg">${esc(j.message)}</td>
      <td>${fmtTime(j.created_at)}</td>
      <td>${fmtBytes(j.capture_bytes)}</td>
      <td>${result}</td>
      <td>${objectsCell(j)}</td>
    </tr>`;
  });
  document.querySelector("#jobs tbody").innerHTML = rows.join("");
  document.getElementById("jobs").hidden = rows.length === 0;
  document.getElementById("empty").hidden = rows.length !== 0;
}
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""
