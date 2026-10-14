const state = { summary: null, runs: [], filtered: [], downloads: null };
let downloadPollTimer = null;

const $ = (selector) => document.querySelector(selector);
const fmtNumber = new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 4 });
const fmtDate = new Intl.DateTimeFormat("zh-CN", { dateStyle: "short", timeStyle: "short" });

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function metricText(metric) {
  if (!metric) return "—";
  const suffix = metric.unit === "%" ? "%" : metric.unit ? ` ${metric.unit}` : "";
  return `${fmtNumber.format(metric.value)}${suffix}`;
}

function setAlert(message = "") {
  const alert = $("#alert");
  alert.textContent = message;
  alert.classList.toggle("hidden", !message);
}

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

async function postJson(url) {
  const response = await fetch(url, { method: "POST" });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail ?? `${response.status} ${response.statusText}`);
  return payload;
}

function formatBytes(value) {
  if (value == null) return "未知";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let amount = value;
  let unit = 0;
  while (amount >= 1024 && unit < units.length - 1) { amount /= 1024; unit += 1; }
  return `${amount >= 100 || unit === 0 ? amount.toFixed(0) : amount.toFixed(1)} ${units[unit]}`;
}

function formatDuration(seconds) {
  if (seconds == null || !Number.isFinite(seconds)) return "";
  if (seconds < 60) return `${Math.ceil(seconds)} 秒`;
  if (seconds < 3600) return `${Math.ceil(seconds / 60)} 分钟`;
  return `${(seconds / 3600).toFixed(1)} 小时`;
}

const downloadStateLabels = {
  missing: "未下载",
  partial: "部分文件",
  present: "待校验",
  downloading: "下载中",
  queued: "排队中",
  verifying: "校验中",
  paused: "已暂停",
  failed: "失败",
  complete: "已就绪",
  unavailable: "缺少工具",
};

function resourceActions(item) {
  const actions = [`<a class="resource-link" href="${escapeHtml(item.source_page)}" target="_blank" rel="noreferrer">官方来源 ↗</a>`];
  if (item.can_pause) {
    actions.unshift(`<button class="button resource-action" data-resource="${escapeHtml(item.resource_id)}" data-action="pause">暂停</button>`);
  } else if (item.can_verify) {
    actions.unshift(`<button class="button resource-action" data-resource="${escapeHtml(item.resource_id)}" data-action="verify">校验</button>`);
  } else if (item.can_start) {
    const label = ["partial", "paused", "failed", "present"].includes(item.state) ? "继续下载" : "开始下载";
    actions.unshift(`<button class="button resource-action" data-resource="${escapeHtml(item.resource_id)}" data-action="start">${label}</button>`);
  }
  return actions.join("");
}

function resourceCard(item) {
  const progress = item.progress == null ? 0 : Math.max(0, Math.min(100, item.progress * 100));
  const sizePrefix = item.size_is_estimate ? "≈ " : "";
  const transfer = item.state === "downloading" && item.speed_bps > 0
    ? `${formatBytes(item.speed_bps)}/s${item.eta_seconds ? ` · 剩余 ${formatDuration(item.eta_seconds)}` : ""}`
    : `${formatBytes(item.bytes_downloaded)} / ${sizePrefix}${formatBytes(item.bytes_total)} · 磁盘可用 ${formatBytes(item.disk_free_bytes)}`;
  return `
    <article class="resource-card resource-${escapeHtml(item.state)}">
      <div class="resource-card-top">
        <span class="resource-kind">${item.kind === "model" ? "MODEL" : "DATASET"}</span>
        <span class="download-status status-${escapeHtml(item.state)}">${escapeHtml(downloadStateLabels[item.state] ?? item.state)}</span>
      </div>
      <h3>${escapeHtml(item.display_name)}</h3>
      <p>${escapeHtml(item.description)}</p>
      <div class="resource-tags">
        ${item.task_ids.map((task) => `<span>${escapeHtml(task)}</span>`).join("")}
        <span>${escapeHtml(item.license)}</span><span>${escapeHtml(item.engine ?? item.method)}</span>
      </div>
      <div class="progress-track"><div class="progress-fill" style="width:${progress.toFixed(2)}%"></div></div>
      <div class="progress-copy"><span>${progress.toFixed(1)}%</span><span>${escapeHtml(transfer)}</span></div>
      <div class="resource-path" title="${escapeHtml(item.destination)}">${escapeHtml(item.destination)}</div>
      ${item.notes ? `<div class="resource-note">${escapeHtml(item.notes)}</div>` : ""}
      ${item.error ? `<div class="resource-error">${escapeHtml(item.error)}</div>` : ""}
      <div class="resource-actions">${resourceActions(item)}</div>
    </article>`;
}

function renderDownloads() {
  const catalog = state.downloads;
  if (!catalog) return;
  $("#resource-count").textContent = `${catalog.resource_count} RESOURCES`;
  $("#resource-complete").textContent = catalog.state_counts.complete ?? 0;
  $("#resource-active").textContent = (catalog.state_counts.downloading ?? 0) + (catalog.state_counts.verifying ?? 0);
  $("#resource-local-size").textContent = formatBytes(catalog.downloaded_bytes);
  $("#resource-total-size").textContent = `≈ ${formatBytes(catalog.expected_size_bytes)}`;
  const groups = [
    ["model", "模型权重"],
    ["dataset", "任务数据"],
  ];
  $("#resource-groups").innerHTML = groups.map(([kind, label]) => {
    const items = catalog.items.filter((item) => item.kind === kind);
    return `<section class="resource-group"><div class="resource-group-title"><h3>${label}</h3><span>${items.length}</span></div><div class="resource-grid">${items.map(resourceCard).join("")}</div></section>`;
  }).join("");
  if (!catalog.enabled) setAlert("Portal 下载控制已关闭；设置 RS_EVAL_DOWNLOADS_ENABLED=true 后重启可启用。");
}

function scheduleDownloadPoll() {
  window.clearTimeout(downloadPollTimer);
  const active = state.downloads?.items.some((item) => ["queued", "downloading", "verifying"].includes(item.state));
  downloadPollTimer = window.setTimeout(loadDownloads, active ? 1500 : 10000);
}

async function loadDownloads() {
  try {
    state.downloads = await getJson("/api/downloads");
    renderDownloads();
  } catch (error) {
    setAlert(`加载下载状态失败：${error.message}`);
  } finally {
    scheduleDownloadPoll();
  }
}

async function handleResourceAction(button) {
  const resourceId = button.dataset.resource;
  const action = button.dataset.action;
  button.disabled = true;
  try {
    await postJson(`/api/downloads/${encodeURIComponent(resourceId)}/${action}`);
    await loadDownloads();
  } catch (error) {
    setAlert(`${action === "pause" ? "暂停" : action === "verify" ? "校验" : "启动"}失败：${error.message}`);
    button.disabled = false;
  }
}

function setSelectOptions(element, values, label) {
  const current = element.value;
  element.innerHTML = `<option value="">全部${label}</option>`;
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    element.appendChild(option);
  });
  element.value = values.includes(current) ? current : "";
}

function renderSummary() {
  const summary = state.summary;
  $("#stat-total").textContent = summary.run_count;
  $("#stat-completed").textContent = summary.status_counts.completed ?? 0;
  $("#stat-running").textContent = summary.status_counts.running ?? 0;
  $("#stat-invalid").textContent = summary.invalid_run_count;
  setSelectOptions($("#task-filter"), summary.task_ids, "任务");
  setSelectOptions($("#model-filter"), summary.model_ids, "模型");
  setSelectOptions($("#protocol-filter"), summary.protocol_ids, "协议");
  setSelectOptions($("#status-filter"), Object.keys(summary.status_counts).sort(), "状态");
  if (summary.invalid_run_count) setAlert(`${summary.invalid_run_count} 个 run.json 未通过契约校验，请查看 API summary.errors。`);
}

function applyFilters() {
  const task = $("#task-filter").value;
  const model = $("#model-filter").value;
  const protocol = $("#protocol-filter").value;
  const status = $("#status-filter").value;
  state.filtered = state.runs.filter((run) =>
    (!task || run.task_id === task) &&
    (!model || run.model_id === model) &&
    (!protocol || run.protocol_id === protocol) &&
    (!status || run.status === status)
  );
  $("#visible-count").textContent = `${state.filtered.length} RUNS`;
  renderRuns();
  renderCoverage();
  renderChart();
}

function renderRuns() {
  const body = $("#runs-body");
  body.innerHTML = state.filtered.map((run) => `
    <tr data-run-id="${escapeHtml(run.run_id)}">
      <td><span class="run-id">${escapeHtml(run.run_id)}</span></td>
      <td>${escapeHtml(run.task_id)}</td>
      <td>${escapeHtml(run.model_id)}</td>
      <td>${escapeHtml(run.protocol_id)}</td>
      <td>${run.seed}</td>
      <td><span class="metric">${escapeHtml(metricText(run.primary_metric))}</span><br><span class="muted">${escapeHtml(run.primary_metric?.name ?? "no metric")}</span></td>
      <td><span class="status status-${escapeHtml(run.status)}">${escapeHtml(run.status)}</span></td>
      <td class="muted">${fmtDate.format(new Date(run.updated_at))}</td>
    </tr>`).join("");
  body.querySelectorAll("tr").forEach((row) => row.addEventListener("click", () => openRun(row.dataset.runId)));
  $("#empty-runs").classList.toggle("hidden", state.filtered.length !== 0);
  $(".table-wrap").classList.toggle("hidden", state.filtered.length === 0);
}

function renderCoverage() {
  const tasks = state.summary.task_ids;
  const counts = Object.fromEntries(tasks.map((task) => [task, 0]));
  state.filtered.forEach((run) => { counts[run.task_id] = (counts[run.task_id] ?? 0) + 1; });
  const maximum = Math.max(1, ...Object.values(counts));
  $("#coverage").innerHTML = tasks.map((task) => `
    <div class="coverage-item">
      <span class="coverage-id">${escapeHtml(task)}</span>
      <div class="coverage-track"><div class="coverage-fill" style="width:${(counts[task] / maximum) * 100}%"></div></div>
      <span class="coverage-count">${counts[task]}</span>
    </div>`).join("") || `<div class="muted">尚无任务记录</div>`;
}

function mean(values) { return values.reduce((sum, value) => sum + value, 0) / values.length; }
function std(values) {
  if (values.length < 2) return 0;
  const avg = mean(values);
  return Math.sqrt(values.reduce((sum, value) => sum + (value - avg) ** 2, 0) / (values.length - 1));
}

function renderChart() {
  const target = $("#chart");
  const task = $("#task-filter").value;
  const comparable = state.filtered.filter((run) => run.status === "completed" && run.primary_metric);
  if (!task || !comparable.length) {
    target.className = "chart empty-chart";
    target.innerHTML = `<div><strong>等待可比较结果</strong><span>${task ? "该筛选下没有已完成的主指标。" : "选择一个任务，Portal会按模型和协议汇总不同种子。"}</span></div>`;
    $("#chart-caption").textContent = "选择单个任务后显示";
    return;
  }

  const metricNames = [...new Set(comparable.map((run) => run.primary_metric.name))];
  if (metricNames.length !== 1) {
    target.className = "chart empty-chart";
    target.innerHTML = `<div><strong>主指标口径不一致</strong><span>${escapeHtml(metricNames.join(" / "))}</span></div>`;
    $("#chart-caption").textContent = "请修正运行协议";
    return;
  }

  const groups = new Map();
  comparable.forEach((run) => {
    const key = `${run.model_id} · ${run.protocol_id}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(run.primary_metric.value);
  });
  const bars = [...groups.entries()].map(([label, values]) => ({ label, value: mean(values), std: std(values), n: values.length }));
  const values = bars.map((bar) => bar.value);
  const minValue = Math.min(0, ...values);
  const maxValue = Math.max(...values, 1e-6);
  const width = Math.max(660, bars.length * 120);
  const height = 310, left = 54, right = 20, top = 24, bottom = 74;
  const innerWidth = width - left - right, innerHeight = height - top - bottom;
  const scaleY = (value) => top + innerHeight - ((value - minValue) / (maxValue - minValue || 1)) * innerHeight;
  const slot = innerWidth / bars.length;
  const barWidth = Math.min(62, slot * .58);
  let elements = "";
  for (let tick = 0; tick <= 4; tick += 1) {
    const value = minValue + ((maxValue - minValue) * tick) / 4;
    const y = scaleY(value);
    elements += `<line class="chart-grid" x1="${left}" y1="${y}" x2="${width-right}" y2="${y}"/><text class="chart-label" x="${left-8}" y="${y+4}" text-anchor="end">${fmtNumber.format(value)}</text>`;
  }
  bars.forEach((bar, index) => {
    const x = left + slot * index + (slot - barWidth) / 2;
    const y = scaleY(bar.value);
    const baseY = scaleY(minValue);
    elements += `<rect class="chart-bar" x="${x}" y="${y}" width="${barWidth}" height="${Math.max(1, baseY-y)}" rx="7"/>`;
    elements += `<text class="chart-value" x="${x+barWidth/2}" y="${y-8}" text-anchor="middle">${fmtNumber.format(bar.value)} ± ${fmtNumber.format(bar.std)}</text>`;
    elements += `<text class="chart-label" x="${x+barWidth/2}" y="${height-47}" text-anchor="middle"><tspan x="${x+barWidth/2}">${escapeHtml(bar.label.split(" · ")[0])}</tspan><tspan x="${x+barWidth/2}" dy="15">${escapeHtml(bar.label.split(" · ")[1])} · n=${bar.n}</tspan></text>`;
  });
  target.className = "chart";
  target.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="主指标均值和标准差">${elements}</svg>`;
  $("#chart-caption").textContent = `${metricNames[0]} · mean ± sample std`;
}

function curveSvg(points) {
  if (!points.length) return `<div class="muted">该运行没有记录曲线。</div>`;
  const groups = new Map();
  points.forEach((point) => {
    const key = `${point.split}/${point.name}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(point);
  });
  const selected = [...groups.entries()].slice(0, 4);
  const all = selected.flatMap(([, rows]) => rows);
  const maxStep = Math.max(1, ...all.map((row) => row.step));
  const minValue = Math.min(...all.map((row) => row.value));
  const maxValue = Math.max(...all.map((row) => row.value));
  const width = 620, height = 180, margin = 16;
  const x = (step) => margin + (step / maxStep) * (width - margin * 2);
  const y = (value) => height - margin - ((value - minValue) / (maxValue - minValue || 1)) * (height - margin * 2);
  const colors = ["#65e3b2", "#ffcc75", "#7ab8ff", "#ff7a7a"];
  const paths = selected.map(([key, rows], index) => {
    rows.sort((a, b) => a.step - b.step);
    const pointsText = rows.map((row) => `${x(row.step)},${y(row.value)}`).join(" ");
    return `<polyline points="${pointsText}" fill="none" stroke="${colors[index]}" stroke-width="2"/><text x="${margin + index*145}" y="12" fill="${colors[index]}" font-size="9">${escapeHtml(key)}</text>`;
  }).join("");
  return `<div class="curve"><svg viewBox="0 0 ${width} ${height}">${paths}</svg></div>`;
}

async function openRun(runId) {
  try {
    const run = await getJson(`/api/runs/${encodeURIComponent(runId)}`);
    const metrics = run.metrics.map((metric) => `
      <div class="metric-card ${metric.is_primary ? "primary" : ""}">
        <span>${escapeHtml(metric.split)} · ${escapeHtml(metric.name)}</span>
        <strong>${escapeHtml(metricText(metric))}</strong>
      </div>`).join("");
    const artifacts = run.artifacts.map((artifact) => `
      <a class="artifact-link" href="/api/artifacts/${encodeURIComponent(run.run_id)}/${artifact.path.split("/").map(encodeURIComponent).join("/")}" target="_blank" rel="noreferrer">
        <span>${escapeHtml(artifact.title ?? artifact.path)}</span><span class="muted">${escapeHtml(artifact.kind)} ↗</span>
      </a>`).join("");
    $("#drawer-content").innerHTML = `
      <div class="detail-eyebrow">${escapeHtml(run.task_id)} / ${escapeHtml(run.protocol_id)}</div>
      <h2 id="drawer-title">${escapeHtml(run.run_id)}</h2>
      <div class="detail-subtitle"><span class="status status-${escapeHtml(run.status)}">${escapeHtml(run.status)}</span></div>
      <div class="detail-grid">
        <div class="detail-box"><span>Model</span><strong>${escapeHtml(run.model_id)}</strong></div>
        <div class="detail-box"><span>Seed</span><strong>${run.seed}</strong></div>
        <div class="detail-box"><span>Dataset</span><strong>${escapeHtml(run.dataset.dataset_id)} / ${escapeHtml(run.dataset.split)}</strong></div>
        <div class="detail-box"><span>Manifest</span><strong>${escapeHtml(run.dataset.manifest_sha256.slice(0, 16))}…</strong></div>
        <div class="detail-box"><span>Trainable parameters</span><strong>${fmtNumber.format(run.training.trainable_parameters)}</strong></div>
        <div class="detail-box"><span>Wall time</span><strong>${run.training.wall_time_seconds == null ? "—" : fmtNumber.format(run.training.wall_time_seconds / 3600) + " h"}</strong></div>
      </div>
      <h3>Metrics</h3><div class="metric-grid">${metrics || `<span class="muted">暂无指标</span>`}</div>
      <h3>Input bands · ${escapeHtml(run.input.sensor)} · T=${run.input.time_steps}</h3><div class="bands">${run.input.bands.map((band) => `<span class="band">${escapeHtml(band)}</span>`).join("")}</div>
      <h3>Curves</h3>${curveSvg(run.curves)}
      <h3>Artifacts</h3><div class="artifact-list">${artifacts || `<span class="muted">暂无可浏览产物</span>`}</div>
      ${run.notes ? `<h3>Notes</h3><p class="muted">${escapeHtml(run.notes)}</p>` : ""}`;
    $("#drawer").classList.add("open");
    $("#drawer").setAttribute("aria-hidden", "false");
  } catch (error) { setAlert(`无法载入运行 ${runId}: ${error.message}`); }
}

function closeDrawer() {
  $("#drawer").classList.remove("open");
  $("#drawer").setAttribute("aria-hidden", "true");
}

function formatCell(cell) {
  const digits = cell.metric === "mean_iou" ? 3 : 4;
  const mean = cell.mean.toFixed(digits);
  const spread = cell.sd === null ? "" : ` ± ${cell.sd.toFixed(digits)}`;
  return `${mean}${spread} <span class="muted">n=${cell.n}</span>`;
}

function cellTooltip(cell) {
  const lines = Object.entries(cell.values).map(([seed, value]) => `seed ${seed}: ${value.toFixed(4)}`);
  const extra = [];
  if (cell.learning_rate !== "-") extra.push(`lr ${cell.learning_rate}`);
  if (cell.input_px !== "-") extra.push(`输入 ${cell.input_px}px`);
  if (cell.variants > 1) extra.push(`同一单元格另有 ${cell.variants - 1} 组配置`);
  return [cell.protocol, ...extra, ...lines].join("\n");
}

function renderMatrix(data) {
  const head = $("#matrix-head");
  const body = $("#matrix-body");
  const empty = $("#matrix-empty");
  if (!data.cells.length) {
    head.innerHTML = "";
    body.innerHTML = "";
    empty.classList.remove("hidden");
    return;
  }
  empty.classList.add("hidden");
  const taskHeads = data.tasks.map((task) => `<th title="${escapeHtml(task.metric)}">${escapeHtml(task.id)}<small>${escapeHtml(task.name)}</small></th>`).join("");
  head.innerHTML = `<tr><th>轨道</th><th>模型</th>${taskHeads}</tr>`;
  const index = new Map(data.cells.map((cell) => [`${cell.track}|${cell.model}|${cell.task}`, cell]));
  body.innerHTML = data.rows.map((row, rowIndex) => {
    const track = data.tracks.find((item) => item.id === row.track);
    const sameTrack = data.rows.filter((item) => item.track === row.track);
    const first = data.rows.findIndex((item) => item.track === row.track) === rowIndex;
    const trackCell = first ? `<th rowspan="${sameTrack.length}" class="track-cell">${escapeHtml(row.track)}<small>${escapeHtml(track ? track.name : "")}</small></th>` : "";
    const cells = data.tasks.map((task) => {
      const cell = index.get(`${row.track}|${row.model}|${task.id}`);
      if (!cell) return `<td class="empty-cell">—</td>`;
      return `<td title="${escapeHtml(cellTooltip(cell))}">${formatCell(cell)}</td>`;
    }).join("");
    return `<tr>${trackCell}<th class="model-cell">${escapeHtml(row.name)}</th>${cells}</tr>`;
  }).join("");
}

async function load() {
  setAlert("");
  $("#reload").disabled = true;
  try {
    const [summary, runs, downloads, matrix] = await Promise.all([getJson("/api/summary"), getJson("/api/runs"), getJson("/api/downloads"), getJson("/api/matrix")]);
    state.summary = summary;
    state.runs = runs.items;
    state.downloads = downloads;
    renderMatrix(matrix);
    renderSummary();
    renderDownloads();
    applyFilters();
    $("#last-sync").textContent = `同步于 ${new Date().toLocaleTimeString("zh-CN")}`;
  } catch (error) {
    setAlert(`加载结果失败：${error.message}`);
  } finally {
    $("#reload").disabled = false;
    scheduleDownloadPoll();
  }
}

document.addEventListener("DOMContentLoaded", () => {
  ["#task-filter", "#model-filter", "#protocol-filter", "#status-filter"].forEach((selector) => $(selector).addEventListener("change", applyFilters));
  $("#reload").addEventListener("click", load);
  $("#resource-groups").addEventListener("click", (event) => {
    const button = event.target.closest(".resource-action");
    if (button) handleResourceAction(button);
  });
  document.querySelectorAll("[data-close]").forEach((element) => element.addEventListener("click", closeDrawer));
  document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeDrawer(); });
  load();
});
