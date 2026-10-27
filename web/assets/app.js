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
      <td><a class="run-id" href="#/runs/${encodeURIComponent(run.run_id)}" onclick="event.stopPropagation()">${escapeHtml(run.run_id)}</a></td>
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
  loadTaskCards();
  route();
});

// ---------------------------------------------------------------- task pages
// Hash routing keeps the static build free of a server-side catch-all:
//   #/            → matrix dashboard (existing)
//   #/task/<id>   → one task page
//   #/compare     → cross-task table
const taskState = { list: null, detail: null, compare: null };

function fmtScore(value, digits = 4) {
  return value == null || !Number.isFinite(value) ? "—" : value.toFixed(digits);
}

function route() {
  const hash = window.location.hash || "#/";
  const home = $("#view-home");
  const taskView = $("#view-task");
  const compareView = $("#view-compare");
  if (!home || !taskView || !compareView) return;
  const taskMatch = hash.match(/^#\/task\/([A-Za-z0-9]+)$/);
  const runMatch = hash.match(/^#\/runs\/(.+)$/);
  const runView = $("#view-run");
  home.classList.toggle("hidden", Boolean(taskMatch) || hash === "#/compare" || Boolean(runMatch));
  taskView.classList.toggle("hidden", !taskMatch);
  compareView.classList.toggle("hidden", hash !== "#/compare");
  if (runView) runView.classList.toggle("hidden", !runMatch);
  window.scrollTo({ top: 0 });
  if (taskMatch) renderTaskPage(taskMatch[1]);
  if (runMatch) renderRunPage(decodeURIComponent(runMatch[1]));
  if (hash === "#/compare") renderComparePage();
}

async function loadTaskCards() {
  const container = $("#task-cards");
  if (!container) return;
  try {
    const payload = await getJson("/api/tasks");
    taskState.list = payload;
    container.innerHTML = payload.items.map((item) => {
      const best = item.best
        ? `<span class="task-card-score">${escapeHtml(item.best.track)} · ${escapeHtml(item.best.model)} · ${fmtScore(item.best.mean)}</span>`
        : `<span class="task-card-score muted">暂无正式结果</span>`;
      const badges = [
        item.suite === "out_of_suite" ? `<span class="badge badge-warn">域外轨</span>` : "",
        item.has_notes
          ? (item.notes_age_days > 30 ? `<span class="badge badge-warn">结论 ${item.notes_age_days} 天未更新</span>` : "")
          : `<span class="badge">未写结论</span>`,
      ].join("");
      return `<a class="task-card" href="#/task/${encodeURIComponent(item.task_id)}">
        <div class="task-card-head"><strong>${escapeHtml(item.task_id)}</strong><span>${escapeHtml(item.name || "")}</span>${badges}</div>
        <div class="task-card-meta">${escapeHtml(item.sensor || "—")} · ${escapeHtml(item.dataset_id || "—")} · ${escapeHtml(item.metric || "—")}</div>
        <div class="task-card-meta">${item.models} 个模型 · ${item.cells} 个单元 · 最多 ${item.seeds_max} 种子</div>
        ${best}
      </a>`;
    }).join("");
  } catch (error) {
    container.innerHTML = `<p class="muted">任务列表加载失败：${escapeHtml(error.message)}</p>`;
  }
}

function renderWarnings(warnings) {
  if (!warnings || !warnings.length) return "";
  const items = warnings.map((warning) => `<li class="warning-${escapeHtml(warning.kind)}">${escapeHtml(warning.text)}</li>`).join("");
  return `<section class="panel"><div class="panel-title"><div><h2>注意事项</h2></div><span class="chip">${warnings.length}</span></div><ul class="task-warnings">${items}</ul></section>`;
}

function renderNotes(notes) {
  if (!notes) {
    return `<section class="panel"><div class="panel-title"><div><h2>结论</h2></div></div><p class="muted">尚未撰写结论（configs/tasks 中的 notes 为空）。</p></section>`;
  }
  const findings = (notes.findings || []).map((text) => `<li>${escapeHtml(text)}</li>`).join("");
  const caveats = (notes.caveats || []).map((text) => `<li>${escapeHtml(text)}</li>`).join("");
  return `<section class="panel">
      <div class="panel-title"><div><h2>结论</h2></div><span class="chip">截至 ${escapeHtml(notes.updated)}</span></div>
      <p class="task-summary">${escapeHtml(notes.summary)}</p>
      ${findings ? `<h3 class="task-subhead">关键发现</h3><ul class="task-findings">${findings}</ul>` : ""}
      ${caveats ? `<h3 class="task-subhead">已知局限</h3><ul class="task-caveats">${caveats}</ul>` : ""}
    </section>`;
}

function renderDataset(dataset, spec) {
  const manifest = (dataset && dataset.manifest) || [];
  const overlap = dataset ? dataset.overlap : null;
  const rows = manifest.map((entry) => {
    const counts = Object.entries(entry.counts || {}).map(([split, count]) => `${split} ${count}`).join(" · ");
    return `<tr><td><code>${escapeHtml(entry.sha256_prefix)}</code></td><td>${escapeHtml(counts)}</td></tr>`;
  }).join("");
  const overlapText = overlap
    ? `训练与评测共享 ${overlap.shared_groups}/${overlap.eval_groups} 个 group`
    : "未记录 group 重叠";
  return `<section class="panel">
      <div class="panel-title"><div><h2>数据</h2></div><span class="chip">${escapeHtml((spec && spec.sensor) || "—")}</span></div>
      <p class="panel-intro">${escapeHtml((spec && spec.dataset_id) || "—")} · ${spec ? spec.bands.length : 0} 波段 · ${spec ? spec.time_steps : "—"} 时相 · ${spec ? spec.num_outputs : "—"} 类 · 指标 <code>${escapeHtml((spec && spec.primary_metric && spec.primary_metric.name) || "—")}</code></p>
      <table class="mini"><thead><tr><th>manifest</th><th>划分样本数</th></tr></thead><tbody>${rows || `<tr><td colspan="2" class="muted">未找到 manifest</td></tr>`}</tbody></table>
      <p class="muted">${escapeHtml(overlapText)}</p>
    </section>`;
}

function renderCells(cells, tracks) {
  if (!cells.length) {
    return `<section class="panel"><div class="panel-title"><div><h2>各轨道成绩</h2></div></div><p class="muted">该任务暂无正式结果。</p></section>`;
  }
  const trackName = (id) => (tracks.find((track) => track.id === id) || {}).name || "";
  const sorted = [...cells].sort((a, b) => a.track.localeCompare(b.track) || b.mean - a.mean);
  const rows = sorted.map((cell) => `<tr>
      <td><strong>${escapeHtml(cell.track)}</strong><small class="muted"> ${escapeHtml(trackName(cell.track))}</small></td>
      <td>${escapeHtml(cell.model)}</td>
      <td>${cell.n}</td>
      <td>${fmtScore(cell.mean)} ± ${cell.sd == null ? "—" : fmtScore(cell.sd)}</td>
      <td>${Object.entries(cell.values || {}).map(([seed, value]) => `s${seed}=${fmtScore(value)}`).join("，")}</td>
      <td>${escapeHtml(cell.learning_rate || "—")}</td>
      <td>${escapeHtml(cell.head || "—")}</td>
    </tr>`).join("");
  return `<section class="panel">
      <div class="panel-title"><div><h2>各轨道成绩</h2></div><span class="muted">不同轨道不合并排名</span></div>
      <table class="mini"><thead><tr><th>轨道</th><th>模型</th><th>n</th><th>均值 ± 标准差</th><th>各种子</th><th>学习率</th><th>任务头</th></tr></thead><tbody>${rows}</tbody></table>
    </section>`;
}

function renderTaskRuns(runs) {
  const rows = runs.map((run) => `<tr class="${run.excluded ? "excluded-row" : ""}">
      <td><a href="#/runs/${encodeURIComponent(run.run_id)}"><code>${escapeHtml(run.run_id)}</code></a></td>
      <td>${escapeHtml(run.seed)}</td>
      <td>${escapeHtml(run.status)}${run.excluded ? " · 已排除" : ""}</td>
      <td>${run.primary_metric ? metricText(run.primary_metric) : "—"}</td>
    </tr>`).join("");
  return `<section class="panel">
      <div class="panel-title"><div><h2>最近运行</h2></div><span class="chip">${runs.length}</span></div>
      <table class="mini"><thead><tr><th>运行</th><th>种子</th><th>状态</th><th>主指标</th></tr></thead><tbody>${rows || `<tr><td colspan="4" class="muted">暂无运行</td></tr>`}</tbody></table>
    </section>`;
}

async function renderTaskPage(taskId) {
  const body = $("#task-body");
  if (!body) return;
  body.innerHTML = `<p class="muted">加载 ${escapeHtml(taskId)} …</p>`;
  if (!taskState.list) await loadTaskCards();
  const tracks = (taskState.list && taskState.list.tracks) || [];
  try {
    const detail = await getJson(`/api/tasks/${encodeURIComponent(taskId)}`);
    taskState.detail = detail;
    const spec = detail.spec || {};
    const badges = [
      detail.warnings.some((warning) => warning.kind === "out_of_suite") ? `<span class="badge badge-warn">域外轨</span>` : "",
      detail.notes
        ? `<span class="badge${detail.notes_age_days > 30 ? " badge-warn" : ""}">结论截至 ${escapeHtml(detail.notes.updated)}${detail.notes_age_days > 30 ? `（${detail.notes_age_days} 天前）` : ""}</span>`
        : `<span class="badge">未写结论</span>`,
    ].join("");
    body.innerHTML = `<header class="task-header">
        <div>
          <div class="eyebrow">TASK ${escapeHtml(detail.task_id)}</div>
          <h1>${escapeHtml(detail.name)}</h1>
          <p class="muted">${escapeHtml(spec.dataset_id || "—")} · ${escapeHtml(spec.sensor || "—")} · ${(spec.bands || []).length} 波段 · 指标 <code>${escapeHtml((spec.primary_metric && spec.primary_metric.name) || "—")}</code> ${badges}</p>
        </div>
      </header>
      ${renderNotes(detail.notes)}
      ${renderWarnings(detail.warnings)}
      ${renderDataset(detail.dataset, spec)}
      ${renderCells(detail.cells || [], tracks)}
      ${detail.cells && detail.cells.length ? `<section class="panel"><div class="panel-title"><div><h2>种子分布</h2></div><span class="muted">每个点是一次种子运行</span></div>${renderSeedScatter(detail.cells)}</section>` : ""}
      ${renderTaskRuns(detail.runs || [])}`;
  } catch (error) {
    body.innerHTML = `<p class="muted">加载失败：${escapeHtml(error.message)}</p>`;
  }
}

async function renderComparePage() {
  const body = $("#compare-body");
  if (!body) return;
  body.innerHTML = `<p class="muted">加载跨任务对比 …</p>`;
  try {
    const payload = taskState.compare || (await getJson("/api/compare"));
    taskState.compare = payload;
    const tracks = payload.tracks.filter((track) => payload.tasks.some((task) => task.tracks[track.id]));
    const head = `<tr><th>任务</th><th>指标</th>${tracks.map((track) => `<th title="${escapeHtml(track.name)}">${escapeHtml(track.id)}</th>`).join("")}</tr>`;
    const rows = payload.tasks.map((task) => {
      const cells = tracks.map((track) => {
        const entry = task.tracks[track.id];
        if (!entry) return `<td class="empty-cell">—</td>`;
        return `<td>${fmtScore(entry.mean)}<small class="muted"><br>${escapeHtml(entry.model)} · n=${entry.n}</small></td>`;
      }).join("");
      const suite = task.suite === "out_of_suite" ? ` <span class="badge badge-warn">域外</span>` : "";
      return `<tr><td><a href="#/task/${encodeURIComponent(task.task_id)}">${escapeHtml(task.task_id)}</a> ${escapeHtml(task.name || "")}${suite}</td><td><code>${escapeHtml(task.metric || "—")}</code></td>${cells}</tr>`;
    }).join("");
    body.innerHTML = `<header class="task-header"><div><div class="eyebrow">CROSS-TASK</div><h1>跨任务对比</h1><p class="muted">每格是该任务该轨道的最强单元（按均值）。不同任务的指标不可直接比大小，只用于看覆盖与一致性。</p></div></header>
      <section class="panel"><div class="table-wrap"><table class="matrix"><thead>${head}</thead><tbody>${rows}</tbody></table></div></section>`;
  } catch (error) {
    body.innerHTML = `<p class="muted">加载失败：${escapeHtml(error.message)}</p>`;
  }
}

window.addEventListener("hashchange", route);

// ------------------------------------------------------------- run detail
function svgLine(points, { width = 520, height = 120, stroke = "#6aa9ff" } = {}) {
  const values = points.map((point) => point.value).filter((value) => Number.isFinite(value));
  if (!values.length) return `<p class="muted">无曲线数据</p>`;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const stepX = points.length > 1 ? width / (points.length - 1) : width;
  const coords = points.map((point, index) => {
    const x = index * stepX;
    const y = height - ((point.value - min) / span) * (height - 8) - 4;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  return `<svg class="spark" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img">
      <polyline fill="none" stroke="${stroke}" stroke-width="1.6" points="${coords}" />
    </svg>
    <div class="muted spark-range">${fmtScore(min, 4)} → ${fmtScore(max, 4)}</div>`;
}

function renderSeedScatter(cells) {
  if (!cells.length) return "";
  const sorted = [...cells].sort((a, b) => b.mean - a.mean);
  const all = sorted.flatMap((cell) => Object.values(cell.values || {}).filter((v) => Number.isFinite(v)));
  if (!all.length) return "";
  const min = Math.min(...all);
  const max = Math.max(...all);
  const span = max - min || 1;
  const width = 640;
  const rowHeight = 22;
  const height = sorted.length * rowHeight + 16;
  const x = (value) => 8 + ((value - min) / span) * (width - 16);
  const rows = sorted.map((cell, index) => {
    const y = 8 + index * rowHeight;
    const seeds = Object.entries(cell.values || {})
      .filter(([, value]) => Number.isFinite(value))
      .map(([seed, value]) => `<circle cx="${x(value).toFixed(1)}" cy="${y}" r="3.4" fill="#6aa9ff"><title>s${escapeHtml(seed)} = ${fmtScore(value)}</title></circle>`)
      .join("");
    const mean = `<line x1="${x(cell.mean).toFixed(1)}" y1="${y - 6}" x2="${x(cell.mean).toFixed(1)}" y2="${y + 6}" stroke="#ffb870" stroke-width="2" />`;
    const label = `${cell.track} · ${cell.model}`;
    return `${seeds}${mean}<text x="4" y="${y + 4}" fill="#9aa4b2" font-size="10">${escapeHtml(label.slice(0, 0))}</text>`;
  }).join("");
  const labels = sorted.map((cell, index) => `<div class="scatter-label">${escapeHtml(cell.track)} · ${escapeHtml(cell.model)} · n=${cell.n}</div>`).join("");
  return `<div class="scatter">
      <div class="scatter-plot"><svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img">${rows}</svg></div>
      <div class="scatter-labels">${labels}</div>
      <p class="muted">蓝点 = 单个种子，橙色竖线 = 均值；横轴为指标区间 ${fmtScore(min)}–${fmtScore(max)}。</p>
    </div>`;
}

function renderRunPage(runId) {
  const body = $("#run-body");
  if (!body) return;
  body.innerHTML = `<p class="muted">加载 ${escapeHtml(runId)} …</p>`;
  getJson(`/api/runs/${encodeURIComponent(runId)}`)
    .then((run) => {
      const metrics = (run.metrics || []).map((metric) => `<tr class="${metric.is_primary ? "primary-row" : ""}">
          <td>${escapeHtml(metric.name)}${metric.class_name ? `<small class="muted"> / ${escapeHtml(metric.class_name)}</small>` : ""}</td>
          <td>${escapeHtml(metric.split)}</td>
          <td>${metric.value == null ? "—" : fmtScore(metric.value)}</td>
          <td>${metric.is_primary ? "主指标" : ""}</td>
        </tr>`).join("");
      const curves = {};
      (run.curves || []).forEach((point) => {
        const key = `${point.name}/${point.split}`;
        (curves[key] = curves[key] || []).push({ step: point.step, value: point.value });
      });
      const curveBlocks = Object.entries(curves).map(([key, points]) => `<div class="curve"><div class="curve-title">${escapeHtml(key)}</div>${svgLine(points)}</div>`).join("");
      const tags = Object.entries(run.tags || {}).map(([key, value]) => `<tr><td><code>${escapeHtml(key)}</code></td><td>${escapeHtml(value)}</td></tr>`).join("");
      const artifacts = (run.artifacts || []).map((artifact) => `<li><a href="/api/artifacts/${encodeURIComponent(runId)}/${encodeURIComponent(artifact.path)}">${escapeHtml(artifact.path)}</a> <span class="muted">${escapeHtml(artifact.title || artifact.kind)}</span></li>`).join("");
      const training = run.training || {};
      const input = run.input || {};
      body.innerHTML = `<header class="task-header">
          <div>
            <div class="eyebrow">${escapeHtml(run.task_id)} · ${escapeHtml(run.protocol_id)} · seed ${escapeHtml(run.seed)}</div>
            <h1>${escapeHtml(run.model_id)}</h1>
            <p class="muted"><code>${escapeHtml(run.run_id)}</code> · ${escapeHtml(run.status)}</p>
          </div>
        </header>
        <section class="panel"><div class="panel-title"><div><h2>指标</h2></div><span class="chip">${(run.metrics || []).length}</span></div>
          <table class="mini"><thead><tr><th>指标</th><th>split</th><th>值</th><th></th></tr></thead><tbody>${metrics || `<tr><td colspan="4" class="muted">无指标</td></tr>`}</tbody></table>
        </section>
        <section class="panel"><div class="panel-title"><div><h2>训练</h2></div></div>
          <table class="mini"><tbody>
            <tr><th>epochs</th><td>${escapeHtml(training.epochs_completed ?? "—")}</td><th>耗时</th><td>${formatDuration(training.wall_time_seconds)}</td></tr>
            <tr><th>可训练参数</th><td>${escapeHtml(training.trainable_parameters ?? "—")}</td><th>总参数</th><td>${escapeHtml(training.total_parameters ?? "—")}</td></tr>
            <tr><th>任务头</th><td>${escapeHtml(training.head_id ?? "—")}</td><th>峰值显存</th><td>${training.peak_gpu_memory_mb ? `${Math.round(training.peak_gpu_memory_mb)} MB` : "—"}</td></tr>
          </tbody></table>
        </section>
        <section class="panel"><div class="panel-title"><div><h2>输入与数据</h2></div></div>
          <table class="mini"><tbody>
            <tr><th>传感器</th><td>${escapeHtml(input.sensor ?? "—")}</td><th>预处理</th><td>${escapeHtml(input.preprocessing_id ?? "—")}</td></tr>
            <tr><th>波段</th><td colspan="3">${escapeHtml((input.bands || []).join(", ") || "—")}</td></tr>
            <tr><th>数据集</th><td>${escapeHtml((run.dataset || {}).dataset_id ?? "—")}</td><th>manifest</th><td><code>${escapeHtml(((run.dataset || {}).manifest_sha256 || "").slice(0, 16) || "—")}</code></td></tr>
          </tbody></table>
        </section>
        ${curveBlocks ? `<section class="panel"><div class="panel-title"><div><h2>曲线</h2></div></div><div class="curves">${curveBlocks}</div></section>` : ""}
        <section class="panel"><div class="panel-title"><div><h2>标签</h2></div></div>
          <table class="mini"><tbody>${tags || `<tr><td class="muted">无</td></tr>`}</tbody></table>
        </section>
        ${artifacts ? `<section class="panel"><div class="panel-title"><div><h2>产物</h2></div></div><ul>${artifacts}</ul></section>` : ""}`;
    })
    .catch((error) => { body.innerHTML = `<p class="muted">加载失败：${escapeHtml(error.message)}</p>`; });
}
