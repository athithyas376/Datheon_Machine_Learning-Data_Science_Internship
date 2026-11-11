/* ChurnIQ front-end: shared helpers + per-page logic (selected via <body data-page>) */
"use strict";

/* ===================== Helpers ===================== */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = v => String(v ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const nf = n => Number(n).toLocaleString("en-US");
const money = (n, d = 0) => "$" + Number(n).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const COLORS = { blue: "#3b82f6", cyan: "#22d3ee", violet: "#8b5cf6", green: "#10b981", amber: "#f59e0b", red: "#f43f5e", muted: "#8d9bc4" };
const RISK_COLOR = { Low: COLORS.green, Medium: COLORS.amber, High: COLORS.red };

async function api(url, opts = {}) {
  let res;
  try { res = await fetch(url, opts); }
  catch (e) { throw new Error("Cannot reach the server. Is `python app.py` still running?"); }
  let data = null;
  try { data = await res.json(); } catch (e) { /* non-JSON */ }
  if (!res.ok) throw new Error((data && data.error) || `Request failed (${res.status})`);
  return data;
}

function toast(msg, type = "info") {
  const icons = { success: "check-circle-fill", error: "x-octagon-fill", info: "info-circle-fill" };
  const el = document.createElement("div");
  el.className = `toast-msg ${type}`;
  el.innerHTML = `<i class="bi bi-${icons[type]}"></i><div>${esc(msg)}</div>`;
  $("#toasts").appendChild(el);
  setTimeout(() => { el.style.transition = "all .3s"; el.style.opacity = 0; el.style.transform = "translateX(30px)"; setTimeout(() => el.remove(), 300); }, 4200);
}

function busy(show, text = "Working…") {
  $("#busyText").textContent = text;
  $("#busyOverlay").classList.toggle("show", show);
}

function setBtnLoading(btn, loading, html) {
  if (loading) { btn.dataset.html = btn.innerHTML; btn.disabled = true; btn.innerHTML = `<span class="spinner-border me-2"></span>${html || "Loading…"}`; }
  else { btn.disabled = false; if (btn.dataset.html) btn.innerHTML = btn.dataset.html; }
}

function countUp(el, target, { decimals = 0, prefix = "", suffix = "" } = {}) {
  const dur = 900, t0 = performance.now();
  const step = t => {
    const p = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - p, 3);
    el.textContent = prefix + (target * e).toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals }) + suffix;
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

const BASE_LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
  font: { family: "Inter, sans-serif", color: "#c9d5ff", size: 12 },
  margin: { l: 48, r: 18, t: 16, b: 64 },
  xaxis: { gridcolor: "rgba(148,176,255,0.08)", zerolinecolor: "rgba(148,176,255,0.15)", automargin: true },
  yaxis: { gridcolor: "rgba(148,176,255,0.08)", zerolinecolor: "rgba(148,176,255,0.15)", automargin: true },
  legend: { orientation: "h", y: -0.32, x: 0.5, xanchor: "center", font: { size: 11 } },
  hoverlabel: { bgcolor: "#0f1a3a", bordercolor: "#3b82f6", font: { color: "#fff" } },
};
function layout(extra = {}) {
  const l = JSON.parse(JSON.stringify(BASE_LAYOUT));
  for (const k of Object.keys(extra)) {
    l[k] = (typeof extra[k] === "object" && !Array.isArray(extra[k]) && l[k]) ? { ...l[k], ...extra[k] } : extra[k];
  }
  return l;
}
function plot(id, data, lay = {}) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.remove("chart-loading");
  return Plotly.newPlot(el, data, layout(lay), { displayModeBar: false, responsive: true });
}
function chartError(id, msg) {
  const el = document.getElementById(id);
  if (el) { el.classList.remove("chart-loading"); el.innerHTML = `<div class="text-muted text-center pt-5"><i class="bi bi-exclamation-circle"></i> ${esc(msg)}</div>`; }
}

function badge(text, kind) { return `<span class="badge-soft ${String(kind || text).toLowerCase()}">${esc(text)}</span>`; }
function probBar(pct) {
  const c = pct >= 60 ? COLORS.red : pct >= 30 ? COLORS.amber : COLORS.green;
  return `<span class="prob-bar"><span style="width:${Math.min(100, pct)}%;background:${c}"></span></span>${pct.toFixed(1)}%`;
}

function renderPager(el, page, pages, go) {
  const items = [];
  const add = (label, p, disabled, active) =>
    items.push(`<li class="page-item ${disabled ? "disabled" : ""} ${active ? "active" : ""}"><a class="page-link" href="#" data-p="${p}">${label}</a></li>`);
  add("&laquo;", page - 1, page <= 1);
  const win = new Set([1, pages, page - 1, page, page + 1]);
  let last = 0;
  for (let p = 1; p <= pages; p++) {
    if (!win.has(p)) continue;
    if (p - last > 1) items.push(`<li class="page-item disabled"><span class="page-link">…</span></li>`);
    add(p, p, false, p === page); last = p;
  }
  add("&raquo;", page + 1, page >= pages);
  el.innerHTML = items.join("");
  $$("a", el).forEach(a => a.addEventListener("click", e => {
    e.preventDefault();
    const li = a.parentElement;
    if (li.classList.contains("disabled") || li.classList.contains("active")) return;
    go(parseInt(a.dataset.p, 10));
  }));
}

/* ===================== Shell ===================== */
function initShell() {
  const sb = $("#sidebar"), bd = $("#sideBackdrop");
  const toggle = open => { sb.classList.toggle("open", open); bd.classList.toggle("show", open); };
  $("#menuBtn").addEventListener("click", () => toggle(!sb.classList.contains("open")));
  bd.addEventListener("click", () => toggle(false));
  $$(".side-link").forEach(a => a.addEventListener("click", () => toggle(false)));
  api("/api/model-metrics").then(m => {
    $("#sideModel").textContent = `${m.best_model} · AUC ${m.models[m.best_model].test.roc_auc.toFixed(3)}`;
    $("#pillText").textContent = `${m.best_model} active`;
  }).catch(() => { $("#sideModel").textContent = "Not trained"; $("#pillText").textContent = "Model missing"; });
  window.addEventListener("load", () => setTimeout(() => $("#pageLoader").classList.add("done"), 250));
  setTimeout(() => $("#pageLoader").classList.add("done"), 4000); // safety net
  // Plotly charts inside hidden tabs need a resize once shown
  $$('[data-bs-toggle="tab"]').forEach(t => t.addEventListener("shown.bs.tab", () => {
    $$(".js-plotly-plot").forEach(p => Plotly.Plots.resize(p));
  }));
}

/* ===================== Overview ===================== */
async function pageOverview() {
  let s;
  try { s = await api("/api/stats"); }
  catch (e) { toast(e.message, "error"); ["chChurn", "chCharges", "chContract", "chRisk"].forEach(i => chartError(i, e.message)); return; }
  countUp($("#kpiTotal"), s.total_customers);
  countUp($("#kpiChurned"), s.churned);
  countUp($("#kpiRetained"), s.retained);
  countUp($("#kpiRate"), s.churn_rate, { decimals: 1, suffix: "%" });
  countUp($("#kpiAvg"), s.avg_monthly_charges, { decimals: 2, prefix: "$" });
  countUp($("#kpiRisk"), s.revenue_at_risk, { prefix: "$" });

  plot("chChurn", [{
    type: "pie", hole: 0.62, labels: ["Retained", "Churned"], values: [s.retained, s.churned],
    marker: { colors: [COLORS.blue, COLORS.red], line: { color: "#0a1128", width: 3 } },
    textinfo: "percent", hovertemplate: "%{label}: %{value:,} (%{percent})<extra></extra>", sort: false,
  }], {
    margin: { l: 10, r: 10, t: 10, b: 10 },
    annotations: [{ text: `<b>${s.churn_rate.toFixed(1)}%</b><br><span style="font-size:11px">churn rate</span>`, showarrow: false, font: { size: 24, color: "#fff" } }],
  });

  const h = s.charges_hist;
  plot("chCharges", [
    { type: "bar", x: h.centers, y: h.retained, name: "Retained", marker: { color: COLORS.blue }, hovertemplate: "$%{x}: %{y} customers<extra>Retained</extra>" },
    { type: "bar", x: h.centers, y: h.churned, name: "Churned", marker: { color: COLORS.red }, hovertemplate: "$%{x}: %{y} customers<extra>Churned</extra>" },
  ], { barmode: "stack", xaxis: { title: "Monthly charges ($)" }, yaxis: { title: "Customers" } });

  const c = s.contract;
  plot("chContract", [
    { type: "bar", x: c.labels, y: c.retained, name: "Retained", marker: { color: COLORS.blue } },
    { type: "bar", x: c.labels, y: c.churned, name: "Churned", marker: { color: COLORS.red } },
    { type: "scatter", mode: "lines+markers+text", x: c.labels, y: c.churn_rate, yaxis: "y2", name: "Churn rate %", line: { color: COLORS.amber, width: 3 }, marker: { size: 9 }, text: c.churn_rate.map(v => v + "%"), textposition: "top center", textfont: { color: COLORS.amber } },
  ], { barmode: "stack", yaxis: { title: "Customers" }, yaxis2: { overlaying: "y", side: "right", showgrid: false, range: [0, Math.max(...c.churn_rate) * 1.4], ticksuffix: "%" }, margin: { r: 50 } });

  const r = s.risk;
  plot("chRisk", [
    { type: "bar", x: r.levels, y: r.all, name: "All customers", marker: { color: r.levels.map(l => RISK_COLOR[l]), opacity: 0.55 }, text: r.all.map(nf), textposition: "outside" },
    { type: "bar", x: r.levels, y: r.retained, name: "Still active", marker: { color: r.levels.map(l => RISK_COLOR[l]) }, text: r.retained.map(nf), textposition: "outside" },
  ], { barmode: "group", yaxis: { title: "Customers" }, xaxis: { title: "Predicted risk level" }, uniformtext: { minsize: 10 } });
  $("#riskSub").textContent = `Low <${r.bands.low_max * 100}% · High ≥${r.bands.high_min * 100}%`;
  $("#riskBadges").innerHTML = r.levels.map((l, i) =>
    `<span class="badge-soft ${l.toLowerCase()}">${l}: ${r.actual_churn_rate[i]}% actually churned</span>`).join("") +
    `<span class="badge-soft medium">${nf(s.high_risk_retained)} active high-risk customers</span>`;
}

/* ===================== Analytics ===================== */
function stackedBar(id, g, { rate = true, horizontal = false } = {}) {
  const ax = horizontal ? "y" : "x", vx = horizontal ? "x" : "y";
  const traces = [
    { type: "bar", name: "Retained", [ax]: g.labels, [vx]: g.retained, marker: { color: COLORS.blue }, orientation: horizontal ? "h" : "v" },
    { type: "bar", name: "Churned", [ax]: g.labels, [vx]: g.churned, marker: { color: COLORS.red }, orientation: horizontal ? "h" : "v" },
  ];
  const lay = { barmode: "stack", yaxis: {}, xaxis: {} };
  if (rate && !horizontal) {
    traces.push({ type: "scatter", mode: "lines+markers", x: g.labels, y: g.churn_rate, yaxis: "y2", name: "Churn rate %", line: { color: COLORS.amber, width: 3 }, marker: { size: 8 }, hovertemplate: "%{y}%<extra>Churn rate</extra>" });
    lay.yaxis2 = { overlaying: "y", side: "right", showgrid: false, ticksuffix: "%", range: [0, Math.max(...g.churn_rate) * 1.35] };
    lay.margin = { r: 50 };
  }
  plot(id, traces, lay);
}
function rateBar(id, g, title = "Churn rate (%)") {
  const colors = g.churn_rate.map(v => v >= 35 ? COLORS.red : v >= 20 ? COLORS.amber : COLORS.green);
  plot(id, [{
    type: "bar", x: g.labels, y: g.churn_rate, marker: { color: colors }, text: g.churn_rate.map(v => v + "%"), textposition: "outside",
    customdata: g.total, hovertemplate: "%{x}<br>Churn rate: %{y}%<br>Customers: %{customdata:,}<extra></extra>",
  }], { yaxis: { title, ticksuffix: "%", range: [0, Math.max(...g.churn_rate) * 1.25] } });
}
function donutRate(id, g) {
  plot(id, [{
    type: "pie", hole: 0.55, labels: g.labels, values: g.total, sort: false,
    marker: { colors: [COLORS.blue, COLORS.violet, COLORS.cyan], line: { color: "#0a1128", width: 3 } },
    customdata: g.churn_rate, textinfo: "label+percent", textposition: "inside",
    hovertemplate: "%{label}<br>%{value:,} customers<br>Churn rate %{customdata}%<extra></extra>",
  }], { margin: { l: 8, r: 8, t: 8, b: 8 }, showlegend: false });
}

async function pageAnalytics() {
  let a;
  try { a = await api("/api/analytics"); }
  catch (e) { toast(e.message, "error"); $$(".chart").forEach(c => chartError(c.id, e.message)); return; }
  donutRate("chGender", a.gender); donutRate("chSenior", a.senior);
  donutRate("chPartner", a.partner); donutRate("chDependents", a.dependents);
  stackedBar("chInternet", a.internet); stackedBar("chContract", a.contract);
  rateBar("chTech", a.tech_support); rateBar("chPayment", a.payment);
  plot("chTenure", [
    { type: "histogram", x: a.tenure_values.retained, name: "Retained", marker: { color: COLORS.blue }, opacity: 0.75, nbinsx: 24 },
    { type: "histogram", x: a.tenure_values.churned, name: "Churned", marker: { color: COLORS.red }, opacity: 0.75, nbinsx: 24 },
  ], { barmode: "overlay", xaxis: { title: "Tenure (months)" }, yaxis: { title: "Customers" } });
  rateBar("chTenureBand", a.tenure); rateBar("chChargeBand", a.charges);

  const pick = (g, l) => g.churn_rate[g.labels.indexOf(l)];
  const items = [
    ["Seniors churn at " + pick(a.senior, "Yes") + "% vs " + pick(a.senior, "No") + "% for non-seniors."],
    ["Customers without a partner churn at " + pick(a.partner, "No") + "% vs " + pick(a.partner, "Yes") + "% with one."],
    ["Customers without dependents churn at " + pick(a.dependents, "No") + "% vs " + pick(a.dependents, "Yes") + "% with dependents."],
    ["Gender has little effect: Female " + pick(a.gender, "Female") + "% vs Male " + pick(a.gender, "Male") + "%."],
  ];
  $("#demoInsights").innerHTML = items.map(i => `<div class="insight"><i class="bi bi-lightbulb-fill"></i><div>${esc(i[0])}</div></div>`).join("");
}

/* ===================== Single prediction ===================== */
const SAMPLES = {
  high: { tenure: 3, MonthlyCharges: 95.5, TotalCharges: 286.5, Contract: "Month-to-month", PaymentMethod: "Electronic check", PaperlessBilling: "Yes", InternetService: "Fiber optic", PhoneService: "Yes", MultipleLines: "No", OnlineSecurity: "No", OnlineBackup: "No", DeviceProtection: "No", TechSupport: "No", StreamingTV: "Yes", StreamingMovies: "Yes", gender: "Female", SeniorCitizen: "0", Partner: "No", Dependents: "No" },
  low: { tenure: 60, MonthlyCharges: 58.2, TotalCharges: 3492, Contract: "Two year", PaymentMethod: "Credit card (automatic)", PaperlessBilling: "No", InternetService: "DSL", PhoneService: "Yes", MultipleLines: "Yes", OnlineSecurity: "Yes", OnlineBackup: "Yes", DeviceProtection: "Yes", TechSupport: "Yes", StreamingTV: "No", StreamingMovies: "No", gender: "Male", SeniorCitizen: "0", Partner: "Yes", Dependents: "Yes" },
};

function pagePrediction() {
  const form = $("#predForm"), total = $("#f_TotalCharges");
  let totalTouched = false;
  total.addEventListener("input", () => { totalTouched = true; });
  const autoTotal = () => { if (!totalTouched) total.value = (+$("#f_tenure").value * +$("#f_MonthlyCharges").value).toFixed(2); };
  $("#f_tenure").addEventListener("input", autoTotal);
  $("#f_MonthlyCharges").addEventListener("input", autoTotal);

  const fill = s => { Object.entries(s).forEach(([k, v]) => { const el = $("#f_" + k); if (el) el.value = v; }); totalTouched = true; };
  $("#sampleHigh").addEventListener("click", () => { fill(SAMPLES.high); toast("High-risk example loaded", "info"); });
  $("#sampleLow").addEventListener("click", () => { fill(SAMPLES.low); toast("Loyal-customer example loaded", "info"); });

  form.addEventListener("submit", async ev => {
    ev.preventDefault();
    const errBox = $("#predError"); errBox.classList.add("d-none");
    const payload = {};
    new FormData(form).forEach((v, k) => payload[k] = v);
    for (const k of ["tenure", "MonthlyCharges", "TotalCharges"]) {
      if (payload[k] === "" || isNaN(+payload[k]) || +payload[k] < 0) {
        errBox.textContent = `Please enter a valid non-negative number for ${k}.`; errBox.classList.remove("d-none"); return;
      }
    }
    const btn = $("#predBtn"); setBtnLoading(btn, true, "Analysing customer…");
    try {
      const r = await api("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      renderResult(r);
      if (window.innerWidth < 1200) $("#resultCard").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (e) { errBox.textContent = e.message; errBox.classList.remove("d-none"); toast(e.message, "error"); }
    finally { setBtnLoading(btn, false); }
  });
}

function renderResult(r) {
  const lvl = r.risk_level.toLowerCase(), color = RISK_COLOR[r.risk_level];
  const card = $("#resultCard");
  card.className = "glass result-card";
  card.innerHTML = `
    <div class="glow" style="background:${color}"></div>
    <div class="position-relative">
      <div class="kpi-label mb-1">Churn probability</div>
      <div id="gauge" style="height:230px"></div>
      <span class="risk-chip ${lvl}"><i class="bi bi-${lvl === "high" ? "exclamation-octagon-fill" : lvl === "medium" ? "exclamation-triangle-fill" : "shield-fill-check"}"></i>${esc(r.risk_level.toUpperCase())} RISK</span>
      <p class="text-muted small mt-3 mb-4">Model: <b>${esc(r.model)}</b> · decision threshold ${r.threshold}% ·
        predicted <b>${r.predicted_churn === "Yes" ? "to churn" : "to stay"}</b></p>
      <h6 class="text-start mb-3"><i class="bi bi-lightbulb-fill text-warning me-2"></i>Retention recommendations</h6>
      <ul class="rec-list">${r.recommendations.map((t, i) => `<li style="animation-delay:${i * 0.07}s"><i class="bi bi-check2-circle"></i><span>${esc(t)}</span></li>`).join("")}</ul>
    </div>`;
  Plotly.newPlot("gauge", [{
    type: "indicator", mode: "gauge+number", value: r.churn_probability, number: { suffix: "%", font: { size: 44, color: "#fff" }, valueformat: ".1f" },
    gauge: { axis: { range: [0, 100], tickcolor: "#8d9bc4" }, bar: { color: color, thickness: 0.3 }, bgcolor: "rgba(255,255,255,0.05)", borderwidth: 0,
      steps: [{ range: [0, 30], color: "rgba(16,185,129,0.22)" }, { range: [30, 60], color: "rgba(245,158,11,0.22)" }, { range: [60, 100], color: "rgba(244,63,94,0.22)" }] },
  }], layout({ margin: { l: 25, r: 25, t: 20, b: 0 } }), { displayModeBar: false, responsive: true });
}

/* ===================== Bulk prediction ===================== */
function pageBulk() {
  const dz = $("#dropzone"), input = $("#csvInput"), btn = $("#bulkBtn");
  let file = null, rows = [], page = 1; const PER = 15;
  const setFile = f => {
    $("#bulkError").classList.add("d-none"); $("#bulkWarn").classList.add("d-none");
    if (!f) return;
    if (!f.name.toLowerCase().endsWith(".csv")) { showErr("Please choose a .csv file."); return; }
    file = f; $("#fileName").innerHTML = `<span class="fname"><i class="bi bi-file-earmark-check"></i> ${esc(f.name)}</span> · ${(f.size / 1024).toFixed(1)} KB`;
    btn.disabled = false;
  };
  const showErr = m => { const b = $("#bulkError"); b.textContent = m; b.classList.remove("d-none"); };
  dz.addEventListener("click", () => input.click());
  input.addEventListener("change", () => setFile(input.files[0]));
  ["dragenter", "dragover"].forEach(e => dz.addEventListener(e, ev => { ev.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach(e => dz.addEventListener(e, ev => { ev.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", ev => setFile(ev.dataTransfer.files[0]));

  const draw = () => {
    const pages = Math.max(1, Math.ceil(rows.length / PER));
    page = Math.min(page, pages);
    $("#bulkBody").innerHTML = rows.slice((page - 1) * PER, page * PER).map(r => `<tr>
      <td><b>${esc(r.id)}</b></td><td>${esc(r.tenure)}</td><td>${esc(r.contract)}</td><td>${money(+r.monthly || 0, 2)}</td>
      <td>${probBar(r.probability)}</td><td>${badge(r.predicted)}</td><td>${badge(r.risk)}</td></tr>`).join("");
    renderPager($("#bulkPager"), page, pages, p => { page = p; draw(); });
  };

  btn.addEventListener("click", async () => {
    if (!file) return;
    $("#bulkError").classList.add("d-none"); $("#bulkWarn").classList.add("d-none");
    const fd = new FormData(); fd.append("file", file);
    setBtnLoading(btn, true, "Scoring customers…"); busy(true, "Running model predictions…");
    try {
      const r = await api("/api/bulk-predict", { method: "POST", body: fd });
      rows = r.rows; page = 1;
      countUp($("#bTotal"), r.summary.total); countUp($("#bHigh"), r.summary.high_risk);
      countUp($("#bMed"), r.summary.medium_risk); countUp($("#bLow"), r.summary.low_risk);
      $("#bNote").textContent = r.truncated ? `(top 1,000 by risk shown · full results in CSV)` : `(sorted by risk · avg ${r.summary.avg_probability}%)`;
      $("#downloadBtn").href = `/api/download/${r.file_id}`;
      $("#bulkResults").classList.remove("d-none"); draw();
      if (r.warnings.length) { const w = $("#bulkWarn"); w.innerHTML = r.warnings.map(esc).join("<br>"); w.classList.remove("d-none"); }
      toast(`Scored ${nf(r.summary.total)} customers · ${nf(r.summary.high_risk)} high-risk`, "success");
      $("#bulkResults").scrollIntoView({ behavior: "smooth" });
    } catch (e) { showErr(e.message); toast(e.message, "error"); }
    finally { setBtnLoading(btn, false); busy(false); }
  });
}

/* ===================== Model performance ===================== */
async function pageModel() {
  let m;
  try { m = await api("/api/model-metrics"); }
  catch (e) { toast(e.message, "error"); $$(".chart").forEach(c => chartError(c.id, e.message)); return; }
  const names = Object.keys(m.models), best = m.best_model;
  const mc = { "Random Forest": COLORS.green, "XGBoost": COLORS.blue };

  $("#bestName").textContent = best;
  $("#bestNote").textContent = `CV ROC-AUC ${m.models[best].cv.roc_auc_mean.toFixed(4)} · test ROC-AUC ${m.models[best].test.roc_auc.toFixed(4)} · trained ${m.trained_at}`;
  $("#splitNote").textContent = `${nf(m.dataset.train_rows)} train / ${nf(m.dataset.test_rows)} test · seed ${m.seed}`;
  $("#imbNote").innerHTML = `<i class="bi bi-balance-scale"></i> Imbalance handling: ` + names.map(n => `${esc(n)} → <code>${esc(m.imbalance_strategy[n])}</code>`).join(" · ");

  plot("chAuc", [
    { type: "bar", name: "CV ROC-AUC", x: names, y: names.map(n => m.models[n].cv.roc_auc_mean), error_y: { type: "data", array: names.map(n => m.models[n].cv.roc_auc_std), color: "#fff" }, marker: { color: COLORS.violet }, text: names.map(n => m.models[n].cv.roc_auc_mean.toFixed(3)), textposition: "inside", insidetextanchor: "start" },
    { type: "bar", name: "Test ROC-AUC", x: names, y: names.map(n => m.models[n].test.roc_auc), marker: { color: COLORS.cyan }, text: names.map(n => m.models[n].test.roc_auc.toFixed(3)), textposition: "inside", insidetextanchor: "start" },
  ], { barmode: "group", yaxis: { range: [0.7, 0.9], title: "ROC-AUC" } });

  plot("chRoc", [
    ...names.map(n => ({ type: "scatter", mode: "lines", name: `${n} (AUC ${m.models[n].test.roc_auc.toFixed(3)})`, x: m.models[n].roc_curve.fpr, y: m.models[n].roc_curve.tpr, line: { color: mc[n], width: 3 }, fill: n === best ? "tozeroy" : "none", fillcolor: "rgba(59,130,246,0.08)" })),
    { type: "scatter", mode: "lines", name: "Random (0.5)", x: [0, 1], y: [0, 1], line: { color: COLORS.muted, dash: "dash" } },
  ], { xaxis: { title: "False positive rate", range: [0, 1] }, yaxis: { title: "True positive rate", range: [0, 1.02] } });

  $("#cmpBody").innerHTML = names.map(n => {
    const r = m.models[n], t = r.test;
    return `<tr class="${n === best ? "best-row" : ""}"><td><b>${esc(n)}</b> ${n === best ? '<span class="badge-soft low ms-1"><i class="bi bi-trophy-fill"></i> Best</span>' : ""}</td>
      <td>${r.cv.roc_auc_mean.toFixed(4)} ± ${r.cv.roc_auc_std.toFixed(4)}</td><td>${t.roc_auc.toFixed(4)}</td><td>${t.accuracy.toFixed(4)}</td>
      <td>${t.precision.toFixed(4)}</td><td>${t.recall.toFixed(4)}</td><td>${t.f1.toFixed(4)}</td><td>${(r.threshold * 100).toFixed(1)}%</td></tr>`;
  }).join("");

  const tabs = $("#modelTabs");
  tabs.innerHTML = names.map(n => `<li class="nav-item"><button class="nav-link ${n === best ? "active" : ""}" data-model="${esc(n)}" type="button">${esc(n)}</button></li>`).join("");
  const show = name => {
    const r = m.models[name], c = r.confusion_matrix, t = r.test;
    $("#cmSub").textContent = `${name} · threshold ${(r.threshold * 100).toFixed(1)}%`;
    const z = [[c.tn, c.fp], [c.fn, c.tp]], lab = [["True Negative", "False Positive"], ["False Negative", "True Positive"]];
    plot("chCm", [{
      type: "heatmap", z, x: ["Predicted: Stay", "Predicted: Churn"], y: ["Actual: Stay", "Actual: Churn"], colorscale: [[0, "#0f1a3a"], [1, "#3b82f6"]], showscale: false, ygap: 4, xgap: 4,
      text: z.map((row, i) => row.map((v, j) => `<b>${nf(v)}</b><br><span style="font-size:11px">${lab[i][j]}</span>`)), texttemplate: "%{text}", hovertemplate: "%{y} / %{x}: %{z}<extra></extra>", textfont: { color: "#fff", size: 16 },
    }], { yaxis: { autorange: "reversed" }, margin: { l: 90, t: 10, b: 50 } });

    const top = r.feature_importance.slice(0, 10).reverse();
    $("#fiSub").textContent = name;
    plot("chFi", [{ type: "bar", orientation: "h", x: top.map(f => f.importance), y: top.map(f => f.feature), marker: { color: top.map((_, i) => `rgba(59,130,246,${0.4 + 0.06 * i})`), line: { color: COLORS.cyan, width: 0 } }, hovertemplate: "%{y}: %{x:.4f}<extra></extra>" }],
      { margin: { l: 200 }, xaxis: { title: "Importance" } });

    plot("chPrf", [{ type: "bar", x: ["Precision", "Recall", "F1-score", "Accuracy"], y: [t.precision, t.recall, t.f1, t.accuracy], marker: { color: [COLORS.blue, COLORS.green, COLORS.violet, COLORS.cyan] }, text: [t.precision, t.recall, t.f1, t.accuracy].map(v => v.toFixed(3)), textposition: "outside" }], { yaxis: { range: [0, 1] } });

    const g = r.feature_importance_grouped.slice(0, 10).reverse();
    plot("chFg", [{ type: "bar", orientation: "h", x: g.map(f => f.importance), y: g.map(f => f.feature), marker: { color: COLORS.violet } }], { margin: { l: 130 } });
  };
  tabs.addEventListener("click", ev => {
    const b = ev.target.closest("[data-model]"); if (!b) return;
    $$(".nav-link", tabs).forEach(x => x.classList.toggle("active", x === b)); show(b.dataset.model);
  });
  show(best);

  $("#retrainBtn").addEventListener("click", async ev => {
    const btn = ev.currentTarget;
    if (!confirm("Retrain both models from the dataset? This takes about 20-40 seconds.")) return;
    setBtnLoading(btn, true, "Training…"); busy(true, "Training Random Forest & XGBoost…");
    try { await api("/api/retrain", { method: "POST" }); toast("Models retrained", "success"); setTimeout(() => location.reload(), 700); }
    catch (e) { toast(e.message, "error"); setBtnLoading(btn, false); busy(false); }
  });
}

/* ===================== Retention ===================== */
async function pageRetention() {
  let d;
  try { d = await api("/api/retention"); }
  catch (e) { toast(e.message, "error"); $$(".chart").forEach(c => chartError(c.id, e.message)); return; }
  countUp($("#rHigh"), d.high_risk_total); countUp($("#rHighRet"), d.high_risk_retained);
  countUp($("#rBase"), d.overall_churn_rate, { decimals: 1, suffix: "%" });

  const f = [...d.factors].reverse();
  plot("chFactors", [{ type: "bar", orientation: "h", x: f.map(i => i.importance), y: f.map(i => i.feature), marker: { color: f.map((_, i) => `rgba(244,63,94,${0.35 + 0.08 * i})`) }, hovertemplate: "%{y}: %{x:.3f}<extra></extra>" }], { margin: { l: 130 }, xaxis: { title: "Relative importance" } });

  ["contract", "charges", "tenure"].forEach(k => {
    const rows = d[k];
    const g = { labels: rows.map(r => r.label), churn_rate: rows.map(r => r.churn_rate), total: rows.map(r => r.customers) };
    rateBar("chRate_" + k, g, "");
    $("#sug_" + k).innerHTML = d.suggestions[k].map(s => `<div class="insight"><i class="bi bi-lightbulb-fill"></i><div>${esc(s)}</div></div>`).join("");
  });

  $("#hrBody").innerHTML = d.high_risk_customers.map(c => `<tr><td><b>${esc(c.id)}</b></td><td>${c.tenure} mo</td><td>${esc(c.contract)}</td><td>${money(c.monthly, 2)}</td><td>${probBar(c.probability)}</td><td style="white-space:normal;min-width:260px">${esc(c.action)}</td></tr>`).join("")
    || `<tr><td colspan="6" class="text-center text-muted py-4">No high-risk active customers.</td></tr>`;
}

/* ===================== Customers ===================== */
function pageCustomers() {
  const st = { page: 1, sort: "churn_probability", order: "desc" };
  const params = () => {
    const p = new URLSearchParams({ q: $("#q").value.trim(), churn: $("#fChurn").value, contract: $("#fContract").value, risk: $("#fRisk").value });
    return p;
  };
  async function load() {
    const p = params(); p.set("page", st.page); p.set("per_page", $("#perPage").value); p.set("sort", st.sort); p.set("order", st.order);
    $("#custBody").style.opacity = 0.45;
    try {
      const d = await api("/api/customers?" + p);
      st.page = d.page;
      $("#custCount").textContent = `${nf(d.total)} customers`;
      $("#custBody").innerHTML = d.rows.map(r => `<tr><td><b>${esc(r.customerID)}</b></td><td>${esc(r.gender)}</td><td>${esc(r.Senior)}</td><td>${r.tenure}</td><td>${esc(r.Contract)}</td><td>${esc(r.InternetService)}</td><td>${esc(r.PaymentMethod)}</td><td>${money(r.MonthlyCharges, 2)}</td><td>${money(r.TotalCharges, 2)}</td><td>${badge(r.ChurnLabel)}</td><td>${probBar(r.churn_probability * 100)}</td><td>${badge(r.risk_level)}</td></tr>`).join("")
        || `<tr><td colspan="12" class="text-center text-muted py-5">No customers match these filters.</td></tr>`;
      renderPager($("#custPager"), d.page, d.pages, pg => { st.page = pg; load(); });
      $("#exportBtn").href = "/api/customers/export?" + params();
    } catch (e) { toast(e.message, "error"); }
    $("#custBody").style.opacity = 1;
  }
  let t; $("#q").addEventListener("input", () => { clearTimeout(t); t = setTimeout(() => { st.page = 1; load(); }, 280); });
  ["#fChurn", "#fContract", "#fRisk", "#perPage"].forEach(s => $(s).addEventListener("change", () => { st.page = 1; load(); }));
  $("#resetBtn").addEventListener("click", () => { $("#q").value = ""; ["#fChurn", "#fContract", "#fRisk"].forEach(s => $(s).value = ""); st.page = 1; load(); });
  $$("th.sortable").forEach(th => th.addEventListener("click", () => {
    const k = th.dataset.sort; st.order = (st.sort === k && st.order === "desc") ? "asc" : "desc"; st.sort = k; st.page = 1;
    $$("th.sortable").forEach(x => x.innerHTML = x.innerHTML.replace(/ [▲▼]$/, ""));
    th.innerHTML += st.order === "asc" ? " ▲" : " ▼"; load();
  }));
  load();
}

/* ===================== About ===================== */
async function pageAbout() {
  try {
    const m = await api("/api/model-metrics"), d = m.dataset;
    $("#dsInfo").innerHTML = `<table class="table table-dark-glass mb-0"><tbody>
      <tr><td class="text-muted">Source</td><td>IBM Telco Customer Churn</td></tr>
      <tr><td class="text-muted">Customers</td><td>${nf(d.rows)}</td></tr>
      <tr><td class="text-muted">Features used</td><td>${d.features} (4 numeric, 15 categorical)</td></tr>
      <tr><td class="text-muted">Churn rate</td><td>${(d.churn_rate * 100).toFixed(2)}%</td></tr>
      <tr><td class="text-muted">Train / test</td><td>${nf(d.train_rows)} / ${nf(d.test_rows)}</td></tr>
      <tr><td class="text-muted">Best model</td><td><b>${esc(m.best_model)}</b> (test AUC ${m.models[m.best_model].test.roc_auc.toFixed(3)})</td></tr></tbody></table>`;
  } catch (e) { $("#dsInfo").innerHTML = `<span class="text-muted">${esc(e.message)}</span>`; }
}

/* ===================== Boot ===================== */
document.addEventListener("DOMContentLoaded", () => {
  initShell();
  const page = document.body.dataset.page;
  ({ overview: pageOverview, analytics: pageAnalytics, prediction: pagePrediction, bulk: pageBulk,
     model: pageModel, retention: pageRetention, customers: pageCustomers, about: pageAbout }[page] || (() => {}))();
});
