/* =========================================================
   SafeResponse Engine — app.js
   Vanilla JS, no build step, no external deps.
   ========================================================= */

/* ── Theme management ───────────────────────────────────── */
(function initTheme() {
  const stored = localStorage.getItem("sre_theme");
  const preferred = window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  const theme = stored || preferred;
  if (theme === "light") {
    document.documentElement.dataset.theme = "light";
  }
})();

function toggleTheme() {
  const root = document.documentElement;
  const next = root.dataset.theme === "light" ? "dark" : "light";
  if (next === "dark") {
    delete root.dataset.theme;
  } else {
    root.dataset.theme = "light";
  }
  localStorage.setItem("sre_theme", next);
  updateThemeIcons();
}

function updateThemeIcons() {
  const isDark = document.documentElement.dataset.theme !== "light";
  const icon = isDark ? "☀️" : "🌙";
  const label = isDark ? "Switch to light mode" : "Switch to dark mode";
  document.querySelectorAll(".theme-toggle-icon, .header-theme-icon").forEach((el) => {
    el.textContent = icon;
  });
  document.querySelectorAll(".header-theme-btn, .theme-toggle").forEach((el) => {
    el.setAttribute("aria-label", label);
    el.title = label;
  });
}

/* ── DOM refs ───────────────────────────────────────────── */
const form          = document.getElementById("chatForm");
const input         = document.getElementById("messageInput");
const messagesEl    = document.getElementById("messages");
const emptyState    = document.getElementById("emptyState");
const sendButton    = document.getElementById("sendButton");
const sendLabel     = sendButton.querySelector(".send-label");
const newChatButton = document.getElementById("newChatButton");
const chatListItems = document.getElementById("chatListItems");
const emptyChats    = document.getElementById("emptyChats");
const chatCount     = document.getElementById("chatCount");
const chatSubtitle  = document.getElementById("chatSubtitle");

/* ── State ──────────────────────────────────────────────── */
let conversationId = localStorage.getItem("saferesponse_conversation_id");
let conversations  = [];
let pollTimer      = null;

/* ── Pipeline stages ────────────────────────────────────── */
const STAGES = [
  { key: "query",        label: "Query" },
  { key: "retrieval",    label: "Retrieval" },
  { key: "generation",   label: "Generation" },
  { key: "trace",        label: "Trace" },
  { key: "verification", label: "Verification" },
  { key: "fusion",       label: "Fusion" },
  { key: "output",       label: "Output" },
];

// Estimated cumulative progress per stage (ms from job start)
const STAGE_TIMING = [200, 700, 1800, 2600, 3600, 4400, 5200];

/* ── Utilities ──────────────────────────────────────────── */
function syncEmptyState() {
  emptyState.hidden = Boolean(messagesEl.querySelector(".message-row"));
}

function trimText(text, maxLen) {
  const s = String(text || "").replace(/\s+/g, " ").trim();
  return s.length <= maxLen ? s : `${s.slice(0, maxLen - 1).trim()}…`;
}

function formatTime(ts) {
  if (!ts) return "";
  const d    = new Date(ts * 1000);
  const today = new Date();
  const sameDay = d.toDateString() === today.toDateString();
  return new Intl.DateTimeFormat(undefined, {
    hour:   sameDay ? "numeric"   : undefined,
    minute: sameDay ? "2-digit"   : undefined,
    month:  sameDay ? undefined   : "short",
    day:    sameDay ? undefined   : "numeric",
  }).format(d);
}

function setSendState(label, disabled = false) {
  sendButton.disabled = disabled;
  sendLabel.textContent = label;
}

function clearMessages() {
  messagesEl.querySelectorAll(".message-row").forEach((n) => n.remove());
  syncEmptyState();
}

function escapeHtml(str) {
  return String(str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/* ── Render assistant text (paragraphs / lists) ─────────── */
function renderAssistantText(container, text) {
  container.innerHTML = "";
  const value = String(text || "").trim();
  if (!value) return;

  const blocks = value.split(/\n{2,}/);
  blocks.forEach((block) => {
    const lines = block.split("\n").filter((l) => l.trim());
    const isList      = lines.length > 1 && lines.every((l) => /^[-*]\s+/.test(l.trim()));
    const isNumbered  = lines.length > 1 && lines.every((l) => /^\d+\.\s+/.test(l.trim()));

    if (isList || isNumbered) {
      const list = document.createElement(isNumbered ? "ol" : "ul");
      lines.forEach((l) => {
        const li = document.createElement("li");
        li.textContent = l.replace(/^([-*]|\d+\.)\s+/, "");
        list.appendChild(li);
      });
      container.appendChild(list);
      return;
    }

    const p = document.createElement("p");
    p.textContent = block;
    container.appendChild(p);
  });
}

/* ── Verdict rendering ──────────────────────────────────── */
const DECISION_META = {
  ACCEPT:  { cls: "badge-accept",  icon: "✓", label: "Verified"      },
  RERANK:  { cls: "badge-rerank",  icon: "⟳", label: "Re-ranked"     },
  REWRITE: { cls: "badge-rewrite", icon: "✎", label: "Rewritten"     },
  REJECT:  { cls: "badge-reject",  icon: "✕", label: "Rejected"      },
  DIRECT:  { cls: "badge-direct",  icon: "→", label: "Direct reply"  },
};

const CONF_CLS = {
  green: "conf-high",
  amber: "conf-med",
  red:   "conf-low",
  HIGH:  "conf-high",
  MEDIUM:"conf-med",
  LOW:   "conf-low",
};

function buildVerdictBlock(data) {
  const decision = (data.decision || "DIRECT").toUpperCase();
  const meta     = DECISION_META[decision] || DECISION_META.DIRECT;
  const confKey  = data.confidence_color || data.confidence;
  const confCls  = CONF_CLS[confKey] || "conf-null";

  const block = document.createElement("div");
  block.className = "verdict-block";
  block.setAttribute("role", "region");
  block.setAttribute("aria-label", "Safety verdict");

  /* ─ Header ─ */
  const header = document.createElement("div");
  header.className = "verdict-header";

  // Decision badge
  const badge = document.createElement("span");
  badge.className = `decision-badge ${meta.cls}`;
  badge.setAttribute("aria-label", `Decision: ${decision}`);
  badge.innerHTML = `<span aria-hidden="true">${meta.icon}</span> ${meta.label}`;
  header.appendChild(badge);

  // Confidence chip
  if (data.confidence) {
    const chip = document.createElement("span");
    chip.className = `conf-chip ${confCls}`;
    chip.setAttribute("aria-label", `Confidence: ${data.confidence}`);
    chip.innerHTML = `<span aria-hidden="true">◆</span> ${data.confidence}`;
    header.appendChild(chip);
  }

  // Timing badge (top-right)
  const total = data.pipeline_summary?.total_stage_time_seconds;
  if (total != null) {
    const tb = document.createElement("span");
    tb.className = "timing-badge";
    tb.textContent = `${Number(total).toFixed(2)}s`;
    tb.setAttribute("aria-label", `Total pipeline time: ${total}s`);
    header.appendChild(tb);
  }

  block.appendChild(header);

  /* ─ Body ─ */
  const body = document.createElement("div");
  body.className = "verdict-body";

  // Source citation
  if (data.source) {
    const card = document.createElement("div");
    card.className = "source-card";
    card.setAttribute("aria-label", `Source: ${data.source}`);
    card.innerHTML = `
      <span class="source-icon" aria-hidden="true">📄</span>
      <div>
        <span class="source-label">Source</span>
        <span class="source-text">${escapeHtml(data.source)}</span>
      </div>`;
    body.appendChild(card);
  }

  // Decision reason
  if (data.decision_reason) {
    const dr = document.createElement("div");
    dr.className = "decision-reason";
    dr.innerHTML = `<span class="decision-reason-label">Verdict rationale</span>${escapeHtml(data.decision_reason)}`;
    body.appendChild(dr);
  }

  // Risk panel (collapsible)
  const hasRisk = (data.risk_explanation || (data.warnings && data.warnings.length));
  if (hasRisk) {
    const panel = document.createElement("div");
    panel.className = "risk-panel";
    const riskScore = data.risk_score != null ? ` · Risk ${Math.round(data.risk_score * 100)}%` : "";
    panel.innerHTML = `
      <button class="risk-toggle" aria-expanded="false" aria-controls="risk-content-${Date.now()}">
        <span>⚠ Risk signals${riskScore}</span>
        <span class="risk-toggle-icon" aria-hidden="true">▶</span>
      </button>
      <div class="risk-content" role="region">
        ${data.risk_explanation
          ? `<p class="risk-explanation">${escapeHtml(data.risk_explanation)}</p>`
          : ""}
        ${data.warnings && data.warnings.length
          ? `<div class="warning-chips" role="list" aria-label="Warning signals">${
              data.warnings.map((w) =>
                `<span class="warning-chip" role="listitem">⚠ ${escapeHtml(w)}</span>`
              ).join("")
            }</div>`
          : ""}
      </div>`;
    const toggle = panel.querySelector(".risk-toggle");
    toggle.addEventListener("click", () => {
      const open = panel.classList.toggle("is-open");
      toggle.setAttribute("aria-expanded", String(open));
    });
    body.appendChild(panel);
  }

  // Pipeline stage timings
  const timings = data.pipeline_summary?.stage_timings_seconds;
  if (timings && Object.keys(timings).length) {
    const section = document.createElement("div");
    section.className = "timings-section";
    const title = document.createElement("div");
    title.className = "timings-title";
    title.textContent = "Pipeline stages";
    section.appendChild(title);

    const maxVal = Math.max(...Object.values(timings).filter((v) => v != null), 0.001);
    Object.entries(timings).forEach(([stage, val]) => {
      if (val == null) return;
      const pct = Math.min(100, Math.round((val / maxVal) * 100));
      const row = document.createElement("div");
      row.className = "timing-row";
      row.innerHTML = `
        <span class="timing-name">${escapeHtml(stage.replace(/_/g, " "))}</span>
        <div class="timing-bar-wrap" role="progressbar" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100" aria-label="${stage}: ${val}s">
          <div class="timing-bar" style="width:${pct}%"></div>
        </div>
        <span class="timing-val">${Number(val).toFixed(2)}s</span>`;
      section.appendChild(row);
    });
    body.appendChild(section);
  }

  block.appendChild(body);
  return block;
}

/* ── Pipeline progress UI ───────────────────────────────── */
function buildPipelineProgress() {
  const wrap = document.createElement("div");
  wrap.className = "pipeline-progress";
  wrap.setAttribute("aria-label", "Pipeline running");
  wrap.setAttribute("aria-live", "polite");

  const label = document.createElement("div");
  label.className = "pipeline-label";
  label.textContent = "Running pipeline";
  wrap.appendChild(label);

  const stages = document.createElement("div");
  stages.className = "pipeline-stages";

  STAGES.forEach(({ key, label: stageName }) => {
    const el = document.createElement("div");
    el.className = "pipeline-stage";
    el.dataset.stageKey = key;
    el.innerHTML = `<span class="stage-dot"></span><span>${stageName}</span>`;
    stages.appendChild(el);
  });

  wrap.appendChild(stages);
  return wrap;
}

function advancePipeline(progressEl, stageIndex) {
  const stageEls = progressEl.querySelectorAll(".pipeline-stage");
  stageEls.forEach((el, i) => {
    el.classList.remove("stage-done", "stage-active");
    const dot = el.querySelector(".stage-dot");
    if (i < stageIndex) {
      el.classList.add("stage-done");
      dot.textContent = "✓";
    } else if (i === stageIndex) {
      el.classList.add("stage-active");
      dot.textContent = "";
    } else {
      dot.textContent = "";
    }
  });
}

function markPipelineDone(progressEl) {
  const stageEls = progressEl.querySelectorAll(".pipeline-stage");
  stageEls.forEach((el) => {
    el.classList.remove("stage-active");
    el.classList.add("stage-done");
    el.querySelector(".stage-dot").textContent = "✓";
  });
}

/* ── Add a message row ──────────────────────────────────── */
function addMessage(role, text) {
  const row = document.createElement("article");
  row.className = `message-row ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = role === "user" ? "U" : "S";

  const content = document.createElement("div");
  content.className = "message-content";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  content.appendChild(bubble);
  row.appendChild(avatar);
  row.appendChild(content);
  messagesEl.appendChild(row);

  if (role === "assistant") {
    renderAssistantText(bubble, text);
  } else {
    bubble.textContent = text;
  }

  syncEmptyState();
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return row;
}

/* ── Write completed message + verdict ──────────────────── */
function writeMessage(row, data) {
  const bubble  = row.querySelector(".message-bubble");
  const content = row.querySelector(".message-content");

  // Remove old pipeline progress if any
  const oldProgress = content.querySelector(".pipeline-progress");
  if (oldProgress) oldProgress.remove();

  if (typeof data === "string") {
    // Error / plain string path
    renderAssistantText(bubble, data);
    row.classList.remove("is-loading");
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return;
  }

  const answerText = data.formatted_response || data.answer || "";
  renderAssistantText(bubble, answerText);
  row.classList.remove("is-loading");

  // Append verdict block (not for blank answers)
  if (answerText) {
    const verdict = buildVerdictBlock(data);
    content.appendChild(verdict);
  }

  messagesEl.scrollTop = messagesEl.scrollHeight;
}

/* ── HTTP helper ────────────────────────────────────────── */
async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const rawBody  = await response.text();
  let data = {};
  try { data = rawBody ? JSON.parse(rawBody) : {}; } catch { data = { detail: rawBody || "Non-JSON response." }; }
  if (!response.ok) throw new Error(data.detail || "Request failed");
  return data;
}

/* ── Conversation list ──────────────────────────────────── */
async function refreshConversations() {
  try {
    const data = await requestJson("/v1/conversations?limit=60");
    conversations = data.conversations || [];
  } catch { conversations = []; }
  renderChatList();
}

function renderChatList() {
  chatListItems.innerHTML = "";
  chatCount.textContent  = conversations.length;
  emptyChats.hidden      = conversations.length > 0;

  conversations.forEach((chat) => {
    const item = document.createElement("button");
    item.className = "chat-list-item";
    if (chat.conversation_id === conversationId) item.classList.add("is-active");
    item.type = "button";
    item.dataset.conversationId = chat.conversation_id;
    item.setAttribute("aria-label", `Load chat: ${chat.title || "New chat"}`);

    const title = document.createElement("span");
    title.className = "chat-title";
    title.textContent = trimText(chat.title || "New chat", 44);

    const meta = document.createElement("span");
    meta.className = "chat-meta";
    meta.textContent = trimText(chat.preview || formatTime(chat.updated_at), 54);

    const time = document.createElement("span");
    time.className = "chat-time";
    time.textContent = formatTime(chat.updated_at);

    item.appendChild(title);
    item.appendChild(meta);
    item.appendChild(time);
    item.addEventListener("click", () => loadConversation(chat.conversation_id));
    chatListItems.appendChild(item);
  });
}

/* ── Load a conversation ────────────────────────────────── */
async function loadConversation(nextId) {
  try {
    const conversation = await requestJson(`/v1/conversations/${encodeURIComponent(nextId)}`);
    conversationId = conversation.conversation_id;
    localStorage.setItem("saferesponse_conversation_id", conversationId);
    clearMessages();
    conversation.turns.forEach((turn) => {
      if (turn.role === "user" || turn.role === "assistant") {
        const row = addMessage(turn.role, turn.text || "");
        // Re-render verdict if stored result is available
        if (turn.role === "assistant" && turn.result && typeof turn.result === "object") {
          const content = row.querySelector(".message-content");
          if (turn.result.decision || turn.result.confidence) {
            content.appendChild(buildVerdictBlock(turn.result));
          }
        }
      }
    });
    chatSubtitle.textContent = "Saved chat";
    renderChatList();
  } catch {
    conversationId = null;
    localStorage.removeItem("saferesponse_conversation_id");
    clearMessages();
    chatSubtitle.textContent = "Start a new verified chat";
    renderChatList();
  }
}

/* ── Job polling ────────────────────────────────────────── */
function waitForJob(jobId, progressEl) {
  return new Promise((resolve, reject) => {
    const start    = Date.now();
    let attempts   = 0;
    let stageIndex = 0;

    pollTimer = window.setInterval(async () => {
      attempts += 1;
      setSendState(attempts % 2 === 0 ? "Running…" : "Checking…", true);

      // Advance pipeline stage indicator
      const elapsed = Date.now() - start;
      while (stageIndex < STAGES.length - 1 && elapsed > STAGE_TIMING[stageIndex]) {
        stageIndex += 1;
      }
      if (progressEl) advancePipeline(progressEl, stageIndex);

      try {
        const job = await requestJson(`/v1/chat/jobs/${jobId}`);
        if (job.status === "completed") {
          window.clearInterval(pollTimer);
          pollTimer = null;
          if (progressEl) markPipelineDone(progressEl);
          resolve(job.result);
        }
        if (job.status === "failed") {
          window.clearInterval(pollTimer);
          pollTimer = null;
          reject(new Error(job.error || "Pipeline job failed"));
        }
      } catch (err) {
        window.clearInterval(pollTimer);
        pollTimer = null;
        reject(err);
      }
    }, 1000);
  });
}

/* ── New chat ───────────────────────────────────────────── */
function startNewChat() {
  conversationId = null;
  localStorage.removeItem("saferesponse_conversation_id");
  clearMessages();
  chatSubtitle.textContent = "Start a new verified chat";
  renderChatList();
  input.focus();
}

/* ── Event listeners ────────────────────────────────────── */
newChatButton.addEventListener("click", startNewChat);

document.querySelectorAll(".header-theme-btn, .theme-toggle").forEach((btn) => {
  btn.addEventListener("click", toggleTheme);
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

/* ── Form submit ────────────────────────────────────────── */
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = input.value.trim();
  if (!message) return;

  addMessage("user", message);
  input.value = "";
  input.style.height = "auto";

  const assistantRow = addMessage("assistant", "");
  const content      = assistantRow.querySelector(".message-content");
  const bubble       = assistantRow.querySelector(".message-bubble");
  assistantRow.classList.add("is-loading");

  // "Thinking..." placeholder
  const thinkP = document.createElement("p");
  thinkP.textContent = "Thinking";
  bubble.appendChild(thinkP);

  // Pipeline progress strip
  const progressEl = buildPipelineProgress();
  content.appendChild(progressEl);
  advancePipeline(progressEl, 0);

  setSendState("Queued…", true);

  try {
    const job = await requestJson("/v1/chat/jobs", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({
        conversation_id: conversationId,
        message,
        run_pipeline: true,
      }),
    });

    const data = await waitForJob(job.job_id, progressEl);

    if (data.conversation_id) {
      conversationId = data.conversation_id;
      localStorage.setItem("saferesponse_conversation_id", conversationId);
    }

    // Small pause so user sees "done" pipeline state
    await new Promise((r) => setTimeout(r, 300));

    writeMessage(assistantRow, data);
    chatSubtitle.textContent = "Saved chat";
    await refreshConversations();

  } catch (err) {
    // Remove progress strip on error
    const prog = content.querySelector(".pipeline-progress");
    if (prog) prog.remove();
    writeMessage(assistantRow, err.message || "Something went wrong.");
  } finally {
    if (pollTimer) { window.clearInterval(pollTimer); pollTimer = null; }
    setSendState("Send", false);
    input.focus();
  }
});

/* ── Boot ───────────────────────────────────────────────── */
async function boot() {
  updateThemeIcons();
  await refreshConversations();
  if (conversationId) await loadConversation(conversationId);
  syncEmptyState();
  input.focus();
}

boot();
