const state = {
  config: null,
  selectedModels: new Set(),
  selectedSuites: new Set(),
  activeRunId: null,
  pollTimer: null,
  currentRunData: null,
  selectedCompareRuns: new Set(),
  compareSearchQuery: "",
  compareSortCol: null,
  compareSortDir: "asc",
  modelCatalog: null,
  allModels: [],
};

// ---------- theme ----------
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("localbench-theme", theme);
}
(function initTheme() {
  applyTheme(localStorage.getItem("localbench-theme") || "dark");
})();
document.getElementById("theme-toggle").addEventListener("click", () => {
  const current = document.documentElement.getAttribute("data-theme");
  applyTheme(current === "dark" ? "light" : "dark");
});

// ---------- tabs ----------
document.querySelectorAll("nav.tabs button").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("nav.tabs button").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`view-${btn.dataset.view}`).classList.add("active");
    if (btn.dataset.view === "history") loadRunList();
    if (btn.dataset.view === "playground") populatePlaygroundModels();
    if (btn.dataset.view === "compare") loadComparePicker();
    if (btn.dataset.view === "settings") loadSettings();
  });
});

async function api(path, opts) {
  const resp = await fetch(path, opts);
  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`${resp.status}: ${body}`);
  }
  return resp.json();
}

const SUITE_DESCRIPTIONS = {
  json_schema: "json_schema — asks the model to produce structured JSON matching a given schema (nested objects, enums, regex patterns, etc.), validated with jsonschema. Tests instruction-precision for structured output.",
  coding: "coding — 8 small HumanEval-style problems (factorial, palindrome check, binary search, etc.). The model's code is actually executed in a sandboxed subprocess against real test cases.",
  logic_math: "logic_math — synthetic arithmetic and logic puzzles generated dynamically (exact match ground truth). Tests basic reasoning without needing outside knowledge.",
  instruction_following: "instruction_following — asks for text following precise mechanical constraints (exact 3 paragraphs, no letter 'e', ends with specific phrase). Graded by literally counting/checking output.",
  pattern_reasoning: "pattern_reasoning — small abstract grid-transformation puzzles (infer rule from 2 examples, apply to new grid). Hardest, most abstract-reasoning-heavy suite by design.",
  long_context: "long_context — a real 1000+ line source file excerpt with a planted 'needle' value to retrieve or a mechanically injected bug to locate by line number. Tests context retention."
};

// ---------- New Run ----------
function renderChecklist(container, items, selectedSet) {
  container.innerHTML = "";
  items.forEach((item) => {
    const label = document.createElement("label");
    label.className = "chip" + (selectedSet.has(item) ? " checked" : "");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = selectedSet.has(item);
    cb.addEventListener("change", () => {
      if (cb.checked) selectedSet.add(item);
      else selectedSet.delete(item);
      label.classList.toggle("checked", cb.checked);
    });
    label.appendChild(cb);
    label.appendChild(document.createTextNode(item));

    if (SUITE_DESCRIPTIONS[item]) {
      const infoBtn = document.createElement("span");
      infoBtn.className = "info-icon";
      infoBtn.innerHTML = " &#9432;";
      infoBtn.title = SUITE_DESCRIPTIONS[item];
      infoBtn.style.cursor = "help";
      infoBtn.style.opacity = "0.7";
      infoBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        alert(SUITE_DESCRIPTIONS[item]);
      });
      label.appendChild(infoBtn);
    }

    container.appendChild(label);
  });
}

// ---------- Model Parsing & Grid Cards ----------
function parseModelMetadata(name) {
  const meta = {
    family: null,
    size: null,
    tags: [],
    isEmbedding: false,
  };

  const lower = name.toLowerCase();

  if (lower.includes("qwen")) meta.family = "Qwen";
  else if (lower.includes("gemma")) meta.family = "Gemma";
  else if (lower.includes("mistral") || lower.includes("devstral")) meta.family = "Mistral";
  else if (lower.includes("nvidia") || lower.includes("nemotron")) meta.family = "NVIDIA";
  else if (lower.includes("glm")) meta.family = "GLM";
  else if (lower.includes("gpt-oss")) meta.family = "GPT-OSS";
  else if (lower.includes("llama")) meta.family = "Llama";
  else if (lower.includes("phi")) meta.family = "Phi";

  const sizeMatch = lower.match(/(\d+(?:\.\d+)?)(b)\b/);
  if (sizeMatch) {
    meta.size = `${sizeMatch[1].toUpperCase()}B`;
  }

  if (lower.includes("coder")) meta.tags.push("Coder");
  if (lower.includes("reasoning")) meta.tags.push("Reasoning");
  if (lower.includes("qat")) meta.tags.push("QAT");
  if (lower.includes("mtp")) meta.tags.push("MTP");
  if (lower.includes("vision") || lower.includes("vl")) meta.tags.push("Vision");

  const quantMatch = name.match(/Q\d+_[K_A-Z0-9]+/i);
  if (quantMatch) {
    meta.tags.push(quantMatch[0].toUpperCase());
  }

  if (lower.includes("embed") || lower.includes("nomic")) {
    meta.isEmbedding = true;
  }

  return meta;
}

function updateModelSelectCount(allModels, selectedSet) {
  const countEl = document.getElementById("model-select-count");
  if (countEl) {
    countEl.textContent = `Selected: ${selectedSet.size} of ${allModels.length}`;
  }
}

function formatBytesGB(bytes) {
  return bytes ? `${(bytes / 1024 ** 3).toFixed(1)}GB` : null;
}

function buildBadgesFromCatalog(entry) {
  // Real fields from /api/models/catalog (lms ls --json) -- never guessed.
  const badges = [];
  if (entry.publisher) badges.push({ text: entry.publisher, cls: "family" });
  if (entry.params) badges.push({ text: entry.params, cls: "size" });
  if (entry.quantization) badges.push({ text: entry.quantization, cls: "quant" });
  const sizeStr = formatBytesGB(entry.size_bytes);
  if (sizeStr) badges.push({ text: sizeStr, cls: "quant" });
  if (entry.context_length) badges.push({ text: `${entry.context_length.toLocaleString()} ctx`, cls: "quant" });
  if (entry.vision) badges.push({ text: "Vision", cls: "quant" });
  if (entry.tool_use) badges.push({ text: "Tool Use", cls: "quant" });
  if (entry.type === "embedding") badges.push({ text: "Embedding (Non-Chat)", cls: "warn" });
  return badges;
}

function buildBadgesFromNameGuess(modelName) {
  // Fallback only, for runtimes with no /api/models/catalog entry (e.g.
  // Ollama/llama.cpp) -- a best-effort guess from the name string, clearly
  // a lower-confidence source than real catalog data.
  const meta = parseModelMetadata(modelName);
  const badges = [];
  if (meta.family) badges.push({ text: meta.family, cls: "family" });
  if (meta.size) badges.push({ text: meta.size, cls: "size" });
  meta.tags.forEach((tag) => badges.push({ text: tag, cls: "quant" }));
  if (meta.isEmbedding) badges.push({ text: "Embedding (Non-Chat)", cls: "warn" });
  return badges;
}

function renderModelGrid(container, models, selectedSet, searchQuery = "") {
  state.allModels = models;
  container.innerHTML = "";
  const query = searchQuery.trim().toLowerCase();
  const filtered = models.filter((m) => !query || m.toLowerCase().includes(query));
  const catalog = state.modelCatalog || {};

  filtered.forEach((modelName) => {
    const isChecked = selectedSet.has(modelName);
    const catalogEntry = catalog[modelName];
    const badgeData = catalogEntry ? buildBadgesFromCatalog(catalogEntry) : buildBadgesFromNameGuess(modelName);

    const card = document.createElement("label");
    card.className = "model-card" + (isChecked ? " checked" : "");

    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = isChecked;
    cb.addEventListener("change", () => {
      if (cb.checked) selectedSet.add(modelName);
      else selectedSet.delete(modelName);
      card.classList.toggle("checked", cb.checked);
      updateModelSelectCount(models, selectedSet);
    });
    card.appendChild(cb);

    const info = document.createElement("div");
    info.className = "model-card-info";

    const nameEl = document.createElement("div");
    nameEl.className = "model-card-name";
    nameEl.textContent = (catalogEntry && catalogEntry.display_name) || modelName;
    info.appendChild(nameEl);

    const badges = document.createElement("div");
    badges.className = "model-card-badges";
    badgeData.forEach(({ text, cls }) => {
      const b = document.createElement("span");
      b.className = `model-badge ${cls}`;
      b.textContent = text;
      badges.appendChild(b);
    });
    if (!catalogEntry) {
      const b = document.createElement("span");
      b.className = "model-badge warn";
      b.title = "No catalog data from the runtime -- badges above are guessed from the name only";
      b.textContent = "unverified";
      badges.appendChild(b);
    }

    info.appendChild(badges);
    card.appendChild(info);
    container.appendChild(card);
  });

  if (filtered.length === 0) {
    container.innerHTML = '<p class="empty-state">No models match search filter.</p>';
  }

  updateModelSelectCount(models, selectedSet);
}

const SUITE_METADATA = {
  json_schema: {
    icon: "⚙️",
    title: "JSON Schema",
    description: "Structured JSON output validated against a schema — exact pass/fail.",
    info: "json_schema — asks the model to produce structured JSON matching a given schema (nested objects, enums, regex patterns, etc.), validated with jsonschema. Tests instruction-precision for structured output.",
  },
  coding: {
    icon: "🐍",
    title: "Coding Sandbox",
    description: "Generated Python executed against real unit tests in an isolated subprocess.",
    info: "coding — 8 small HumanEval-style problems (factorial, palindrome check, binary search, etc.). The model's code is actually executed in a sandboxed subprocess against real test cases.",
  },
  logic_math: {
    icon: "🧩",
    title: "Logic & Math",
    description: "Synthetic arithmetic/logic puzzles graded by exact match.",
    info: "logic_math — synthetic arithmetic and logic puzzles generated dynamically (exact match ground truth). Tests basic reasoning without needing outside knowledge.",
  },
  instruction_following: {
    icon: "📋",
    title: "Instruction Following",
    description: "Mechanical format constraints (word/paragraph/keyword counts), IFEval-style rule checking.",
    info: "instruction_following — asks for text following precise mechanical constraints (exact 3 paragraphs, no letter 'e', ends with specific phrase). Graded by literally counting/checking output.",
  },
  pattern_reasoning: {
    icon: "🧠",
    title: "Pattern Reasoning",
    description: "Abstract grid-transformation puzzles, ARC-AGI-style, graded by exact output match.",
    info: "pattern_reasoning — small abstract grid-transformation puzzles (infer rule from 2 examples, apply to new grid). Hardest, most abstract-reasoning-heavy suite by design.",
  },
  long_context: {
    icon: "📄",
    title: "Long Context",
    description: "Needle-in-haystack and bug-location retrieval across a long real document.",
    info: "long_context — a real 1000+ line source file excerpt with a planted 'needle' value to retrieve or a mechanically injected bug to locate by line number. Tests context retention.",
  },
};

function renderSuiteGrid(container, suitesConfig, selectedSet) {
  container.innerHTML = "";
  const suiteNames = Object.keys(suitesConfig);

  suiteNames.forEach((suiteName) => {
    const isChecked = selectedSet.has(suiteName);
    const meta = SUITE_METADATA[suiteName] || {
      icon: "⚙️",
      title: suiteName,
      description: "Suite benchmark evaluation",
      info: suiteName,
    };

    const suiteInfo = typeof suitesConfig[suiteName] === "object" ? suitesConfig[suiteName] : {};
    const taskCount = suiteInfo.task_count != null ? suiteInfo.task_count : 0;

    const card = document.createElement("label");
    card.className = "suite-card" + (isChecked ? " checked" : "");

    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = isChecked;
    cb.addEventListener("change", () => {
      if (cb.checked) selectedSet.add(suiteName);
      else selectedSet.delete(suiteName);
      card.classList.toggle("checked", cb.checked);
    });
    card.appendChild(cb);

    const info = document.createElement("div");
    info.className = "suite-card-info";

    const header = document.createElement("div");
    header.className = "suite-card-header";

    const titleEl = document.createElement("div");
    titleEl.className = "suite-card-title";
    titleEl.textContent = `${meta.icon} ${meta.title}`;
    header.appendChild(titleEl);

    info.appendChild(header);

    const descEl = document.createElement("div");
    descEl.className = "suite-card-desc";
    descEl.textContent = meta.description;
    info.appendChild(descEl);

    const footer = document.createElement("div");
    footer.className = "suite-card-footer";

    const badge = document.createElement("span");
    badge.className = "suite-task-badge";
    badge.textContent = `${taskCount} tasks`;
    footer.appendChild(badge);

    const infoBtn = document.createElement("span");
    infoBtn.className = "info-icon";
    infoBtn.innerHTML = " &#9432;";
    infoBtn.title = meta.info;
    infoBtn.style.cursor = "help";
    infoBtn.style.opacity = "0.7";
    infoBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      alert(meta.info);
    });
    footer.appendChild(infoBtn);

    info.appendChild(footer);
    card.appendChild(info);
    container.appendChild(card);
  });
}

async function loadModelCatalog() {
  try {
    const data = await api("/api/models/catalog");
    state.modelCatalog = data.models || {};
  } catch (e) {
    state.modelCatalog = {};
  }
}

async function loadConfig() {
  state.config = await api("/api/config");
  state.config.models.forEach((m) => state.selectedModels.add(m));
  Object.entries(state.config.suites).forEach(([name, sVal]) => {
    const isEnabled = typeof sVal === "object" ? sVal.enabled : sVal;
    if (isEnabled) state.selectedSuites.add(name);
  });
  if (!state.modelCatalog) await loadModelCatalog();
  renderModelGrid(document.getElementById("model-checklist"), state.config.models, state.selectedModels);
  renderSuiteGrid(
    document.getElementById("suite-checklist"),
    state.config.suites,
    state.selectedSuites
  );
  await loadFrontierJudgeCard();
}

async function loadFrontierJudgeCard() {
  const card = document.getElementById("frontier-judge-card");
  const disabledNote = document.getElementById("frontier-judge-disabled-note");
  const disabledText = document.getElementById("frontier-judge-disabled-text");
  try {
    const settings = await api("/api/settings");
    state.judgeSettings = settings.judge;
    state.judgeKeys = settings.keys;
    state.judgeSdks = settings.sdks || null;
    state.judgeSdkPackages = settings.sdk_packages || null;
    const provider = settings.judge.provider || "anthropic";
    const hasKeyForProvider = !!settings.keys[provider];

    if (settings.judge.enabled && hasKeyForProvider) {
      card.style.display = "block";
      disabledNote.style.display = "none";
      document.getElementById("run-judge-provider").value = provider;
      document.getElementById("run-judge-model").value = settings.judge.model || "";
      await Promise.all([loadJudgeModelOptions(provider), updateFrontierJudgeInfo()]);
    } else {
      card.style.display = "none";
      disabledNote.style.display = "block";
      // Diagnose exactly what's missing rather than one generic message --
      // "not enabled" and "no key for the selected provider" are different
      // problems with different fixes, and it's easy to fix one while
      // forgetting the other (e.g. picking + saving a provider/model
      // without separately checking "Enabled").
      const missing = [];
      if (!settings.judge.enabled) missing.push('the "Enabled" checkbox is unchecked');
      // provider is escaped: it originates from config.yaml, which a user can
      // hand-edit to anything, and this string goes into innerHTML.
      if (!hasKeyForProvider) missing.push(`no API key is set for ${escapeHtml(PROVIDER_LABELS[provider] || provider)} (the currently configured provider)`);
      disabledText.innerHTML =
        `Frontier judge isn't active yet: ${missing.join(" and ")}. Fix this in ` +
        `<a href="#" data-view-link="settings">Settings</a> to unlock an optional 7th, paid, non-deterministic suite.`;
    }
  } catch {
    card.style.display = "none";
    disabledNote.style.display = "none";
  }
}

async function updateFrontierJudgeInfo() {
  const provider = document.getElementById("run-judge-provider").value;
  const model = document.getElementById("run-judge-model").value.trim();
  const includeEl = document.getElementById("include-frontier-graded");
  const detailEl = document.getElementById("frontier-judge-detail");
  const estimateEl = document.getElementById("frontier-judge-estimate");
  const numTasks = state.judgeSettings?.num_tasks ?? 6;
  const passThreshold = state.judgeSettings?.pass_threshold ?? 7;

  const costEl2 = document.getElementById("frontier-judge-cost");
  const hasKey = !!(state.judgeKeys && state.judgeKeys[provider]);
  if (!hasKey) {
    includeEl.checked = false;
    includeEl.disabled = true;
    detailEl.textContent = `No API key set for ${PROVIDER_LABELS[provider] || provider} -- add one in Settings to use it as the judge.`;
    estimateEl.textContent = "";
    if (costEl2) costEl2.textContent = "";
    return;
  }
  // A key alone isn't enough: each provider's SDK is an optional install, so
  // check it here rather than letting the run die on its first judge call.
  const sdkOk = !state.judgeSdks || state.judgeSdks[provider] !== false;
  if (!sdkOk) {
    const pkg = (state.judgeSdkPackages && state.judgeSdkPackages[provider]) || provider;
    includeEl.checked = false;
    includeEl.disabled = true;
    detailEl.textContent =
      `The ${PROVIDER_LABELS[provider] || provider} SDK isn't installed, so this judge can't run yet. ` +
      `Install it with:  pip install ${pkg}`;
    estimateEl.textContent = "";
    if (costEl2) costEl2.textContent = "";
    return;
  }
  includeEl.disabled = false;
  detailEl.textContent =
    `${numTasks} task(s), pass threshold ${passThreshold}/10 -- exactly ${numTasks * 2} paid API calls ` +
    `(${numTasks} to generate tasks + ${numTasks} to grade responses). Generates its own tasks and grades this ` +
    `model's responses; kept separate from the deterministic pass rates. See TASK_SPEC.md.`;

  const costEl = document.getElementById("frontier-judge-cost");
  if (!model) {
    estimateEl.textContent = "";
    costEl.textContent = "";
    return;
  }
  estimateEl.textContent = "Checking for time estimate from a previous run with this judge...";
  costEl.textContent = "";
  try {
    const history = await api(`/api/settings/judge/history?provider=${encodeURIComponent(provider)}&model=${encodeURIComponent(model)}`);
    if (history.found) {
      const totalSeconds = history.avg_seconds_per_task * numTasks;
      estimateEl.textContent =
        `Estimated duration: ~${fmtDuration(totalSeconds)}, based on ${fmtNum(history.avg_seconds_per_task, 1)}s/task ` +
        `observed in a previous run (${history.run_id}) with this same judge. Actual time varies with prompt/response length.`;
    } else {
      estimateEl.textContent =
        "No previous run with this exact provider/model yet -- duration can't be estimated until after the first run. " +
        "Task and call counts above are exact regardless.";
    }
  } catch {
    estimateEl.textContent = "";
  }

  try {
    const cost = await api(
      `/api/settings/judge/cost-estimate?provider=${encodeURIComponent(provider)}&model=${encodeURIComponent(model)}&num_tasks=${numTasks}`
    );
    if (cost.available) {
      costEl.textContent =
        `Estimated cost: ~$${cost.total_cost_usd} (OpenRouter's live pricing: $${cost.prompt_rate_per_1m}/$${cost.completion_rate_per_1m} per 1M ` +
        `prompt/completion tokens, applied to real token usage from run ${cost.based_on_run}).`;
    } else {
      costEl.textContent = cost.reason;
    }
  } catch {
    costEl.textContent = "";
  }
}

async function loadJudgeModelOptions(provider, datalistId = "run-judge-model-options", statusId = "frontier-judge-model-status") {
  const datalist = document.getElementById(datalistId);
  const statusEl = document.getElementById(statusId);
  datalist.innerHTML = "";
  statusEl.textContent = "Loading available models...";
  try {
    const { models } = await api(`/api/settings/judge/models?provider=${encodeURIComponent(provider)}`);
    models.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m;
      datalist.appendChild(opt);
    });
    statusEl.textContent = models.length
      ? `${models.length} model(s) available from ${PROVIDER_LABELS[provider] || provider} -- start typing to see suggestions.`
      : `Could not fetch a live model list for ${PROVIDER_LABELS[provider] || provider} -- type the model id manually.`;
  } catch {
    statusEl.textContent = `Could not fetch a live model list for ${PROVIDER_LABELS[provider] || provider} -- type the model id manually.`;
  }
}

async function onJudgeProviderChange() {
  const provider = document.getElementById("run-judge-provider").value;
  const modelInput = document.getElementById("run-judge-model");
  // Switching provider invalidates whatever model string was there before
  // (e.g. an Anthropic model id left in the field after switching to
  // Gemini) -- restore the saved default only if it's for this exact
  // provider, otherwise clear it so a stale/mismatched value is never
  // silently submitted.
  modelInput.value = state.judgeSettings?.provider === provider ? (state.judgeSettings.model || "") : "";
  await Promise.all([loadJudgeModelOptions(provider), updateFrontierJudgeInfo()]);
}

function fmtDuration(seconds) {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  const rem = Math.round(seconds % 60);
  return `${minutes}m ${rem}s`;
}

document.getElementById("run-judge-provider")?.addEventListener("change", onJudgeProviderChange);
document.getElementById("run-judge-model")?.addEventListener("change", updateFrontierJudgeInfo);

document.addEventListener("click", (e) => {
  const link = e.target.closest("[data-view-link]");
  if (!link) return;
  e.preventDefault();
  document.querySelector(`[data-view="${link.dataset.viewLink}"]`)?.click();
});

// Attach Suite Selection Controls
const suiteSelectAllBtn = document.getElementById("suite-select-all");
if (suiteSelectAllBtn) {
  suiteSelectAllBtn.addEventListener("click", () => {
    if (state.config?.suites) {
      Object.keys(state.config.suites).forEach((s) => state.selectedSuites.add(s));
      renderSuiteGrid(document.getElementById("suite-checklist"), state.config.suites, state.selectedSuites);
    }
  });
}
const suiteDeselectAllBtn = document.getElementById("suite-deselect-all");
if (suiteDeselectAllBtn) {
  suiteDeselectAllBtn.addEventListener("click", () => {
    state.selectedSuites.clear();
    if (state.config?.suites) {
      renderSuiteGrid(document.getElementById("suite-checklist"), state.config.suites, state.selectedSuites);
    }
  });
}

// Attach Search & Selection Controls
const modelSearchInput = document.getElementById("model-search");
if (modelSearchInput) {
  modelSearchInput.addEventListener("input", (e) => {
    if (state.allModels) {
      renderModelGrid(document.getElementById("model-checklist"), state.allModels, state.selectedModels, e.target.value);
    }
  });
}

const selectAllBtn = document.getElementById("model-select-all");
if (selectAllBtn) {
  selectAllBtn.addEventListener("click", () => {
    if (state.allModels) {
      const query = (document.getElementById("model-search")?.value || "").trim().toLowerCase();
      state.allModels.filter((m) => !query || m.toLowerCase().includes(query)).forEach((m) => state.selectedModels.add(m));
      renderModelGrid(document.getElementById("model-checklist"), state.allModels, state.selectedModels, query);
    }
  });
}

const deselectAllBtn = document.getElementById("model-deselect-all");
if (deselectAllBtn) {
  deselectAllBtn.addEventListener("click", () => {
    if (state.allModels) {
      const query = (document.getElementById("model-search")?.value || "").trim().toLowerCase();
      state.allModels.filter((m) => !query || m.toLowerCase().includes(query)).forEach((m) => state.selectedModels.delete(m));
      renderModelGrid(document.getElementById("model-checklist"), state.allModels, state.selectedModels, query);
    }
  });
}

document.getElementById("detect-models").addEventListener("click", async () => {
  const status = document.getElementById("detect-status");
  status.textContent = "detecting...";
  try {
    const [data] = await Promise.all([api("/api/models/detect"), loadModelCatalog()]);
    renderModelGrid(document.getElementById("model-checklist"), data.models, state.selectedModels);
    status.textContent = `found ${data.models.length} model(s) on the runtime`;
  } catch (e) {
    status.textContent = "detection failed: " + e.message;
  }
});

document.getElementById("start-run").addEventListener("click", async () => {
  const includeFrontierEl = document.getElementById("include-frontier-graded");
  const runFrontierGraded = !!(includeFrontierEl && includeFrontierEl.checked);
  const judgeProvider = document.getElementById("run-judge-provider")?.value;
  const judgeModel = document.getElementById("run-judge-model")?.value.trim();
  if (runFrontierGraded) {
    const numTasks = state.judgeSettings?.num_tasks ?? "?";
    const proceed = confirm(
      `This will also run the frontier-graded suite: ${numTasks} task(s) sent to ` +
      `${judgeProvider}/${judgeModel || "?"} for generation AND grading ` +
      `(${typeof numTasks === "number" ? numTasks * 2 : "2x"} paid API calls). This is non-deterministic and costs real money. Continue?`
    );
    if (!proceed) return;
  }

  const btn = document.getElementById("start-run");
  btn.disabled = true;
  document.getElementById("run-progress").style.display = "block";
  document.getElementById("run-log").textContent = "";
  try {
    const { run_id } = await api("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        models: Array.from(state.selectedModels),
        suites: Array.from(state.selectedSuites),
        run_frontier_graded: runFrontierGraded,
        judge_override: runFrontierGraded ? { provider: judgeProvider, model: judgeModel } : undefined,
      }),
    });
    state.activeRunId = run_id;
    localStorage.setItem("localbench-active-run", run_id);
    pollRun();
  } catch (e) {
    document.getElementById("run-log").textContent = "failed to start: " + e.message;
    btn.disabled = false;
  }
});

async function resumeActiveRunIfAny() {
  const savedRunId = localStorage.getItem("localbench-active-run");
  if (!savedRunId) return;
  try {
    const data = await api(`/api/run/${savedRunId}/status`);
    if (data.done) {
      localStorage.removeItem("localbench-active-run");
      return;
    }
    state.activeRunId = savedRunId;
    document.getElementById("start-run").disabled = true;
    document.getElementById("run-progress").style.display = "block";
    pollRun();
  } catch (e) {
    localStorage.removeItem("localbench-active-run");
  }
}

function pollRun() {
  clearTimeout(state.pollTimer);
  const poll = async () => {
    let data;
    try {
      data = await api(`/api/run/${state.activeRunId}/status`);
    } catch (e) {
      document.getElementById("run-log").textContent += `\n\n(lost connection to run: ${e.message})`;
      localStorage.removeItem("localbench-active-run");
      document.getElementById("start-run").disabled = false;
      return;
    }
    document.getElementById("run-log").textContent = data.log.join("\n");
    document.getElementById("run-log").scrollTop = document.getElementById("run-log").scrollHeight;
    renderLiveMonitor(data.resource_samples || []);

    const banner = document.getElementById("confirm-banner");
    if (data.status === "waiting_confirm") {
      banner.style.display = "flex";
      document.getElementById("confirm-message").textContent = data.pending_message;
    } else {
      banner.style.display = "none";
    }

    if (data.done) {
      localStorage.removeItem("localbench-active-run");
      document.getElementById("start-run").disabled = false;
      if (data.error) {
        document.getElementById("run-log").textContent += `\n\nERROR: ${data.error}`;
      } else {
        document.getElementById("run-log").textContent += `\n\nDone -- saved as run ${data.result_run_id}`;
      }
      return;
    }
    state.pollTimer = setTimeout(poll, 1500);
  };
  poll();
}

document.getElementById("confirm-continue").addEventListener("click", async () => {
  await api(`/api/run/${state.activeRunId}/continue`, { method: "POST" });
});

// ---------- Live Hardware Monitor ----------
const LIVE_METRICS = [
  { key: "cpu_percent", label: "CPU", unit: "%", fixedMax: 100 },
  { key: "ram_used_gb", label: "RAM", unit: "GB", totalKey: "ram_total_gb" },
  { key: "gpu_util_percent", label: "GPU", unit: "%", fixedMax: 100 },
  { key: "gpu_mem_used_gb", label: "GPU Memory", unit: "GB" },
  { key: "disk_read_mb_s", label: "Disk Read", unit: "MB/s" },
  { key: "disk_write_mb_s", label: "Disk Write", unit: "MB/s" },
];

function buildSparklinePath(values, width, height, yMax) {
  if (values.length < 2) return "";
  const stepX = width / (values.length - 1);
  return values
    .map((v, i) => {
      const x = i * stepX;
      const y = height - (Math.min(v, yMax) / yMax) * height;
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

function renderLiveMonitor(samples) {
  const grid = document.getElementById("live-monitor-grid");
  if (!grid) return;
  if (samples.length === 0) {
    grid.innerHTML = '<p class="muted">Waiting for first sample...</p>';
    return;
  }

  const width = 220;
  const height = 46;
  grid.innerHTML = "";

  LIVE_METRICS.forEach((metric) => {
    const values = samples.map((s) => s[metric.key]).filter((v) => v != null);
    if (values.length === 0) return;
    const current = values[values.length - 1];
    const yMax = metric.fixedMax ?? Math.max(...values, 0.001) * 1.15;

    const tile = document.createElement("div");
    tile.className = "live-monitor-tile";

    const header = document.createElement("div");
    header.className = "live-monitor-header";
    const labelEl = document.createElement("span");
    labelEl.className = "live-monitor-label";
    labelEl.textContent = metric.label;
    header.appendChild(labelEl);
    const valueEl = document.createElement("span");
    valueEl.className = "live-monitor-value";
    const total = metric.totalKey ? samples[samples.length - 1][metric.totalKey] : null;
    valueEl.textContent = total != null
      ? `${fmtNum(current, 1)}/${fmtNum(total, 1)} ${metric.unit}`
      : `${fmtNum(current, metric.unit === "%" ? 0 : 2)} ${metric.unit}`;
    header.appendChild(valueEl);
    tile.appendChild(header);

    const svgNs = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(svgNs, "svg");
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("class", "live-monitor-svg");
    const path = document.createElementNS(svgNs, "path");
    path.setAttribute("d", buildSparklinePath(values, width, height, total ?? yMax));
    path.setAttribute("class", "live-monitor-line");
    svg.appendChild(path);
    tile.appendChild(svg);

    grid.appendChild(tile);
  });
}

// ---------- History ----------
// Illustrative only -- NOT a live price lookup, and not tied to any specific
// current model. Token counts below are the REAL prompt_tokens/completion_tokens
// captured during these runs, never estimated; only the $/token rate is a
// placeholder, since we have no live pricing source. Change these two
// constants to match whatever cloud model you'd actually be comparing against.
const ILLUSTRATIVE_RATE_PER_M_INPUT = 3; // USD per 1M input tokens
const ILLUSTRATIVE_RATE_PER_M_OUTPUT = 15; // USD per 1M output tokens

function calculateApiSavings(runs) {
  let totalProblems = 0;
  let totalPromptTokens = 0;
  let totalCompletionTokens = 0;
  runs.forEach((r) => {
    if (!r.models) return;
    Object.values(r.models).forEach((m) => {
      if (!m.suites) return;
      Object.values(m.suites).forEach((s) => {
        (s.problems || []).forEach((p) => {
          totalProblems++;
          totalPromptTokens += p.prompt_tokens || 0;
          totalCompletionTokens += p.completion_tokens || 0;
        });
      });
    });
  });
  const cost =
    (totalPromptTokens / 1e6) * ILLUSTRATIVE_RATE_PER_M_INPUT +
    (totalCompletionTokens / 1e6) * ILLUSTRATIVE_RATE_PER_M_OUTPUT;
  return {
    totalProblems,
    totalPromptTokens,
    totalCompletionTokens,
    cost: cost.toFixed(2),
  };
}

function renderHistoryStats(runs) {
  const statsEl = document.getElementById("history-stats");
  if (runs.length === 0) {
    statsEl.innerHTML = "";
    return;
  }
  // Note: /api/runs intentionally returns lightweight summaries (model names
  // only, no per-suite/per-problem data) so listing history stays cheap even
  // with many saved runs. That means an illustrative-cost tile can't be
  // computed honestly here without fetching every run's full detail -- it
  // lives in the Compare view's export bar instead, where full run data is
  // already loaded for the selected runs.
  const distinctModels = new Set(runs.flatMap((r) => r.models));
  const latest = runs[0];

  statsEl.innerHTML = `
    <div class="stat-tile"><div class="value accent">${runs.length}</div><div class="label">Runs recorded</div></div>
    <div class="stat-tile"><div class="value">${distinctModels.size}</div><div class="label">Models tested</div></div>
    <div class="stat-tile"><div class="value" style="font-size:16px">${latest.started_at || latest.run_id}</div><div class="label">Latest run</div></div>
  `;
}

async function loadRunList() {
  const runs = await api("/api/runs");
  renderHistoryStats(runs);
  const container = document.getElementById("run-list");
  container.innerHTML = "";
  if (runs.length === 0) {
    container.innerHTML = '<p class="empty-state">No runs yet -- start one from the New Run tab.</p>';
    return;
  }
  runs.forEach((run) => {
    const item = document.createElement("div");
    item.className = "run-list-item";
    item.setAttribute("role", "button");
    item.setAttribute("tabindex", "0");
    const gpu = Array.isArray(run.hardware?.gpu) ? run.hardware.gpu[0]?.name : run.hardware?.gpu;

    const info = document.createElement("div");
    const idLine = document.createElement("div");
    idLine.className = "run-id";
    idLine.textContent = run.run_id;
    const metaLine = document.createElement("div");
    metaLine.className = "meta";
    metaLine.textContent = `${run.started_at || ""} · ${run.models.join(", ")} · ${gpu || "unknown GPU"}`;
    info.appendChild(idLine);
    info.appendChild(metaLine);
    item.appendChild(info);

    const deleteBtn = document.createElement("button");
    deleteBtn.className = "small delete-run-btn";
    deleteBtn.textContent = "Delete";
    deleteBtn.title = `Delete run ${run.run_id}`;
    deleteBtn.addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!confirm(`Delete run ${run.run_id}? This cannot be undone.`)) return;
      try {
        await api(`/api/runs/${run.run_id}`, { method: "DELETE" });
        state.selectedCompareRuns.delete(run.run_id);
        compareRunCache.delete(run.run_id);
        if (state.currentRunData?.run_id === run.run_id) {
          state.currentRunData = null;
          document.getElementById("run-detail-card").style.display = "none";
          document.getElementById("run-detail-body").innerHTML = "";
        }
        await loadRunList();
      } catch (err) {
        alert("Failed to delete run: " + err.message);
      }
    });
    item.appendChild(deleteBtn);

    item.addEventListener("click", () => showRunDetail(run.run_id));
    item.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        showRunDetail(run.run_id);
      }
    });
    container.appendChild(item);
  });
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str == null ? "" : String(str);
  return div.innerHTML;
}

function fmtNum(x, digits = 2) {
  return x === null || x === undefined ? "n/a" : Number(x).toFixed(digits);
}
function fmtPct(x) {
  return x === null || x === undefined ? "n/a" : Math.round(x * 100) + "%";
}

async function showRunDetail(runId) {
  const data = await api(`/api/runs/${runId}`);
  state.currentRunData = data;
  document.getElementById("run-detail-card").style.display = "block";
  document.getElementById("run-detail-title").textContent = `Run ${runId}`;
  document.getElementById("run-detail-links").innerHTML = `
    <a href="/api/runs/${runId}/report.md" target="_blank">Markdown</a>
    <a href="/api/runs/${runId}/report.pdf" target="_blank">PDF</a>
    <a href="/api/runs/${runId}/raw.json" target="_blank">Raw JSON</a>`;

  const suiteNames = new Set();
  Object.values(data.models).forEach((m) => Object.keys(m.suites).forEach((s) => suiteNames.add(s)));
  // Model/suite names come from the runtime and from saved run files, so they
  // are escaped rather than interpolated raw into option markup.
  const suiteFilter = document.getElementById("suite-filter");
  suiteFilter.innerHTML = '<option value="">All suites</option>' +
    Array.from(suiteNames).map((s) => `<option value="${escapeHtml(s)}">${escapeHtml(s)}</option>`).join("");
  const modelFilter = document.getElementById("model-filter");
  modelFilter.innerHTML = '<option value="">All models</option>' +
    Object.keys(data.models).map((m) => `<option value="${escapeHtml(m)}">${escapeHtml(m)}</option>`).join("");

  suiteFilter.onchange = () => renderRunDetail(data, suiteFilter.value, modelFilter.value);
  modelFilter.onchange = () => renderRunDetail(data, suiteFilter.value, modelFilter.value);
  renderRunDetail(data, "", "");
}

function renderRunDetail(data, suiteFilterVal, modelFilterVal) {
  const suiteNames = new Set();
  Object.values(data.models).forEach((m) => Object.keys(m.suites).forEach((s) => suiteNames.add(s)));

  let html = "";
  suiteNames.forEach((suiteName) => {
    if (suiteFilterVal && suiteFilterVal !== suiteName) return;
    html += `<h3>${escapeHtml(suiteName)}</h3><table><tr>
      <th>Model</th><th>Pass Rate (Click for inspection)</th><th>Avg Latency</th><th>Avg TTFT</th>
      <th>Tokens/sec</th>
      <th title="Total GPU memory in use at this suite's peak -- the model's actual VRAM footprint.">VRAM used &#9432;</th>
      <th title="Growth in GPU memory during this suite only. Small is normal: the model is already loaded before the suite starts, so this mostly reflects KV-cache growth, not the model's size.">VRAM &Delta; &#9432;</th>
      <th title="Peak GPU utilization sampled during this suite.">GPU %</th>
      <th title="Growth in TOTAL SYSTEM RAM during this suite, over a baseline taken just before it started. Not model-attributable -- a small number here does NOT mean the model is small, since a GPU-resident model uses VRAM, not RAM.">RAM &Delta; &#9432;</th></tr>`;
    Object.entries(data.models).forEach(([modelName, modelResult]) => {
      if (modelFilterVal && modelFilterVal !== modelName) return;
      const suite = modelResult.suites[suiteName];
      if (!suite) return;
      const resource = suite.resource_usage || {};
      const vram = resource.vram_delta_mb;
      const vramTotal = resource.peak_vram_mb_total;
      const gpuUtil = resource.peak_gpu_util_percent;
      const pillClass = suite.pass_rate >= 0.7 ? "pass" : "fail";
      const icon = suite.pass_rate >= 0.7 ? "✓" : "✗";
      const truncatedCount = (suite.problems || []).filter((p) => p.truncated).length;
      const truncatedBadge = truncatedCount > 0
        ? `<span class="truncated-badge" title="${truncatedCount} of ${suite.total} failure(s) hit the max_tokens limit before finishing -- counted as failures, but may reflect too small a token budget rather than incorrect reasoning. Click the pill to inspect.">&#9888; ${truncatedCount} truncated</span>`
        : "";
      html += `<tr>
        <td>${escapeHtml(modelName)}</td>
        <td>
          <span class="pill ${pillClass} clickable" data-run-id="${escapeHtml(data.run_id)}" data-model-name="${escapeHtml(modelName)}" data-suite-name="${escapeHtml(suiteName)}" title="Click to view detailed prompt/response/failure breakdown">
            <span class="icon">${icon}</span>${fmtPct(suite.pass_rate)} (${suite.pass_count}/${suite.total})
          </span>${truncatedBadge}
        </td>
        <td>${fmtNum(suite.avg_latency_seconds)}s</td>
        <td>${fmtNum(suite.avg_ttft_seconds)}s</td>
        <td>${fmtNum(suite.avg_tokens_per_sec)}</td>
        <td>${vramTotal == null ? '<span class="muted">not captured</span>' : fmtNum(vramTotal / 1024, 2) + " GB"}</td>
        <td>${vram == null ? '<span class="muted">not captured</span>' : fmtNum(vram, 0) + " MB"}</td>
        <td>${gpuUtil == null ? '<span class="muted">n/a</span>' : fmtNum(gpuUtil, 0) + "%"}</td>
        <td>${fmtNum(resource.ram_delta_gb, 2)} GB</td>
      </tr>`;
    });
    html += "</table>";
  });
  const detailBody = document.getElementById("run-detail-body");
  detailBody.innerHTML = html;
  detailBody.querySelectorAll(".pill.clickable[data-run-id]").forEach((el) => {
    el.addEventListener("click", () =>
      openInspectorModalByData(el.dataset.runId, el.dataset.modelName, el.dataset.suiteName)
    );
  });
}

// ---------- Problem Failure Inspector Modal ----------
function openInspectorModalByData(runId, modelName, suiteName) {
  let runData = state.currentRunData;
  if (!runData || runData.run_id !== runId) {
    if (compareRunCache.has(runId)) {
      runData = compareRunCache.get(runId);
    }
  }
  if (!runData) {
    alert(`Run data for ${runId} is loading, please try again.`);
    return;
  }
  showInspectorModal(runData, modelName, suiteName);
}

function showInspectorModal(runData, modelName, suiteName, filterState = "all") {
  const modal = document.getElementById("inspector-modal");
  const title = document.getElementById("inspector-title");
  const summary = document.getElementById("inspector-summary");
  const chipsContainer = document.getElementById("inspector-filter-chips");
  const body = document.getElementById("inspector-body");

  const modelResult = runData.models[modelName];
  if (!modelResult || !modelResult.suites[suiteName]) {
    alert("Suite result not found.");
    return;
  }
  const suiteResult = modelResult.suites[suiteName];
  const problems = suiteResult.problems || [];

  title.textContent = `Inspector: ${modelName} — ${suiteName}`;
  summary.textContent = `Run ${runData.run_id} · ${suiteResult.pass_count}/${suiteResult.total} passed (${fmtPct(suiteResult.pass_rate)})`;

  chipsContainer.innerHTML = "";
  const filterOptions = [
    { id: "all", label: `All (${problems.length})` },
    { id: "failed", label: `Failed (${suiteResult.total - suiteResult.pass_count})` },
    { id: "passed", label: `Passed (${suiteResult.pass_count})` },
  ];
  filterOptions.forEach((opt) => {
    const chip = document.createElement("label");
    chip.className = "chip" + (filterState === opt.id ? " checked" : "");
    chip.textContent = opt.label;
    chip.addEventListener("click", () => showInspectorModal(runData, modelName, suiteName, opt.id));
    chipsContainer.appendChild(chip);
  });

  body.innerHTML = "";
  let visibleCount = 0;
  problems.forEach((p, idx) => {
    const isPass = p.passed === true;
    if (filterState === "failed" && isPass) return;
    if (filterState === "passed" && !isPass) return;
    visibleCount++;

    const card = document.createElement("div");
    card.className = `problem-card ${isPass ? "pass" : "fail"}`;

    const problemId = p.problem_id || `Task #${idx + 1}`;
    const header = document.createElement("div");
    header.className = "problem-header";
    header.innerHTML = `
      <div class="problem-title">${problemId}</div>
      <span class="pill ${isPass ? "pass" : "fail"}"><span class="icon">${isPass ? "✓" : "✗"}</span>${isPass ? "PASSED" : "FAILED"}</span>
    `;
    card.appendChild(header);

    const meta = document.createElement("div");
    meta.className = "problem-meta";
    meta.innerHTML = `
      <span>Latency: ${fmtNum(p.latency_seconds)}s</span>
      <span>TTFT: ${fmtNum(p.ttft_seconds)}s</span>
      <span>Tokens/sec: ${fmtNum(p.tokens_per_sec)}</span>
      ${p.truncated ? '<span style="color:var(--serious)">TRUNCATED</span>' : ""}
    `;
    card.appendChild(meta);

    if (p.score != null) {
      const scoreLabel = document.createElement("div");
      scoreLabel.className = "problem-section-title";
      scoreLabel.textContent = "Judge Score";
      card.appendChild(scoreLabel);
      const scoreBox = document.createElement("div");
      scoreBox.className = "problem-box";
      scoreBox.textContent = `${p.score}/10`;
      card.appendChild(scoreBox);
    }

    if (p.rationale) {
      const ratLabel = document.createElement("div");
      ratLabel.className = "problem-section-title";
      ratLabel.textContent = "Judge Rationale (what it said about this response)";
      card.appendChild(ratLabel);
      const ratBox = document.createElement("div");
      ratBox.className = "problem-box";
      ratBox.textContent = p.rationale;
      card.appendChild(ratBox);
    }

    if (p.prompt) {
      const pLabel = document.createElement("div");
      pLabel.className = "problem-section-title";
      pLabel.textContent = "Prompt";
      card.appendChild(pLabel);
      const pBox = document.createElement("div");
      pBox.className = "problem-box";
      pBox.textContent = typeof p.prompt === "string" ? p.prompt : JSON.stringify(p.prompt, null, 2);
      card.appendChild(pBox);
    }

    const errorText = p.error || p.failure_reason || p.error_message;
    if (errorText && errorText !== p.rationale) {
      const errLabel = document.createElement("div");
      errLabel.className = "problem-section-title";
      errLabel.textContent = "Error / Evaluation Failure";
      card.appendChild(errLabel);
      const errBox = document.createElement("div");
      errBox.className = "problem-box error";
      errBox.textContent = errorText;
      card.appendChild(errBox);
    }

    if (p.response_content || p.response || p.output) {
      const rLabel = document.createElement("div");
      rLabel.className = "problem-section-title";
      rLabel.textContent = "Model Response";
      card.appendChild(rLabel);
      const rBox = document.createElement("div");
      rBox.className = "problem-box";
      rBox.textContent = p.response_content || p.response || p.output || "(empty)";
      card.appendChild(rBox);
    }

    body.appendChild(card);
  });

  if (visibleCount === 0) {
    body.innerHTML = `<p class="empty-state">No problems match filter "${filterState}".</p>`;
  }

  modal.style.display = "flex";
}

document.getElementById("inspector-close").addEventListener("click", () => {
  document.getElementById("inspector-modal").style.display = "none";
});
document.getElementById("inspector-modal").addEventListener("click", (e) => {
  if (e.target.id === "inspector-modal") {
    document.getElementById("inspector-modal").style.display = "none";
  }
});

// ---------- Playground ----------
async function populatePlaygroundModels() {
  if (!state.config) await loadConfig();
  const select = document.getElementById("pg-model");
  let models = state.config.models;
  try {
    const detected = await api("/api/models/detect");
    if (detected.models?.length) models = detected.models;
  } catch (e) {
  }
  select.innerHTML = models
    .map((m) => `<option value="${escapeHtml(m)}">${escapeHtml(m)}</option>`)
    .join("");
}

document.getElementById("pg-run").addEventListener("click", async () => {
  const status = document.getElementById("pg-status");
  const resultBox = document.getElementById("pg-result");
  status.textContent = "running...";
  resultBox.style.display = "none";

  const model = document.getElementById("pg-model").value;
  const prompt = document.getElementById("pg-prompt").value;
  const schemaText = document.getElementById("pg-schema").value.trim();
  let schema = null;
  if (schemaText) {
    try {
      schema = JSON.parse(schemaText);
    } catch (e) {
      status.textContent = "invalid JSON schema: " + e.message;
      return;
    }
  }

  try {
    const data = await api("/api/custom", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, prompt, schema }),
    });
    status.textContent = data.success ? "done" : "call failed";
    document.getElementById("pg-content").textContent = data.success
      ? data.content || "(empty response)"
      : data.error;
    let stats = `latency: ${fmtNum(data.latency_seconds)}s | ttft: ${fmtNum(data.ttft_seconds)}s | tok/s: ${fmtNum(data.tokens_per_sec)}`;
    if (data.truncated) stats += " | TRUNCATED (raise max_tokens)";
    if ("schema_valid" in data) {
      stats += data.schema_valid
        ? " | schema: valid"
        : ` | schema: invalid (${data.schema_error})`;
    }
    document.getElementById("pg-stats").textContent = stats;
    resultBox.style.display = "block";
  } catch (e) {
    status.textContent = "error: " + e.message;
  }
});

// ---------- Compare ----------
const SERIES_COLORS = ["--series-1", "--series-2", "--series-3", "--series-4"];
const compareRunCache = new Map();
const DETERMINISTIC_SUITE_ORDER = [
  "json_schema", "coding", "logic_math", "instruction_following", "pattern_reasoning", "long_context",
];

function seriesColor(index) {
  const varName = SERIES_COLORS[index % SERIES_COLORS.length];
  return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
}

async function loadComparePicker() {
  const runs = await api("/api/runs");
  const picker = document.getElementById("compare-run-picker");
  if (runs.length === 0) {
    picker.innerHTML = '<p class="empty-state">No runs yet -- start one from the New Run tab.</p>';
    return;
  }
  picker.innerHTML = "";
  runs.forEach((run) => {
    const label = document.createElement("label");
    label.className = "chip" + (state.selectedCompareRuns.has(run.run_id) ? " checked" : "");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = state.selectedCompareRuns.has(run.run_id);
    cb.addEventListener("change", () => {
      if (cb.checked) state.selectedCompareRuns.add(run.run_id);
      else state.selectedCompareRuns.delete(run.run_id);
      label.classList.toggle("checked", cb.checked);
      renderCompare();
    });
    label.appendChild(cb);
    label.appendChild(
      document.createTextNode(`${formatRunDate(run.started_at)} — ${run.models.join(", ")}`)
    );
    picker.appendChild(label);
  });
  renderCompare();
}

function roundedTopBarPath(x, y, w, h, r) {
  r = Math.min(r, w / 2, Math.max(h, 0));
  if (h <= 0) return "";
  return `M${x},${y + h} L${x},${y + r} Q${x},${y} ${x + r},${y} L${x + w - r},${y} ` +
    `Q${x + w},${y} ${x + w},${y + r} L${x + w},${y + h} Z`;
}

function buildBarChart({ title, categories, series, getValue, formatValue, yMax, yTicks }) {
  const wrap = document.createElement("div");
  wrap.className = "card";

  const heading = document.createElement("h2");
  heading.textContent = title;
  wrap.appendChild(heading);

  if (series.length > 1) {
    const legend = document.createElement("div");
    legend.className = "chart-legend";
    series.forEach((s, i) => {
      const item = document.createElement("div");
      item.className = "item";
      const swatch = document.createElement("span");
      swatch.className = "swatch";
      swatch.style.background = seriesColor(i);
      item.appendChild(swatch);
      const label = document.createElement("span");
      label.textContent = s.label;
      item.appendChild(label);
      legend.appendChild(item);
    });
    wrap.appendChild(legend);
  }

  const chartWrap = document.createElement("div");
  chartWrap.className = "chart-wrap";
  const width = 900, height = 260;
  const marginLeft = 42, marginBottom = 34, marginTop = 10, marginRight = 10;
  const plotW = width - marginLeft - marginRight;
  const plotH = height - marginTop - marginBottom;

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("class", "chart-svg");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", title);

  yTicks.forEach((tickVal) => {
    const y = marginTop + plotH - (tickVal / yMax) * plotH;
    const line = document.createElementNS(svg.namespaceURI, "line");
    line.setAttribute("x1", marginLeft);
    line.setAttribute("x2", width - marginRight);
    line.setAttribute("y1", y);
    line.setAttribute("y2", y);
    line.setAttribute("class", "gridline");
    svg.appendChild(line);
    const label = document.createElementNS(svg.namespaceURI, "text");
    label.setAttribute("x", marginLeft - 8);
    label.setAttribute("y", y + 3);
    label.setAttribute("text-anchor", "end");
    label.setAttribute("font-size", "10.5");
    label.textContent = formatValue(tickVal);
    svg.appendChild(label);
  });

  const bandWidth = plotW / categories.length;
  const barGap = 2;
  const barWidth = Math.min(24, (bandWidth - 8 - (series.length - 1) * barGap) / series.length);
  const groupWidth = barWidth * series.length + barGap * (series.length - 1);

  categories.forEach((cat, ci) => {
    const bandX = marginLeft + ci * bandWidth;
    const groupStart = bandX + (bandWidth - groupWidth) / 2;

    const catLabel = document.createElementNS(svg.namespaceURI, "text");
    catLabel.setAttribute("x", bandX + bandWidth / 2);
    catLabel.setAttribute("y", height - marginBottom + 18);
    catLabel.setAttribute("text-anchor", "middle");
    catLabel.setAttribute("class", "cat-label");
    catLabel.textContent = cat.length > 14 ? cat.slice(0, 13) + "…" : cat;
    svg.appendChild(catLabel);

    series.forEach((s, si) => {
      const value = getValue(s, cat);
      const x = groupStart + si * (barWidth + barGap);
      const h = value == null ? 0 : (value / yMax) * plotH;
      const y = marginTop + plotH - h;

      const path = document.createElementNS(svg.namespaceURI, "path");
      path.setAttribute("d", roundedTopBarPath(x, y, barWidth, h, 4));
      path.setAttribute("fill", value == null ? "var(--border)" : seriesColor(si));
      path.setAttribute("class", "bar");
      path.setAttribute("tabindex", value == null ? "-1" : "0");
      if (value != null) {
        const showTip = (evt) => showChartTooltip(chartWrap, evt, `${cat} (${s.label})`, formatValue(value));
        path.addEventListener("mousemove", showTip);
        path.addEventListener("mouseleave", () => hideChartTooltip(chartWrap));
        path.addEventListener("focus", (evt) => showChartTooltip(chartWrap, evt, `${cat} (${s.label})`, formatValue(value)));
        path.addEventListener("blur", () => hideChartTooltip(chartWrap));
      }
      svg.appendChild(path);
    });
  });

  chartWrap.appendChild(svg);
  const tooltip = document.createElement("div");
  tooltip.className = "chart-tooltip";
  chartWrap.appendChild(tooltip);
  wrap.appendChild(chartWrap);
  return wrap;
}

// ---------- Pareto Frontier Scatter Plot ----------
function buildParetoChart(series, categories, getSuite) {
  const wrap = document.createElement("div");
  wrap.className = "card";

  const heading = document.createElement("h2");
  heading.textContent = "Pareto Efficiency Frontier (Accuracy vs Speed)";
  wrap.appendChild(heading);

  const desc = document.createElement("p");
  desc.className = "muted";
  desc.textContent = "Models towards top-right offer optimal accuracy and speed. The dashed line marks the Pareto frontier.";
  wrap.appendChild(desc);

  const points = series.map((s, i) => {
    let totalPass = 0, totalCount = 0, totalTokSec = 0, suiteCount = 0;
    categories.forEach((cat) => {
      const suite = getSuite(s, cat);
      if (suite) {
        totalPass += suite.pass_count;
        totalCount += suite.total;
        if (suite.avg_tokens_per_sec) {
          totalTokSec += suite.avg_tokens_per_sec;
          suiteCount++;
        }
      }
    });
    const avgPassRate = totalCount > 0 ? (totalPass / totalCount) * 100 : 0;
    const avgSpeed = suiteCount > 0 ? totalTokSec / suiteCount : 0;
    return { seriesItem: s, index: i, passRate: avgPassRate, speed: avgSpeed };
  });

  const chartWrap = document.createElement("div");
  chartWrap.className = "chart-wrap";
  const width = 900, height = 280;
  const marginLeft = 48, marginBottom = 40, marginTop = 20, marginRight = 30;
  const plotW = width - marginLeft - marginRight;
  const plotH = height - marginTop - marginBottom;

  const maxSpeed = Math.max(10, ...points.map((p) => p.speed * 1.15));

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("class", "chart-svg");

  [0, 25, 50, 75, 100].forEach((tickVal) => {
    const y = marginTop + plotH - (tickVal / 100) * plotH;
    const line = document.createElementNS(svg.namespaceURI, "line");
    line.setAttribute("x1", marginLeft);
    line.setAttribute("x2", width - marginRight);
    line.setAttribute("y1", y);
    line.setAttribute("y2", y);
    line.setAttribute("class", "gridline");
    svg.appendChild(line);

    const label = document.createElementNS(svg.namespaceURI, "text");
    label.setAttribute("x", marginLeft - 8);
    label.setAttribute("y", y + 4);
    label.setAttribute("text-anchor", "end");
    label.setAttribute("font-size", "10.5");
    label.textContent = `${tickVal}%`;
    svg.appendChild(label);
  });

  const xLabel = document.createElementNS(svg.namespaceURI, "text");
  xLabel.setAttribute("x", marginLeft + plotW / 2);
  xLabel.setAttribute("y", height - 8);
  xLabel.setAttribute("text-anchor", "middle");
  xLabel.setAttribute("font-size", "11.5");
  xLabel.setAttribute("class", "cat-label");
  xLabel.textContent = "Avg Speed (Tokens/sec) →";
  svg.appendChild(xLabel);

  // Pareto-optimal for "maximize both speed and accuracy": scan fastest to
  // slowest, keep a point only if no faster-or-equal point already matches
  // or beats its accuracy -- i.e. it's not dominated by anything faster.
  // (Scanning the other direction, as this originally did, only catches
  // points with monotonically-increasing accuracy as speed *decreases* and
  // silently drops genuinely optimal fast+accurate points along the way.)
  const sortedPoints = [...points].sort((a, b) => b.speed - a.speed);
  const frontier = [];
  let maxAcc = -1;
  sortedPoints.forEach((p) => {
    if (p.passRate > maxAcc) {
      frontier.push(p);
      maxAcc = p.passRate;
    }
  });
  frontier.sort((a, b) => a.speed - b.speed); // left-to-right for the drawn line

  if (frontier.length > 1) {
    const dStr = frontier.map((p, i) => {
      const cx = marginLeft + (p.speed / maxSpeed) * plotW;
      const cy = marginTop + plotH - (p.passRate / 100) * plotH;
      return `${i === 0 ? "M" : "L"}${cx},${cy}`;
    }).join(" ");

    const pLine = document.createElementNS(svg.namespaceURI, "path");
    pLine.setAttribute("d", dStr);
    pLine.setAttribute("class", "pareto-line");
    svg.appendChild(pLine);
  }

  points.forEach((p) => {
    const cx = marginLeft + (p.speed / maxSpeed) * plotW;
    const cy = marginTop + plotH - (p.passRate / 100) * plotH;

    const circle = document.createElementNS(svg.namespaceURI, "circle");
    circle.setAttribute("cx", cx);
    circle.setAttribute("cy", cy);
    circle.setAttribute("r", "6");
    circle.setAttribute("fill", seriesColor(p.index));
    circle.setAttribute("class", "scatter-point");

    const textLabel = document.createElementNS(svg.namespaceURI, "text");
    textLabel.setAttribute("x", cx + 9);
    textLabel.setAttribute("y", cy + 4);
    textLabel.setAttribute("font-size", "11");
    textLabel.setAttribute("font-weight", "600");
    textLabel.setAttribute("fill", seriesColor(p.index));
    textLabel.textContent = p.seriesItem.modelName;
    svg.appendChild(textLabel);

    const showTip = (evt) => showChartTooltip(chartWrap, evt, `${p.seriesItem.modelName}`, `${p.passRate.toFixed(1)}% accuracy @ ${p.speed.toFixed(1)} tok/s`);
    circle.addEventListener("mousemove", showTip);
    circle.addEventListener("mouseleave", () => hideChartTooltip(chartWrap));

    svg.appendChild(circle);
  });

  chartWrap.appendChild(svg);
  const tooltip = document.createElement("div");
  tooltip.className = "chart-tooltip";
  chartWrap.appendChild(tooltip);
  wrap.appendChild(chartWrap);
  return wrap;
}

// ---------- TTFT vs Total Latency Chart ----------
function buildTTFTChart(series, categories, getSuite) {
  const maxTtft = Math.max(5, ...series.flatMap((s) => categories.map((cat) => getSuite(s, cat)?.avg_ttft_seconds || 0)));
  const yMax = Math.ceil(maxTtft / 5) * 5 || 5;

  return buildBarChart({
    title: "Time-to-First-Token (TTFT Latency Overhead)",
    categories,
    series,
    getValue: (s, cat) => getSuite(s, cat)?.avg_ttft_seconds ?? null,
    formatValue: (v) => `${fmtNum(v, 2)}s TTFT`,
    yMax: yMax,
    yTicks: [0, Math.round(yMax / 2), yMax],
  });
}

function showChartTooltip(chartWrap, evt, label, valueText) {
  const tooltip = chartWrap.querySelector(".chart-tooltip");
  const wrapRect = chartWrap.getBoundingClientRect();
  const targetRect = evt.target.getBoundingClientRect();
  tooltip.innerHTML = "";
  const valSpan = document.createElement("span");
  valSpan.className = "val";
  valSpan.textContent = valueText;
  tooltip.appendChild(valSpan);
  tooltip.appendChild(document.createTextNode(" " + label));
  const left = targetRect.left - wrapRect.left + (targetRect.width || 12) / 2;
  const top = Math.max(20, targetRect.top - wrapRect.top - 8);
  tooltip.style.left = `${Math.min(Math.max(left, 60), wrapRect.width - 60)}px`;
  tooltip.style.top = `${top}px`;
  tooltip.classList.add("visible");
}
function hideChartTooltip(chartWrap) {
  chartWrap.querySelector(".chart-tooltip").classList.remove("visible");
}

function formatRunDate(isoStr) {
  if (!isoStr) return "unknown";
  const d = new Date(isoStr);
  return isNaN(d) ? isoStr : d.toLocaleString(undefined, {
    year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  });
}
function formatGpu(gpu) {
  if (Array.isArray(gpu)) {
    return gpu.map((g) => `${g.name} (${g.memory || g.memory_approx || "unknown"})`).join(", ");
  }
  return gpu || "unknown";
}
function buildModelDetailsTable(series, runs) {
  const wrap = document.createElement("div");
  wrap.className = "card";
  const heading = document.createElement("h2");
  heading.textContent = "Models & hardware compared";
  wrap.appendChild(heading);

  const table = document.createElement("table");
  table.innerHTML = `<tr>
    <th>Model</th><th>Run date</th><th>Size on disk</th><th>Quantization</th>
    <th>Context length</th><th>CPU</th><th>RAM</th><th>GPU</th>
  </tr>`;
  series.forEach((s) => {
    const run = runs.find((r) => r.run_id === s.runId);
    const info = run?.models[s.modelName]?.runtime_load_info || {};
    const hw = run?.hardware || {};
    const sizeStr = info.size_bytes ? `${(info.size_bytes / 1024 ** 3).toFixed(2)} GB` : "unknown";
    const quantStr = info.quantization?.name || "unknown";
    const ctxStr = info.context_length ?? "unknown";
    const cpuStr = hw.cpu ? `${hw.cpu} (${hw.cpu_count_physical ?? "?"}c/${hw.cpu_count_logical ?? "?"}t)` : "unknown";
    const ramStr = hw.ram_total_gb ? `${hw.ram_total_gb} GB` : "unknown";
    const row = document.createElement("tr");
    const cells = [
      s.modelName, formatRunDate(run?.started_at), sizeStr, quantStr, ctxStr, cpuStr, ramStr, formatGpu(hw.gpu),
    ];
    cells.forEach((text) => {
      const td = document.createElement("td");
      td.textContent = text;
      row.appendChild(td);
    });
    table.appendChild(row);
  });
  wrap.appendChild(table);
  return wrap;
}

// ---------- Full Comparison Table with Sorting, Search & Inspector Pills ----------
function buildFullComparisonTable(series, categories, getSuite, runs) {
  const wrap = document.createElement("div");
  wrap.className = "card";

  const headerRow = document.createElement("div");
  headerRow.className = "table-controls";
  
  const heading = document.createElement("h2");
  heading.textContent = "All numbers & failure inspector";
  heading.style.margin = "0";
  headerRow.appendChild(heading);

  const searchInput = document.createElement("input");
  searchInput.type = "text";
  searchInput.placeholder = "Filter by model or suite...";
  searchInput.value = state.compareSearchQuery;
  searchInput.addEventListener("input", (e) => {
    state.compareSearchQuery = e.target.value;
    renderCompare();
  });
  headerRow.appendChild(searchInput);

  wrap.appendChild(headerRow);

  const note = document.createElement("p");
  note.className = "muted";
  note.textContent = "Click column headers to sort. Click any Pass Rate pill to inspect problem prompts, responses & errors.";
  wrap.appendChild(note);

  let rowsData = [];
  series.forEach((s) => {
    categories.forEach((cat) => {
      const suite = getSuite(s, cat);
      if (!suite) return;
      const resource = suite.resource_usage || {};
      rowsData.push({
        series: s,
        modelName: s.modelName,
        runId: s.runId,
        suiteName: cat,
        passRate: suite.pass_rate,
        passCount: suite.pass_count,
        total: suite.total,
        latency: suite.avg_latency_seconds,
        ttft: suite.avg_ttft_seconds,
        tokSec: suite.avg_tokens_per_sec,
        ramDelta: resource.ram_delta_gb,
        vramDelta: resource.vram_delta_mb,
        vramTotal: resource.peak_vram_mb_total,
        gpuUtil: resource.peak_gpu_util_percent,
        peakCpu: resource.peak_cpu_percent,
        suiteData: suite,
      });
    });
  });

  if (state.compareSearchQuery) {
    const q = state.compareSearchQuery.toLowerCase();
    rowsData = rowsData.filter((r) => r.modelName.toLowerCase().includes(q) || r.suiteName.toLowerCase().includes(q) || r.runId.toLowerCase().includes(q));
  }

  if (state.compareSortCol) {
    const col = state.compareSortCol;
    const dir = state.compareSortDir === "asc" ? 1 : -1;
    rowsData.sort((a, b) => {
      let valA = a[col], valB = b[col];
      if (valA == null) return 1;
      if (valB == null) return -1;
      if (typeof valA === "string") return valA.localeCompare(valB) * dir;
      return (valA - valB) * dir;
    });
  }

  const table = document.createElement("table");

  const cols = [
    { id: "modelName", name: "Model" },
    { id: "runId", name: "Run" },
    { id: "suiteName", name: "Suite" },
    { id: "passRate", name: "Pass Rate" },
    { id: "latency", name: "Avg Latency" },
    { id: "ttft", name: "Avg TTFT" },
    { id: "tokSec", name: "Tokens/sec" },
    { id: "vramTotal", name: "VRAM used" },
    { id: "vramDelta", name: "VRAM Δ" },
    { id: "gpuUtil", name: "GPU %" },
    { id: "ramDelta", name: "RAM Δ" },
    { id: "peakCpu", name: "Peak CPU %" },
  ];

  const headRow = document.createElement("tr");
  cols.forEach((c) => {
    const isSorted = state.compareSortCol === c.id;
    const icon = isSorted ? (state.compareSortDir === "asc" ? " ▲" : " ▼") : "";
    const th = document.createElement("th");
    th.className = "sortable-th";
    th.addEventListener("click", () => handleCompareSort(c.id));
    th.appendChild(document.createTextNode(c.name));
    const sortIcon = document.createElement("span");
    sortIcon.className = "sort-icon";
    sortIcon.textContent = icon;
    th.appendChild(sortIcon);
    headRow.appendChild(th);
  });
  table.appendChild(headRow);

  rowsData.forEach((r) => {
    const vram = r.vramDelta;
    // Hedged, not asserted: a high RAM delta alongside low tok/s is a
    // plausible sign of CPU/VRAM spillover, but these thresholds are
    // uncalibrated -- a large model's normal KV-cache growth or a slow
    // model that still fits in VRAM can look the same. Never presented as
    // a confirmed diagnosis, only a hint to check the runtime's own UI.
    const possibleSpill = r.ramDelta > 2.0 && r.tokSec < 15;

    const row = document.createElement("tr");

    const tdModel = document.createElement("td");
    tdModel.textContent = r.modelName;
    row.appendChild(tdModel);

    const tdRun = document.createElement("td");
    tdRun.textContent = r.runId;
    row.appendChild(tdRun);

    const tdSuite = document.createElement("td");
    tdSuite.textContent = r.suiteName;
    row.appendChild(tdSuite);

    const tdPass = document.createElement("td");
    const pill = document.createElement("span");
    pill.className = `pill ${r.passRate >= 0.7 ? "pass" : "fail"} clickable`;
    pill.title = "Click to inspect problems";
    pill.addEventListener("click", () => openInspectorModalByData(r.runId, r.modelName, r.suiteName));
    const icon = document.createElement("span");
    icon.className = "icon";
    icon.textContent = r.passRate >= 0.7 ? "✓" : "✗";
    pill.appendChild(icon);
    pill.appendChild(document.createTextNode(`${fmtPct(r.passRate)} (${r.passCount}/${r.total})`));
    tdPass.appendChild(pill);
    const truncatedCount = (r.suiteData.problems || []).filter((p) => p.truncated).length;
    if (truncatedCount > 0) {
      const truncBadge = document.createElement("span");
      truncBadge.className = "truncated-badge";
      truncBadge.title = `${truncatedCount} of ${r.total} failure(s) hit the max_tokens limit before finishing -- counted as failures, but may reflect too small a token budget rather than incorrect reasoning. Click the pill to inspect.`;
      truncBadge.textContent = `⚠ ${truncatedCount} truncated`;
      tdPass.appendChild(truncBadge);
    }
    row.appendChild(tdPass);

    const tdLatency = document.createElement("td");
    tdLatency.textContent = `${fmtNum(r.latency)}s`;
    row.appendChild(tdLatency);

    const tdTtft = document.createElement("td");
    tdTtft.textContent = `${fmtNum(r.ttft)}s`;
    row.appendChild(tdTtft);

    const tdTokSec = document.createElement("td");
    tdTokSec.appendChild(document.createTextNode(fmtNum(r.tokSec)));
    if (possibleSpill) {
      tdTokSec.appendChild(document.createTextNode(" "));
      const spillBadge = document.createElement("span");
      spillBadge.className = "spill-badge";
      spillBadge.title =
        "Uncalibrated heuristic (RAM Δ > 2GB and < 15 tok/s) -- a plausible sign of CPU/VRAM " +
        "spillover, not a confirmed diagnosis. Check the runtime's own UI (e.g. LM Studio's Developer tab) to confirm.";
      spillBadge.textContent = "possible spillover";
      tdTokSec.appendChild(spillBadge);
    }
    row.appendChild(tdTokSec);

    const tdVramTotal = document.createElement("td");
    tdVramTotal.textContent = r.vramTotal == null ? "not captured" : `${fmtNum(r.vramTotal / 1024, 2)} GB`;
    tdVramTotal.title = "Total GPU memory in use at this suite's peak -- the model's actual VRAM footprint.";
    row.appendChild(tdVramTotal);

    const tdVram = document.createElement("td");
    tdVram.textContent = vram == null ? "not captured" : `${fmtNum(vram, 0)} MB`;
    tdVram.title = "Growth during this suite only. Small is normal -- the model is already loaded before the suite starts.";
    row.appendChild(tdVram);

    const tdGpuUtil = document.createElement("td");
    tdGpuUtil.textContent = r.gpuUtil == null ? "n/a" : `${fmtNum(r.gpuUtil, 0)}%`;
    row.appendChild(tdGpuUtil);

    const tdRam = document.createElement("td");
    tdRam.textContent = `${fmtNum(r.ramDelta, 2)} GB`;
    tdRam.title = "Growth in TOTAL SYSTEM RAM, not model-attributable. A GPU-resident model uses VRAM, not RAM.";
    row.appendChild(tdRam);

    const tdCpu = document.createElement("td");
    tdCpu.textContent = `${fmtNum(r.peakCpu, 1)}%`;
    row.appendChild(tdCpu);

    table.appendChild(row);
  });

  wrap.appendChild(table);
  return wrap;
}

function handleCompareSort(colId) {
  if (state.compareSortCol === colId) {
    state.compareSortDir = state.compareSortDir === "asc" ? "desc" : "asc";
  } else {
    state.compareSortCol = colId;
    state.compareSortDir = "asc";
  }
  renderCompare();
}

// ---------- Compare Exports (Markdown & CSV) ----------
function exportCompareMarkdown(series, categories, runs) {
  let md = `# Localbench Comparison Report\n\nGenerated: ${new Date().toLocaleString()}\n\n`;
  md += `## Models & Hardware Compared\n\n`;
  md += `| Model | Run | CPU | RAM | GPU |\n|---|---|---|---|---|\n`;
  series.forEach((s) => {
    const run = runs.find((r) => r.run_id === s.runId);
    const hw = run?.hardware || {};
    md += `| ${s.modelName} | ${s.runId} | ${hw.cpu || "unknown"} | ${hw.ram_total_gb || "?"} GB | ${formatGpu(hw.gpu)} |\n`;
  });

  md += `\n## Evaluation Results\n\n`;
  md += `| Model | Suite | Pass Rate | Latency | TTFT | Tokens/sec | RAM Delta |\n|---|---|---|---|---|---|---|\n`;
  series.forEach((s) => {
    const run = runs.find((r) => r.run_id === s.runId);
    categories.forEach((cat) => {
      const suite = run?.models[s.modelName]?.suites[cat];
      if (suite) {
        const res = suite.resource_usage || {};
        md += `| ${s.modelName} | ${cat} | ${fmtPct(suite.pass_rate)} (${suite.pass_count}/${suite.total}) | ${fmtNum(suite.avg_latency_seconds)}s | ${fmtNum(suite.avg_ttft_seconds)}s | ${fmtNum(suite.avg_tokens_per_sec)} | ${fmtNum(res.ram_delta_gb, 2)} GB |\n`;
      }
    });
  });

  navigator.clipboard.writeText(md).then(() => {
    alert("Comparison report markdown copied to clipboard!");
  }).catch((err) => {
    alert("Failed to copy to clipboard: " + err.message);
  });
}

function exportCompareCSV(series, categories, runs) {
  let csv = "Model,Run,Suite,PassRate,PassCount,Total,AvgLatencySeconds,AvgTTFTSeconds,TokensPerSec,RAMDeltaGB,PeakCPUPercent\n";
  series.forEach((s) => {
    const run = runs.find((r) => r.run_id === s.runId);
    categories.forEach((cat) => {
      const suite = run?.models[s.modelName]?.suites[cat];
      if (suite) {
        const res = suite.resource_usage || {};
        csv += `"${s.modelName}","${s.runId}","${cat}",${suite.pass_rate},${suite.pass_count},${suite.total},${suite.avg_latency_seconds || 0},${suite.avg_ttft_seconds || 0},${suite.avg_tokens_per_sec || 0},${res.ram_delta_gb || 0},${res.peak_cpu_percent || 0}\n`;
      }
    });
  });

  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.setAttribute("href", url);
  link.setAttribute("download", `localbench_compare_${new Date().toISOString().slice(0, 10)}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// ---------- Radar / Spider Chart ----------
function buildRadarChart(series, categories, getSuite) {
  const wrap = document.createElement("div");
  wrap.className = "card";

  const heading = document.createElement("h2");
  heading.textContent = "Pass Rate Profile (Radar Chart)";
  wrap.appendChild(heading);

  if (series.length > 1) {
    const legend = document.createElement("div");
    legend.className = "chart-legend";
    series.forEach((s, i) => {
      const item = document.createElement("div");
      item.className = "item";

      const swatch = document.createElement("span");
      swatch.className = "swatch";
      swatch.style.background = seriesColor(i);
      item.appendChild(swatch);

      const label = document.createElement("span");
      label.textContent = s.label;
      item.appendChild(label);

      legend.appendChild(item);
    });
    wrap.appendChild(legend);
  }

  const chartWrap = document.createElement("div");
  chartWrap.className = "chart-wrap";
  const width = 900, height = 340;
  const cx = width / 2, cy = height / 2;
  const rMax = 110;
  const n = categories.length;

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("class", "chart-svg");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Pass Rate Profile Radar Chart");

  const angles = categories.map((_, i) => -Math.PI / 2 + (2 * Math.PI * i) / n);

  // Concentric grid polygons (25%, 50%, 75%, 100%)
  const ticks = [0.25, 0.50, 0.75, 1.0];
  ticks.forEach((tickFrac) => {
    const rLevel = tickFrac * rMax;
    const gridPoints = angles.map((ang) => {
      const x = cx + rLevel * Math.cos(ang);
      const y = cy + rLevel * Math.sin(ang);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");

    const poly = document.createElementNS(svg.namespaceURI, "polygon");
    poly.setAttribute("points", gridPoints);
    poly.setAttribute("class", "gridline");
    poly.setAttribute("fill", "none");
    svg.appendChild(poly);

    const tickLabel = document.createElementNS(svg.namespaceURI, "text");
    tickLabel.setAttribute("x", cx + 5);
    tickLabel.setAttribute("y", cy - rLevel + 3);
    tickLabel.setAttribute("font-size", "9.5");
    tickLabel.setAttribute("fill", "var(--text-muted)");
    tickLabel.textContent = `${Math.round(tickFrac * 100)}%`;
    svg.appendChild(tickLabel);
  });

  // Axis spokes & suite labels
  categories.forEach((cat, i) => {
    const ang = angles[i];
    const ax = cx + rMax * Math.cos(ang);
    const ay = cy + rMax * Math.sin(ang);

    const axisLine = document.createElementNS(svg.namespaceURI, "line");
    axisLine.setAttribute("x1", cx);
    axisLine.setAttribute("y1", cy);
    axisLine.setAttribute("x2", ax);
    axisLine.setAttribute("y2", ay);
    axisLine.setAttribute("class", "gridline");
    svg.appendChild(axisLine);

    const lx = cx + (rMax + 22) * Math.cos(ang);
    const ly = cy + (rMax + 22) * Math.sin(ang);

    let anchor = "middle";
    const cosVal = Math.cos(ang);
    if (Math.abs(cosVal) > 0.15) {
      anchor = cosVal > 0 ? "start" : "end";
    }

    const catLabel = document.createElementNS(svg.namespaceURI, "text");
    catLabel.setAttribute("x", lx);
    catLabel.setAttribute("y", ly + 4);
    catLabel.setAttribute("text-anchor", anchor);
    catLabel.setAttribute("class", "cat-label");
    catLabel.textContent = cat;
    svg.appendChild(catLabel);
  });

  // Overlaid polygons per series
  series.forEach((s, si) => {
    const color = seriesColor(si);
    const pts = [];
    const vertexData = [];

    categories.forEach((cat, i) => {
      const ang = angles[i];
      const suite = getSuite(s, cat);
      const passRate = suite ? suite.pass_rate : null;
      const rVal = passRate != null ? passRate * rMax : 0;
      const px = cx + rVal * Math.cos(ang);
      const py = cy + rVal * Math.sin(ang);
      pts.push(`${px.toFixed(1)},${py.toFixed(1)}`);
      vertexData.push({
        cat,
        passRate,
        valueText: passRate != null ? `${Math.round(passRate * 100)}%` : "n/a",
        px,
        py,
      });
    });

    const polygon = document.createElementNS(svg.namespaceURI, "polygon");
    polygon.setAttribute("points", pts.join(" "));
    polygon.setAttribute("fill", color);
    polygon.setAttribute("fill-opacity", "0.18");
    polygon.setAttribute("stroke", color);
    polygon.setAttribute("stroke-width", "2");
    svg.appendChild(polygon);

    // Interactive vertices with tooltips. A suite that was never run for
    // this model plots at the same r=0 origin as a genuine 0% score would --
    // without a distinct marker style, "not tested" and "failed everything"
    // are visually identical, which misrepresents untested suites as
    // failures. Hollow/muted marker + explicit "not run" tooltip for null,
    // solid series-colored marker for any real (even zero) score.
    vertexData.forEach((v) => {
      const isMissing = v.passRate == null;
      const circle = document.createElementNS(svg.namespaceURI, "circle");
      circle.setAttribute("cx", v.px);
      circle.setAttribute("cy", v.py);
      circle.setAttribute("r", "4.5");
      circle.setAttribute("class", "scatter-point");
      circle.setAttribute("tabindex", "0");
      if (isMissing) {
        circle.setAttribute("fill", "var(--surface)");
        circle.setAttribute("stroke", "var(--text-muted)");
        circle.setAttribute("stroke-width", "1.5");
        circle.setAttribute("stroke-dasharray", "2,1.5");
      } else {
        circle.setAttribute("fill", color);
      }

      const tipLabel = isMissing ? `${v.cat} (${s.label})` : `${v.cat} (${s.label})`;
      const tipValue = isMissing ? "not run" : v.valueText;
      const showTip = (evt) => showChartTooltip(chartWrap, evt, tipLabel, tipValue);
      circle.addEventListener("mousemove", showTip);
      circle.addEventListener("mouseleave", () => hideChartTooltip(chartWrap));
      circle.addEventListener("focus", (evt) => showChartTooltip(chartWrap, evt, tipLabel, tipValue));
      circle.addEventListener("blur", () => hideChartTooltip(chartWrap));

      svg.appendChild(circle);
    });
  });

  chartWrap.appendChild(svg);
  const tooltip = document.createElement("div");
  tooltip.className = "chart-tooltip";
  chartWrap.appendChild(tooltip);
  wrap.appendChild(chartWrap);
  return wrap;
}

async function renderCompare() {
  const output = document.getElementById("compare-output");
  const runIds = Array.from(state.selectedCompareRuns);
  if (runIds.length === 0) {
    output.innerHTML = '<p class="empty-state">Select one or more runs above to compare them.</p>';
    return;
  }

  const runs = [];
  for (const runId of runIds) {
    if (!compareRunCache.has(runId)) {
      compareRunCache.set(runId, await api(`/api/runs/${runId}`));
    }
    runs.push(compareRunCache.get(runId));
  }

  const series = [];
  runs.forEach((run) => {
    Object.keys(run.models).forEach((modelName) => {
      series.push({ key: `${run.run_id}::${modelName}`, label: `${modelName} (${run.run_id})`, runId: run.run_id, modelName });
    });
  });

  const suiteSet = new Set();
  runs.forEach((run) => Object.values(run.models).forEach((m) => Object.keys(m.suites).forEach((s) => {
    if (s !== "frontier_graded") suiteSet.add(s);
  })));
  const categories = DETERMINISTIC_SUITE_ORDER.filter((s) => suiteSet.has(s));

  function getSuite(s, category) {
    const run = runs.find((r) => r.run_id === s.runId);
    return run?.models[s.modelName]?.suites[category] || null;
  }

  output.innerHTML = "";

  const exportBar = document.createElement("div");
  exportBar.className = "export-bar";

  const exportMdBtn = document.createElement("button");
  exportMdBtn.className = "secondary";
  exportMdBtn.textContent = "📋 Copy Markdown Report";
  exportMdBtn.onclick = () => exportCompareMarkdown(series, categories, runs);
  exportBar.appendChild(exportMdBtn);

  const exportCsvBtn = document.createElement("button");
  exportCsvBtn.className = "secondary";
  exportCsvBtn.textContent = "📥 Download CSV";
  exportCsvBtn.onclick = () => exportCompareCSV(series, categories, runs);
  exportBar.appendChild(exportCsvBtn);

  const savings = calculateApiSavings(runs);
  const savingsBadge = document.createElement("span");
  savingsBadge.className = "muted";
  savingsBadge.style.marginLeft = "auto";
  savingsBadge.title =
    `Illustrative only, not a live price lookup: ${savings.totalPromptTokens.toLocaleString()} real prompt + ` +
    `${savings.totalCompletionTokens.toLocaleString()} real completion tokens at a placeholder ` +
    `$${ILLUSTRATIVE_RATE_PER_M_INPUT}/$${ILLUSTRATIVE_RATE_PER_M_OUTPUT} per 1M tokens.`;
  savingsBadge.appendChild(document.createTextNode("Illustrative cloud cost: "));
  const savingsStrong = document.createElement("strong");
  savingsStrong.style.color = "var(--good)";
  savingsStrong.textContent = `~$${savings.cost}`;
  savingsBadge.appendChild(savingsStrong);
  savingsBadge.appendChild(document.createTextNode(` (${savings.totalProblems} tasks) ⓘ`));
  exportBar.appendChild(savingsBadge);

  output.appendChild(exportBar);

  output.appendChild(buildModelDetailsTable(series, runs));
  output.appendChild(buildParetoChart(series, categories, getSuite));
  output.appendChild(buildRadarChart(series, categories, getSuite));

  output.appendChild(
    buildBarChart({
      title: "Pass rate by suite",
      categories,
      series,
      getValue: (s, cat) => {
        const suite = getSuite(s, cat);
        return suite ? Math.round(suite.pass_rate * 100) : null;
      },
      formatValue: (v) => `${v}%`,
      yMax: 100,
      yTicks: [0, 25, 50, 75, 100],
    })
  );

  const maxTokPerSec = Math.max(
    1,
    ...series.flatMap((s) => categories.map((cat) => getSuite(s, cat)?.avg_tokens_per_sec || 0))
  );
  const tokYMax = Math.ceil(maxTokPerSec / 10) * 10 || 10;
  output.appendChild(
    buildBarChart({
      title: "Tokens/sec by suite",
      categories,
      series,
      getValue: (s, cat) => getSuite(s, cat)?.avg_tokens_per_sec ?? null,
      formatValue: (v) => fmtNum(v, 0),
      yMax: tokYMax,
      yTicks: [0, tokYMax / 2, tokYMax],
    })
  );

  output.appendChild(buildTTFTChart(series, categories, getSuite));
  output.appendChild(buildPerSuiteBreakdown(series, categories, getSuite));
  output.appendChild(buildFullComparisonTable(series, categories, getSuite, runs));

  const frontierSection = buildFrontierJudgeSection(series, runs, getSuite);
  if (frontierSection) output.appendChild(frontierSection);
}

// Per-suite, per-problem outcome grid. The aggregate charts above answer
// "which model is better"; this answers "on WHICH task, and did every model
// fail the same one?" -- a problem all models fail is usually a problem with
// the problem, and that's invisible in a pass-rate average.
function buildPerSuiteBreakdown(series, categories, getSuite) {
  const wrap = document.createElement("div");
  wrap.className = "card";

  const heading = document.createElement("h2");
  heading.textContent = "Per-problem results by suite";
  wrap.appendChild(heading);

  const note = document.createElement("p");
  note.className = "muted";
  note.textContent =
    "Every individual task, per model. Click any cell to inspect that problem's prompt, response and error. " +
    "A column where every model fails usually says more about the task than the models.";
  wrap.appendChild(note);

  const legend = document.createElement("div");
  legend.className = "chart-legend";
  [
    ["pass", "Passed"],
    ["fail", "Failed"],
    ["trunc", "Failed (hit token limit)"],
    ["absent", "Not run"],
  ].forEach(([cls, label]) => {
    const item = document.createElement("span");
    item.className = "legend-item";
    const swatch = document.createElement("span");
    swatch.className = `cellmark cellmark-${cls}`;
    item.appendChild(swatch);
    item.appendChild(document.createTextNode(label));
    legend.appendChild(item);
  });
  wrap.appendChild(legend);

  categories.forEach((cat) => {
    // Union of problem ids across models, preserving first-seen order so the
    // columns line up even when a model errored out partway through a suite.
    const problemIds = [];
    const seen = new Set();
    series.forEach((s) => {
      (getSuite(s, cat)?.problems || []).forEach((p) => {
        if (!seen.has(p.problem_id)) {
          seen.add(p.problem_id);
          problemIds.push(p.problem_id);
        }
      });
    });
    if (problemIds.length === 0) return;

    const subHeading = document.createElement("h3");
    subHeading.textContent = `${SUITE_METADATA[cat]?.title || cat} (${problemIds.length} tasks)`;
    wrap.appendChild(subHeading);

    const scroller = document.createElement("div");
    scroller.className = "grid-scroll";
    const table = document.createElement("table");
    table.className = "problem-grid";

    const headRow = document.createElement("tr");
    const corner = document.createElement("th");
    corner.textContent = "Model";
    headRow.appendChild(corner);
    problemIds.forEach((pid) => {
      const th = document.createElement("th");
      th.className = "grid-col-head";
      th.title = pid;
      th.textContent = pid;
      headRow.appendChild(th);
    });
    const rateHead = document.createElement("th");
    rateHead.textContent = "Pass rate";
    headRow.appendChild(rateHead);
    table.appendChild(headRow);

    series.forEach((s) => {
      const suite = getSuite(s, cat);
      const row = document.createElement("tr");
      const nameCell = document.createElement("td");
      nameCell.className = "grid-row-head";
      nameCell.textContent = s.modelName;
      nameCell.title = `${s.modelName} (${s.runId})`;
      row.appendChild(nameCell);

      const byId = new Map((suite?.problems || []).map((p) => [p.problem_id, p]));
      problemIds.forEach((pid) => {
        const p = byId.get(pid);
        const td = document.createElement("td");
        td.className = "grid-cell";
        const mark = document.createElement("span");
        let cls = "absent";
        let tip = `${pid}: not run for this model`;
        if (p) {
          if (p.passed) {
            cls = "pass";
            tip = `${pid}: passed`;
          } else if (p.truncated) {
            cls = "trunc";
            tip = `${pid}: hit the token limit before finishing -- ${p.error || ""}`;
          } else {
            cls = "fail";
            tip = `${pid}: ${p.error || "failed"}`;
          }
        }
        mark.className = `cellmark cellmark-${cls}`;
        td.title = tip;
        if (p && suite) {
          td.classList.add("clickable");
          td.addEventListener("click", () => openInspectorModalByData(s.runId, s.modelName, cat));
        }
        td.appendChild(mark);
        row.appendChild(td);
      });

      const rateCell = document.createElement("td");
      rateCell.className = "grid-rate";
      rateCell.textContent = suite ? `${fmtPct(suite.pass_rate)} (${suite.pass_count}/${suite.total})` : "n/a";
      row.appendChild(rateCell);

      table.appendChild(row);
    });

    // "How many models solved this task" -- makes a universally-failed task
    // (or a trivially-passed one) obvious at a glance.
    const footRow = document.createElement("tr");
    const footLabel = document.createElement("td");
    footLabel.className = "grid-row-head";
    footLabel.textContent = "Solved by";
    footLabel.title = "How many of the compared models passed this task.";
    footRow.appendChild(footLabel);
    problemIds.forEach((pid) => {
      let passed = 0;
      let attempted = 0;
      series.forEach((s) => {
        const p = (getSuite(s, cat)?.problems || []).find((x) => x.problem_id === pid);
        if (p) {
          attempted++;
          if (p.passed) passed++;
        }
      });
      const td = document.createElement("td");
      td.className = "grid-tally";
      td.textContent = attempted ? `${passed}/${attempted}` : "-";
      if (attempted && passed === 0) td.classList.add("tally-none");
      if (attempted && passed === attempted) td.classList.add("tally-all");
      footRow.appendChild(td);
    });
    footRow.appendChild(document.createElement("td"));
    table.appendChild(footRow);

    scroller.appendChild(table);
    wrap.appendChild(scroller);
  });

  return wrap;
}

function buildFrontierJudgeSection(series, runs, getSuite) {
  const rows = series
    .map((s) => ({ s, suite: getSuite(s, "frontier_graded") }))
    .filter((r) => r.suite);
  if (rows.length === 0) return null;

  const wrap = document.createElement("div");
  wrap.className = "card";

  const heading = document.createElement("h2");
  heading.textContent = "Frontier Judge (paid, non-deterministic)";
  wrap.appendChild(heading);

  const note = document.createElement("p");
  note.className = "muted";
  note.textContent =
    "Each model answered its own freshly judge-generated tasks (different literal questions per model, " +
    "not identical fixtures like the suites above) across 6 qualitative categories, graded by the same " +
    "frontier model/rubric per run (see TASK_SPEC.md). The score is a qualitative judge signal, not an " +
    "exact-match result -- kept separate from the deterministic pass rates.";
  wrap.appendChild(note);

  const table = document.createElement("table");
  const headRow = document.createElement("tr");
  ["Model", "Run", "Judged By", "Avg Score (/10)", "Pass Rate", ""].forEach((h) => {
    const th = document.createElement("th");
    th.textContent = h;
    headRow.appendChild(th);
  });
  table.appendChild(headRow);

  rows.forEach(({ s, suite }) => {
    const run = runs.find((r) => r.run_id === s.runId);
    const judgeInfo = run?.config_summary?.frontier_judge;
    const judgedBy = judgeInfo ? `${judgeInfo.provider}/${judgeInfo.model}` : "unknown";

    const row = document.createElement("tr");

    const tdModel = document.createElement("td");
    tdModel.textContent = s.modelName;
    row.appendChild(tdModel);

    const tdRun = document.createElement("td");
    tdRun.textContent = s.runId;
    row.appendChild(tdRun);

    const tdJudge = document.createElement("td");
    tdJudge.textContent = judgedBy;
    row.appendChild(tdJudge);

    const tdScore = document.createElement("td");
    tdScore.textContent = suite.avg_score != null ? fmtNum(suite.avg_score, 1) : "n/a";
    row.appendChild(tdScore);

    const tdPass = document.createElement("td");
    tdPass.textContent = `${fmtPct(suite.pass_rate)} (${suite.pass_count}/${suite.total})`;
    row.appendChild(tdPass);

    const tdExpand = document.createElement("td");
    const expandBtn = document.createElement("button");
    expandBtn.className = "secondary small";
    expandBtn.textContent = "View rationale";
    expandBtn.title = "See what the frontier judge said about each response";
    expandBtn.addEventListener("click", () => openInspectorModalByData(s.runId, s.modelName, "frontier_graded"));
    tdExpand.appendChild(expandBtn);
    row.appendChild(tdExpand);

    table.appendChild(row);
  });

  wrap.appendChild(table);
  return wrap;
}

// ---------- settings ----------
const PROVIDER_LABELS = {
  anthropic: "Anthropic",
  openai: "OpenAI",
  gemini: "Gemini",
  openrouter: "OpenRouter",
};

function apiErrorDetail(err) {
  const match = /^\d+: (.*)$/s.exec(err.message || "");
  if (!match) return err.message || String(err);
  try {
    return JSON.parse(match[1]).detail || match[1];
  } catch {
    return match[1];
  }
}

async function loadSettings() {
  const data = await api("/api/settings");
  document.getElementById("settings-base-url").value = data.runtime.base_url || "";
  document.getElementById("settings-timeout").value = data.runtime.request_timeout_seconds ?? "";
  document.getElementById("settings-unload-cmd").value = data.runtime.unload_all_cmd || "";

  document.getElementById("settings-judge-enabled").checked = !!data.judge.enabled;
  const judgeProvider = data.judge.provider || "anthropic";
  document.getElementById("settings-judge-provider").value = judgeProvider;
  document.getElementById("settings-judge-model").value = data.judge.model || "";
  document.getElementById("settings-judge-num-tasks").value = data.judge.num_tasks ?? "";
  document.getElementById("settings-judge-pass-threshold").value = data.judge.pass_threshold ?? "";
  state.settingsJudgeSaved = { provider: judgeProvider, model: data.judge.model || "" };
  await loadJudgeModelOptions(judgeProvider, "settings-judge-model-options", "settings-judge-model-status");

  renderKeysList(data.keys);
}

document.getElementById("settings-judge-provider")?.addEventListener("change", async (e) => {
  const provider = e.target.value;
  const modelInput = document.getElementById("settings-judge-model");
  modelInput.value = state.settingsJudgeSaved?.provider === provider ? (state.settingsJudgeSaved.model || "") : "";
  await loadJudgeModelOptions(provider, "settings-judge-model-options", "settings-judge-model-status");
});

function renderKeysList(keys) {
  const container = document.getElementById("settings-keys-list");
  container.innerHTML = "";
  Object.entries(PROVIDER_LABELS).forEach(([provider, label]) => {
    const isSet = !!keys[provider];

    const row = document.createElement("div");
    row.className = "key-row";

    const labelEl = document.createElement("span");
    labelEl.className = "key-row-label";
    labelEl.textContent = label;
    row.appendChild(labelEl);

    const statusEl = document.createElement("span");
    statusEl.className = "key-status " + (isSet ? "key-set" : "key-unset");
    statusEl.textContent = isSet ? "● Set" : "○ Not set";
    row.appendChild(statusEl);

    const input = document.createElement("input");
    input.type = "password";
    input.placeholder = isSet ? "•••••••••••• (leave blank to keep)" : "paste API key";
    input.className = "key-row-input";
    row.appendChild(input);

    const saveBtn = document.createElement("button");
    saveBtn.className = "secondary small";
    saveBtn.textContent = "Save";
    saveBtn.addEventListener("click", async () => {
      if (!input.value.trim()) return;
      try {
        const res = await api("/api/settings/keys", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ provider, api_key: input.value }),
        });
        renderKeysList(res.keys);
      } catch (e) {
        alert(`Failed to save ${label} key: ${apiErrorDetail(e)}`);
      }
    });
    row.appendChild(saveBtn);

    const clearBtn = document.createElement("button");
    clearBtn.className = "secondary small";
    clearBtn.textContent = "Clear";
    clearBtn.disabled = !isSet;
    clearBtn.addEventListener("click", async () => {
      try {
        const res = await api(`/api/settings/keys/${provider}`, { method: "DELETE" });
        renderKeysList(res.keys);
      } catch (e) {
        alert(`Failed to clear ${label} key: ${apiErrorDetail(e)}`);
      }
    });
    row.appendChild(clearBtn);

    container.appendChild(row);
  });
}

document.querySelectorAll("[data-preset-url]").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.getElementById("settings-base-url").value = btn.dataset.presetUrl;
  });
});

document.getElementById("settings-runtime-save")?.addEventListener("click", async () => {
  const statusEl = document.getElementById("settings-runtime-status");
  statusEl.textContent = "Saving...";
  try {
    await api("/api/settings/runtime", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        base_url: document.getElementById("settings-base-url").value,
        request_timeout_seconds: Number(document.getElementById("settings-timeout").value),
        unload_all_cmd: document.getElementById("settings-unload-cmd").value,
      }),
    });
    statusEl.textContent = "Saved.";
  } catch (e) {
    statusEl.textContent = `Error: ${apiErrorDetail(e)}`;
  }
});

document.getElementById("settings-runtime-test")?.addEventListener("click", async () => {
  const statusEl = document.getElementById("settings-runtime-status");
  statusEl.textContent = "Testing...";
  try {
    const res = await api("/api/models/detect");
    statusEl.textContent = `Reachable -- ${res.models.length} model(s) visible.`;
  } catch (e) {
    statusEl.textContent = `Unreachable: ${apiErrorDetail(e)}`;
  }
});

document.getElementById("settings-judge-save")?.addEventListener("click", async () => {
  const statusEl = document.getElementById("settings-judge-status");
  statusEl.textContent = "Saving...";
  try {
    await api("/api/settings/judge", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        enabled: document.getElementById("settings-judge-enabled").checked,
        provider: document.getElementById("settings-judge-provider").value,
        model: document.getElementById("settings-judge-model").value,
        num_tasks: Number(document.getElementById("settings-judge-num-tasks").value),
        pass_threshold: Number(document.getElementById("settings-judge-pass-threshold").value),
      }),
    });
    statusEl.textContent = "Saved.";
  } catch (e) {
    statusEl.textContent = `Error: ${apiErrorDetail(e)}`;
  }
});

// ---------- init ----------
loadConfig();
resumeActiveRunIfAny();
