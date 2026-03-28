const LAYOUT_LABELS = {
  1: "1 pane",
  2: "2 panes",
  4: "4 panes",
};

const STORAGE_KEY = "parallel-log-dashboard-layout";

function makePane(index) {
  return {
    id: `pane-${index}`,
    runId: null,
    source: "run.log",
    text: "",
    startOffset: 0,
    endOffset: 0,
    fileSize: 0,
    hasMore: false,
    loading: false,
    requestKey: "",
    queue: [],
    queueTimer: null,
    autoFollow: true,
  };
}

const state = {
  logDir: "",
  runs: new Map(),
  layoutMode: 1,
  panes: [makePane(1)],
  activePaneId: "pane-1",
  draggingRunId: null,
  dropPaneId: null,
  logSearchQuery: "",
  logSearchMatchCount: 0,
  activeLogMatchIndex: -1,
  streamCursor: 0,
  stream: null,
};

function saveLayoutState() {
  try {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        layoutMode: state.layoutMode,
        activePaneId: state.activePaneId,
        panes: state.panes.map((pane) => ({
          id: pane.id,
          runId: pane.runId,
          source: pane.source,
        })),
      })
    );
  } catch (_error) {
    // Ignore storage failures.
  }
}

function loadLayoutState() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return;
    }
    const saved = JSON.parse(raw);
    if (!LAYOUT_LABELS[saved.layoutMode]) {
      return;
    }
    state.layoutMode = saved.layoutMode;
    state.panes = Array.from({ length: saved.layoutMode }, (_unused, index) => {
      const paneId = `pane-${index + 1}`;
      const savedPane = (saved.panes || []).find((pane) => pane.id === paneId);
      const pane = makePane(index + 1);
      if (savedPane) {
        pane.runId = savedPane.runId || null;
        pane.source = savedPane.source || "run.log";
      }
      return pane;
    });
    state.activePaneId = state.panes.some((pane) => pane.id === saved.activePaneId)
      ? saved.activePaneId
      : state.panes[0]?.id || "pane-1";
  } catch (_error) {
    // Ignore malformed storage content.
  }
}

function allRuns() {
  return [...state.runs.values()];
}

function getPaneById(paneId) {
  return state.panes.find((pane) => pane.id === paneId) || null;
}

function getActivePane() {
  return getPaneById(state.activePaneId);
}

function getRunForPane(pane) {
  return pane?.runId ? state.runs.get(pane.runId) || null : null;
}

function getPaneStreamElement(paneId) {
  return document.querySelector(`[data-pane-stream="${paneId}"]`);
}

function isNearBottom(element, threshold = 20) {
  if (!element) {
    return true;
  }
  return element.scrollHeight - element.clientHeight - element.scrollTop < threshold;
}

function formatTime(timestampSeconds) {
  if (!timestampSeconds) {
    return "-";
  }
  return new Date(timestampSeconds * 1000).toLocaleTimeString();
}

function tokenizeSearchQuery(query) {
  return query
    .toLowerCase()
    .split(/\s+/)
    .map((token) => token.trim())
    .filter(Boolean);
}

function buildSearchBlob(run) {
  return [
    run.run_id,
    run.status,
    run.status_label,
    ...(run.files || []),
    run.log_excerpt || "",
  ]
    .join(" ")
    .toLowerCase();
}

function buildScopedSearchBlob(run, mode) {
  if (mode === "run_id") {
    return (run.run_id || "").toLowerCase();
  }
  if (mode === "status") {
    return `${run.status || ""} ${run.status_label || ""}`.toLowerCase();
  }
  if (mode === "files") {
    return (run.files || []).join(" ").toLowerCase();
  }
  if (mode === "logs") {
    return (run.log_excerpt || "").toLowerCase();
  }
  return buildSearchBlob(run);
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function detectBlockKind(line) {
  if (line.includes("STEP ")) {
    return "step";
  }
  if (line.includes("THOUGHT:") || line.includes("💭 THOUGHT")) {
    return "thought";
  }
  if (line.includes("ACTION:") || line.includes("🎬 ACTION")) {
    return "action";
  }
  if (line.includes("MODEL INPUT")) {
    return "input";
  }
  if (line.includes("Observation:")) {
    return "observation";
  }
  return "";
}

function classifyLogLine(line, inheritedKind = "") {
  const classes = ["log-line"];
  const blockKind = detectBlockKind(line);
  const isSystemLine = /^\d{4}-\d{2}-\d{2}/.test(line) && !blockKind;
  if (line.includes("| DEBUG")) {
    classes.push("level-debug");
  }
  if (line.includes("| INFO")) {
    classes.push("level-info");
  }
  if (line.includes("| WARNING")) {
    classes.push("level-warning");
  }
  if (line.includes("| ERROR")) {
    classes.push("level-error");
  }
  if (line.includes("| CRITICAL")) {
    classes.push("level-critical");
  }
  if (line.includes("| reward_spec")) {
    classes.push("name-reward_spec");
  }
  if (line.includes("| interaction")) {
    classes.push("name-interaction");
  }
  if (line.includes("| agent-loop")) {
    classes.push("name-agent-loop");
  }
  if (line.includes("| environment") || line.includes("| deployment") || line.includes("| runtime")) {
    classes.push("name-environment");
  }
  if (blockKind) {
    classes.push(`kind-${blockKind}`);
    classes.push("kind-header");
  } else if (inheritedKind) {
    classes.push(`kind-${inheritedKind}`);
    classes.push("kind-body");
  } else if (isSystemLine) {
    classes.push("kind-system");
  }
  return { classes: classes.join(" "), blockKind };
}

function countMatches(haystack, needle) {
  if (!needle) {
    return 0;
  }
  const source = haystack.toLowerCase();
  let index = 0;
  let count = 0;
  while (index <= source.length - needle.length) {
    const foundAt = source.indexOf(needle, index);
    if (foundAt === -1) {
      break;
    }
    count += 1;
    index = foundAt + needle.length;
  }
  return count;
}

function renderHighlightedText(text, query, nextMatchIndex, activeMatchIndex) {
  if (!query) {
    return text ? escapeHtml(text) : "&nbsp;";
  }
  const lowered = text.toLowerCase();
  let cursor = 0;
  let html = "";
  while (cursor < text.length) {
    const matchAt = lowered.indexOf(query, cursor);
    if (matchAt === -1) {
      html += escapeHtml(text.slice(cursor));
      break;
    }
    if (matchAt > cursor) {
      html += escapeHtml(text.slice(cursor, matchAt));
    }
    const matchIndex = nextMatchIndex();
    const activeClass = matchIndex === activeMatchIndex ? " active" : "";
    html += `<mark class="log-match${activeClass}">${escapeHtml(text.slice(matchAt, matchAt + query.length))}</mark>`;
    cursor = matchAt + query.length;
  }
  return html || "&nbsp;";
}

function renderLogHtml(text, query = "", activeMatchIndex = -1) {
  const lines = text.split("\n");
  let activeBlockKind = "";
  let nextMatchIndex = 0;
  return lines
    .map((line) => {
      const isHeaderLine = /^\d{4}-\d{2}-\d{2}/.test(line);
      const { classes, blockKind } = classifyLogLine(line, !isHeaderLine ? activeBlockKind : "");
      if (blockKind) {
        activeBlockKind = blockKind;
      } else if (isHeaderLine) {
        activeBlockKind = "";
      }
      return `<div class="${classes}">${renderHighlightedText(line, query, () => nextMatchIndex++, activeMatchIndex)}</div>`;
    })
    .join("");
}

function getActiveSearchText() {
  return getActivePane()?.text || "";
}

function syncActivePaneSearchState() {
  const query = state.logSearchQuery.trim().toLowerCase();
  const totalMatches = countMatches(getActiveSearchText(), query);
  state.logSearchMatchCount = totalMatches;
  if (!query || totalMatches === 0) {
    state.activeLogMatchIndex = -1;
  } else if (state.activeLogMatchIndex < 0 || state.activeLogMatchIndex >= totalMatches) {
    state.activeLogMatchIndex = 0;
  }
}

function updateLogSearchUi() {
  if (!state.logSearchQuery.trim()) {
    logSearchCount.textContent = "Search active pane";
  } else if (state.logSearchMatchCount === 0) {
    logSearchCount.textContent = "No matches";
  } else {
    logSearchCount.textContent = `${state.activeLogMatchIndex + 1} / ${state.logSearchMatchCount}`;
  }
  logSearchPrev.disabled = state.logSearchMatchCount === 0;
  logSearchNext.disabled = state.logSearchMatchCount === 0;
}

function scrollActiveLogMatchIntoView() {
  const activeMatch = document.querySelector(`.log-pane[data-pane-id="${state.activePaneId}"] .log-match.active`);
  if (activeMatch) {
    activeMatch.scrollIntoView({ block: "center" });
  }
}

function renderPaneStream(paneId, { stickToBottom = false, restoreScrollTop = null } = {}) {
  const pane = getPaneById(paneId);
  const streamEl = getPaneStreamElement(paneId);
  if (!pane || !streamEl) {
    return;
  }
  const query = paneId === state.activePaneId ? state.logSearchQuery.trim().toLowerCase() : "";
  streamEl.dataset.plainText = pane.text;
  streamEl.innerHTML = renderLogHtml(pane.text, query, paneId === state.activePaneId ? state.activeLogMatchIndex : -1);
  if (restoreScrollTop !== null) {
    streamEl.scrollTop = restoreScrollTop;
  } else if (stickToBottom) {
    streamEl.scrollTop = streamEl.scrollHeight;
  }
}

function resetPaneContent(pane) {
  pane.text = "";
  pane.startOffset = 0;
  pane.endOffset = 0;
  pane.fileSize = 0;
  pane.hasMore = false;
  pane.loading = false;
  pane.requestKey = "";
  pane.queue = [];
  pane.autoFollow = true;
  if (pane.queueTimer) {
    window.clearInterval(pane.queueTimer);
    pane.queueTimer = null;
  }
}

function getAvailableSource(run, requestedSource = "run.log") {
  const availableSources = (run.log_sources || []).map((item) => item.key);
  if (availableSources.includes(requestedSource)) {
    return requestedSource;
  }
  return availableSources[0] || "run.log";
}

function getPaneSummaryLabel() {
  return LAYOUT_LABELS[state.layoutMode] || `${state.layoutMode} panes`;
}

function hasTextSelection() {
  const selection = window.getSelection();
  return Boolean(selection && selection.toString().trim());
}

function clearDragState() {
  state.draggingRunId = null;
  state.dropPaneId = null;
  document.querySelectorAll(".run-card.dragging").forEach((element) => element.classList.remove("dragging"));
  document.querySelectorAll(".log-pane.drag-target").forEach((element) => element.classList.remove("drag-target"));
}

function setDropPaneTarget(paneId) {
  if (state.dropPaneId === paneId) {
    return;
  }
  state.dropPaneId = paneId;
  document.querySelectorAll(".log-pane.drag-target").forEach((element) => {
    element.classList.toggle("drag-target", element.dataset.paneId === paneId);
  });
}

function statCards(stats) {
  const cards = [
    ["Total", stats.total],
    ["Active", stats.active],
    ["Queued", stats.queued],
    ["Running", stats.running],
    ["Verify", stats.verify],
    ["Completed", stats.completed],
  ];
  statsRoot.innerHTML = cards
    .map(
      ([label, value]) => `
        <article class="stat-card card">
          <span class="stat-label">${label}:</span>
          <strong class="stat-value">${value}</strong>
        </article>
      `
    )
    .join("");
}

async function fetchLogChunk(runId, source, before = null) {
  const params = new URLSearchParams({
    run_id: runId,
    source,
    chunk_size: "64000",
  });
  if (before !== null && before !== undefined) {
    params.set("before", String(before));
  }
  const response = await fetch(`/api/log?${params.toString()}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch log chunk: ${response.status}`);
  }
  return response.json();
}

async function loadPaneChunk(paneId, { prepend = false } = {}) {
  const pane = getPaneById(paneId);
  const run = getRunForPane(pane);
  if (!pane || !run || pane.loading) {
    return;
  }

  const before = prepend ? pane.startOffset || null : null;
  pane.loading = true;
  const requestKey = `${pane.id}:${run.run_id}:${pane.source}:${before ?? "latest"}`;
  pane.requestKey = requestKey;

  try {
    const chunk = await fetchLogChunk(run.run_id, pane.source, before);
    if (!getPaneById(paneId) || pane.requestKey !== requestKey) {
      return;
    }

    if (prepend) {
      const streamEl = getPaneStreamElement(paneId);
      const previousHeight = streamEl?.scrollHeight || 0;
      const previousTop = streamEl?.scrollTop || 0;
      const separator = chunk.text && pane.text && !chunk.text.endsWith("\n") ? "\n" : "";
      pane.text = `${chunk.text}${separator}${pane.text}`;
      pane.startOffset = chunk.start_offset;
      pane.endOffset = chunk.end_offset;
      pane.fileSize = chunk.file_size;
      pane.hasMore = chunk.has_more;
      if (paneId === state.activePaneId) {
        syncActivePaneSearchState();
        updateLogSearchUi();
      }
      renderPaneStream(paneId);
      const newStreamEl = getPaneStreamElement(paneId);
      if (newStreamEl) {
        const newHeight = newStreamEl.scrollHeight;
        newStreamEl.scrollTop = newHeight - previousHeight + previousTop;
      }
      return;
    }

    const streamEl = getPaneStreamElement(paneId);
    const restoreScrollTop = pane.autoFollow ? null : streamEl?.scrollTop ?? null;
    pane.text = chunk.text || "";
    pane.startOffset = chunk.start_offset;
    pane.endOffset = chunk.end_offset;
    pane.fileSize = chunk.file_size;
    pane.hasMore = chunk.has_more;
    if (paneId === state.activePaneId) {
      syncActivePaneSearchState();
      updateLogSearchUi();
    }
    renderPaneStream(paneId, {
      stickToBottom: pane.autoFollow,
      restoreScrollTop,
    });
  } catch (_error) {
    // Keep current pane content if fetch fails.
  } finally {
    pane.loading = false;
  }
}

function filteredRuns() {
  const keywords = tokenizeSearchQuery(searchInput.value);
  const status = statusFilter.value;
  const mode = searchMode.value;
  return allRuns()
    .filter((run) => (status === "all" ? true : run.status === status))
    .filter((run) => {
      if (keywords.length === 0) {
        return true;
      }
      const searchBlob = buildScopedSearchBlob(run, mode);
      return keywords.every((keyword) => searchBlob.includes(keyword));
    })
    .map((run, index) => ({ run, index }))
    .sort((left, right) => {
      const leftCompleted = left.run.status === "completed" ? 1 : 0;
      const rightCompleted = right.run.status === "completed" ? 1 : 0;
      if (leftCompleted !== rightCompleted) {
        return leftCompleted - rightCompleted;
      }
      return left.index - right.index;
    })
    .map(({ run }) => run);
}

function renderRuns() {
  const runs = filteredRuns();
  const activeRunId = getRunForPane(getActivePane())?.run_id || null;
  document.getElementById("runCount").textContent = `${runs.length} runs`;
  emptyState.classList.toggle("hidden", runs.length > 0);
  runsRoot.innerHTML = runs
    .map((run) => {
      const activeClass = run.run_id === activeRunId ? "active" : "";
      const stepLabel = run.current_step == null ? "Step -" : `Step ${run.current_step}`;
      const createdLabel = formatTime(run.created_at);
      return `
        <article class="run-card ${activeClass}" data-run-id="${run.run_id}" data-tone="${run.tone}" draggable="true">
          <div class="run-top">
            <div class="run-main">
              <strong title="${run.run_id}">${run.run_id}</strong>
            </div>
            <span class="status-chip" data-tone="${run.tone}">${run.status_label}</span>
          </div>
          <div class="run-subline">
            <span class="created-meta" title="Created ${createdLabel}">${createdLabel}</span>
            <span class="step-meta">${stepLabel}</span>
          </div>
        </article>
      `;
    })
    .join("");

  runsRoot.querySelectorAll(".run-card").forEach((card) => {
    card.addEventListener("click", () => assignRunToActivePane(card.dataset.runId));
    card.addEventListener("dragstart", (event) => {
      state.draggingRunId = card.dataset.runId;
      card.classList.add("dragging");
      if (event.dataTransfer) {
        event.dataTransfer.effectAllowed = "copy";
        event.dataTransfer.setData("text/plain", card.dataset.runId);
      }
    });
    card.addEventListener("dragend", () => {
      clearDragState();
    });
  });
}

function capturePaneScrolls() {
  const scrolls = new Map();
  state.panes.forEach((pane) => {
    const streamEl = getPaneStreamElement(pane.id);
    if (!streamEl) {
      return;
    }
    scrolls.set(pane.id, {
      scrollTop: streamEl.scrollTop,
      atBottom: streamEl.scrollHeight - streamEl.clientHeight - streamEl.scrollTop < 12,
    });
  });
  return scrolls;
}

function renderPaneGrid() {
  const paneSummary = document.getElementById("paneSummary");
  const previousScrolls = capturePaneScrolls();
  paneSummary.textContent = getPaneSummaryLabel();
  paneGrid.dataset.layout = String(state.layoutMode);
  paneGrid.innerHTML = state.panes
    .map((pane, index) => {
      const run = getRunForPane(pane);
      const activeClass = pane.id === state.activePaneId ? "active" : "";
      if (!run) {
        return `
          <article class="log-pane log-pane-empty ${activeClass}" data-pane-id="${pane.id}">
            <div class="log-pane-empty-copy">
              <strong>Pane ${index + 1}</strong>
              <span>Click this pane, choose a run, or drag one here.</span>
            </div>
          </article>
        `;
      }

      const selectedSource = getAvailableSource(run, pane.source);
      pane.source = selectedSource;
      const sourceOptions = (run.log_sources || [])
        .map(
          (source) => `
            <option value="${source.key}" ${source.key === selectedSource ? "selected" : ""}>${source.label}</option>
          `
        )
        .join("");

      return `
        <article class="log-pane ${activeClass}" data-pane-id="${pane.id}">
          <div class="log-pane-head">
            <div class="log-pane-title-row">
              <h3 title="${run.run_id}">${run.run_id}</h3>
              <div class="log-pane-controls">
                ${
                  (run.log_sources || []).length > 1
                    ? `
                      <label class="pane-source-wrap">
                        <select class="pane-source-select" data-pane-id="${pane.id}">
                          ${sourceOptions}
                        </select>
                      </label>
                    `
                    : ""
                }
                <span class="status-chip" data-tone="${run.tone}">${run.status_label}</span>
              </div>
            </div>
          </div>
          <div class="log-stream pane-log-stream" data-pane-stream="${pane.id}"></div>
        </article>
      `;
    })
    .join("");

  paneGrid.querySelectorAll(".log-pane").forEach((paneEl) => {
    paneEl.addEventListener("click", (event) => {
      if (event.target.closest(".pane-source-select")) {
        return;
      }
      if (!paneEl.classList.contains("log-pane-empty") && !event.target.closest(".log-pane-head")) {
        return;
      }
      if (hasTextSelection()) {
        return;
      }
      setActivePane(paneEl.dataset.paneId);
    });
    paneEl.addEventListener("dragover", (event) => {
      if (!state.draggingRunId) {
        return;
      }
      event.preventDefault();
      if (event.dataTransfer) {
        event.dataTransfer.dropEffect = "copy";
      }
      setDropPaneTarget(paneEl.dataset.paneId);
    });
    paneEl.addEventListener("dragenter", (event) => {
      if (!state.draggingRunId) {
        return;
      }
      event.preventDefault();
      setDropPaneTarget(paneEl.dataset.paneId);
    });
    paneEl.addEventListener("dragleave", (event) => {
      if (!state.draggingRunId) {
        return;
      }
      const related = event.relatedTarget;
      if (related instanceof Node && paneEl.contains(related)) {
        return;
      }
      if (state.dropPaneId === paneEl.dataset.paneId) {
        setDropPaneTarget(null);
      }
    });
    paneEl.addEventListener("drop", (event) => {
      if (!state.draggingRunId) {
        return;
      }
      event.preventDefault();
      const runId = event.dataTransfer?.getData("text/plain") || state.draggingRunId;
      clearDragState();
      setActivePane(paneEl.dataset.paneId);
      assignRunToPane(paneEl.dataset.paneId, runId);
    });
  });

  paneGrid.querySelectorAll(".pane-source-select").forEach((selectEl) => {
    selectEl.addEventListener("click", (event) => event.stopPropagation());
    selectEl.addEventListener("change", () => {
      const pane = getPaneById(selectEl.dataset.paneId);
      const run = getRunForPane(pane);
      if (!pane || !run) {
        return;
      }
      resetPaneContent(pane);
      pane.source = getAvailableSource(run, selectEl.value);
      pane.text = (run.log_contents || {})[pane.source] || "";
      saveLayoutState();
      syncActivePaneSearchState();
      updateLogSearchUi();
      renderPaneGrid();
      void loadPaneChunk(pane.id);
    });
  });

  paneGrid.querySelectorAll(".pane-log-stream").forEach((streamEl) => {
    streamEl.addEventListener("scroll", () => {
      const pane = getPaneById(streamEl.dataset.paneStream);
      if (pane) {
        pane.autoFollow = isNearBottom(streamEl);
      }
      if (streamEl.scrollTop < 40 && pane?.hasMore && !pane.loading) {
        void loadPaneChunk(pane.id, { prepend: true });
      }
    });
  });

  syncActivePaneSearchState();
  updateLogSearchUi();
  state.panes.forEach((pane) => {
    const previousScroll = previousScrolls.get(pane.id);
    renderPaneStream(pane.id, {
      stickToBottom: !previousScroll || previousScroll.atBottom,
      restoreScrollTop: previousScroll && !previousScroll.atBottom ? previousScroll.scrollTop : null,
    });
  });
}

function paneNeedsStructuralRefresh(paneId) {
  const pane = getPaneById(paneId);
  const run = getRunForPane(pane);
  const paneEl = document.querySelector(`.log-pane[data-pane-id="${paneId}"]`);
  if (!pane || !run || !paneEl) {
    return true;
  }
  if (paneEl.classList.contains("log-pane-empty")) {
    return true;
  }
  const selectEl = paneEl.querySelector(".pane-source-select");
  const shouldHaveSelect = (run.log_sources || []).length > 1;
  if (Boolean(selectEl) !== shouldHaveSelect) {
    return true;
  }
  if (selectEl && selectEl.options.length !== (run.log_sources || []).length) {
    return true;
  }
  return false;
}

function updatePaneChrome(paneId) {
  const pane = getPaneById(paneId);
  const run = getRunForPane(pane);
  const paneEl = document.querySelector(`.log-pane[data-pane-id="${paneId}"]`);
  if (!pane || !run || !paneEl || paneEl.classList.contains("log-pane-empty")) {
    return;
  }

  const titleEl = paneEl.querySelector(".log-pane-title-row h3");
  if (titleEl) {
    titleEl.textContent = run.run_id;
    titleEl.title = run.run_id;
  }

  const statusEl = paneEl.querySelector(".log-pane-controls .status-chip");
  if (statusEl) {
    statusEl.textContent = run.status_label;
    statusEl.dataset.tone = run.tone;
  }

  const selectEl = paneEl.querySelector(".pane-source-select");
  if (selectEl) {
    const selectedSource = getAvailableSource(run, pane.source);
    pane.source = selectedSource;
    [...selectEl.options].forEach((option) => {
      option.selected = option.value === selectedSource;
    });
  }
}

function updateHeader(snapshot) {
  document.getElementById("logDir").textContent = state.logDir || snapshot.log_dir || "-";
  statCards(snapshot.stats);
}

function mergeRun(run) {
  state.runs.set(run.run_id, run);
}

function reconcilePanesWithRuns() {
  state.panes.forEach((pane) => {
    if (!pane.runId) {
      return;
    }
    const run = state.runs.get(pane.runId);
    if (!run) {
      resetPaneContent(pane);
      pane.runId = null;
      pane.source = "run.log";
      return;
    }
    const nextSource = getAvailableSource(run, pane.source);
    if (nextSource !== pane.source) {
      resetPaneContent(pane);
      pane.source = nextSource;
      pane.text = (run.log_contents || {})[pane.source] ?? "";
      return;
    }
    if (pane.text === "" && pane.startOffset === 0 && pane.endOffset === 0) {
      pane.text = (run.log_contents || {})[pane.source] ?? "";
    }
  });
}

function setActivePane(paneId) {
  state.activePaneId = paneId;
  saveLayoutState();
  renderRuns();
  renderPaneGrid();
}

function assignRunToPane(paneId, runId) {
  const pane = getPaneById(paneId);
  const run = state.runs.get(runId);
  if (!pane || !run) {
    return;
  }
  resetPaneContent(pane);
  pane.runId = runId;
  pane.source = getAvailableSource(run, pane.source);
  pane.text = (run.log_contents || {})[pane.source] || "";
  saveLayoutState();
  syncActivePaneSearchState();
  updateLogSearchUi();
  renderRuns();
  renderPaneGrid();
  void loadPaneChunk(pane.id);
}

function assignRunToActivePane(runId) {
  assignRunToPane(state.activePaneId, runId);
}

function enqueuePaneLogText(runId, source, text) {
  if (!text) {
    return;
  }
  state.panes.forEach((pane) => {
    if (pane.runId !== runId || pane.source !== source) {
      return;
    }
    const segments = text.split(/(\s+)/).filter(Boolean);
    pane.queue.push(...segments);
    pane.endOffset += text.length;
    pane.fileSize += text.length;
    if (!pane.queueTimer) {
      pane.queueTimer = window.setInterval(() => flushPaneQueue(pane.id), 14);
    }
  });
}

function getPaneFlushSize(pane) {
  const run = getRunForPane(pane);
  if (run?.status === "completed") {
    return pane.queue.length;
  }
  if (pane.queue.length > 1200) {
    return 160;
  }
  if (pane.queue.length > 500) {
    return 72;
  }
  if (pane.queue.length > 160) {
    return 24;
  }
  return 6;
}

function flushPaneQueue(paneId) {
  const pane = getPaneById(paneId);
  if (!pane) {
    return;
  }
  const chunk = pane.queue.splice(0, getPaneFlushSize(pane)).join("");
  if (chunk) {
    const streamEl = getPaneStreamElement(paneId);
    const restoreScrollTop = pane.autoFollow ? null : streamEl?.scrollTop ?? null;
    pane.text += chunk;
    if (paneId === state.activePaneId) {
      syncActivePaneSearchState();
      updateLogSearchUi();
    }
    renderPaneStream(paneId, {
      stickToBottom: pane.autoFollow,
      restoreScrollTop,
    });
  }
  if (pane.queue.length === 0 && pane.queueTimer) {
    window.clearInterval(pane.queueTimer);
    pane.queueTimer = null;
  }
}

function computeLocalStats(runs) {
  const stats = {
    total: runs.length,
    queued: 0,
    running: 0,
    verify: 0,
    completed: 0,
    active: 0,
  };
  runs.forEach((run) => {
    stats[run.status] += 1;
  });
  stats.active = stats.running + stats.verify;
  return stats;
}

function resizePaneLayout(mode) {
  const nextCount = Number(mode);
  if (!LAYOUT_LABELS[nextCount]) {
    return;
  }
  const visibleRuns = filteredRuns();
  const current = Array.from({ length: nextCount }, (_unused, index) => {
    const pane = makePane(index + 1);
    const run = visibleRuns[index];
    if (run) {
      pane.runId = run.run_id;
      pane.source = getAvailableSource(run, "run.log");
      pane.text = (run.log_contents || {})[pane.source] || "";
    }
    return pane;
  });

  state.panes.forEach((pane) => {
    if (pane.queueTimer) {
      window.clearInterval(pane.queueTimer);
    }
  });

  state.layoutMode = nextCount;
  state.panes = current;
  state.activePaneId = state.panes[0]?.id || "pane-1";
  saveLayoutState();
  renderRuns();
  renderPaneGrid();
  state.panes
    .filter((pane) => pane.runId)
    .forEach((pane) => {
      void loadPaneChunk(pane.id);
    });
}

function showLogSearch() {
  if (!getRunForPane(getActivePane())) {
    return;
  }
  logSearch.classList.remove("hidden");
  logSearchInput.focus();
  logSearchInput.select();
  syncActivePaneSearchState();
  updateLogSearchUi();
}

function closeLogSearch({ clearQuery = true } = {}) {
  logSearch.classList.add("hidden");
  if (clearQuery) {
    state.logSearchQuery = "";
    state.activeLogMatchIndex = -1;
    logSearchInput.value = "";
    syncActivePaneSearchState();
    updateLogSearchUi();
    renderPaneStream(state.activePaneId);
  }
}

function jumpLogMatch(direction) {
  if (state.logSearchMatchCount === 0) {
    return;
  }
  state.activeLogMatchIndex =
    (state.activeLogMatchIndex + direction + state.logSearchMatchCount) % state.logSearchMatchCount;
  renderPaneStream(state.activePaneId);
  scrollActiveLogMatchIntoView();
}

async function loadSnapshot() {
  const response = await fetch("/api/snapshot");
  const snapshot = await response.json();
  state.logDir = snapshot.log_dir;
  state.streamCursor = snapshot.cursor;
  state.runs = new Map(snapshot.runs.map((run) => [run.run_id, run]));
  reconcilePanesWithRuns();

  if (!state.panes.some((pane) => pane.runId) && snapshot.runs.length > 0) {
    const firstPane = state.panes[0];
    firstPane.runId = snapshot.runs[0].run_id;
    firstPane.source = getAvailableSource(snapshot.runs[0], "run.log");
    firstPane.text = (snapshot.runs[0].log_contents || {})[firstPane.source] || "";
  }

  saveLayoutState();
  updateHeader(snapshot);
  renderRuns();
  renderPaneGrid();

  await Promise.all(
    state.panes
      .filter((pane) => pane.runId)
      .map((pane) => loadPaneChunk(pane.id))
  );
}

function handleStreamEvent(eventType, payload, lastEventId) {
  state.streamCursor = Number(lastEventId || state.streamCursor);

  if (eventType === "run_update") {
    mergeRun(payload.run);
    reconcilePanesWithRuns();
    saveLayoutState();
    renderRuns();
    const matchingPaneIds = state.panes.filter((pane) => pane.runId === payload.run.run_id).map((pane) => pane.id);
    const needsFullRefresh = matchingPaneIds.some((paneId) => paneNeedsStructuralRefresh(paneId));
    if (needsFullRefresh) {
      renderPaneGrid();
    } else {
      matchingPaneIds.forEach((paneId) => {
        updatePaneChrome(paneId);
      });
      if (matchingPaneIds.includes(state.activePaneId)) {
        syncActivePaneSearchState();
        updateLogSearchUi();
      }
    }
    if (payload.run.status === "completed") {
      state.panes
        .filter((pane) => pane.runId === payload.run.run_id && pane.queue.length > 0)
        .forEach((pane) => flushPaneQueue(pane.id));
    }
    updateHeader({
      log_dir: state.logDir,
      stats: computeLocalStats(allRuns()),
    });
    return;
  }

  if (eventType === "run_removed") {
    state.runs.delete(payload.run_id);
    state.panes.forEach((pane) => {
      if (pane.runId === payload.run_id) {
        resetPaneContent(pane);
        pane.runId = null;
      }
    });
    saveLayoutState();
    renderRuns();
    renderPaneGrid();
    updateHeader({
      log_dir: state.logDir,
      stats: computeLocalStats(allRuns()),
    });
    return;
  }

  if (eventType === "log_append") {
    enqueuePaneLogText(payload.run_id, payload.source || "run.log", payload.text);
  }
}

function connectStream() {
  if (state.stream) {
    state.stream.close();
  }

  const stream = new EventSource(`/api/stream?cursor=${state.streamCursor}`);
  state.stream = stream;
  const eventTypes = ["connected", "run_update", "log_append", "file_changed", "status_changed", "run_removed"];
  eventTypes.forEach((eventType) => {
    stream.addEventListener(eventType, (event) => {
      const payload = event.data ? JSON.parse(event.data) : {};
      handleStreamEvent(eventType, payload, event.lastEventId);
    });
  });

  stream.onerror = () => {
    stream.close();
    window.setTimeout(connectStream, 1500);
  };
}

const statsRoot = document.getElementById("stats");
const runsRoot = document.getElementById("runs");
const emptyState = document.getElementById("emptyState");
const paneGrid = document.getElementById("paneGrid");
const paneSummary = document.getElementById("paneSummary");
const searchInput = document.getElementById("searchInput");
const searchMode = document.getElementById("searchMode");
const statusFilter = document.getElementById("statusFilter");
const layoutModeSelect = document.getElementById("layoutModeSelect");
const logSearch = document.getElementById("logSearch");
const logSearchInput = document.getElementById("logSearchInput");
const logSearchCount = document.getElementById("logSearchCount");
const logSearchPrev = document.getElementById("logSearchPrev");
const logSearchNext = document.getElementById("logSearchNext");
const logSearchClose = document.getElementById("logSearchClose");

loadLayoutState();

searchInput.addEventListener("input", renderRuns);
searchMode.addEventListener("change", renderRuns);
statusFilter.addEventListener("change", renderRuns);

layoutModeSelect.addEventListener("change", () => {
  resizePaneLayout(layoutModeSelect.value);
});

logSearchInput.addEventListener("input", () => {
  state.logSearchQuery = logSearchInput.value;
  state.activeLogMatchIndex = 0;
  syncActivePaneSearchState();
  updateLogSearchUi();
  renderPaneStream(state.activePaneId);
});

logSearchInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    jumpLogMatch(event.shiftKey ? -1 : 1);
    return;
  }
  if (event.key === "Escape") {
    event.preventDefault();
    closeLogSearch();
  }
});

logSearchPrev.addEventListener("click", () => jumpLogMatch(-1));
logSearchNext.addEventListener("click", () => jumpLogMatch(1));
logSearchClose.addEventListener("click", () => closeLogSearch());

document.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "f" && getRunForPane(getActivePane())) {
    event.preventDefault();
    showLogSearch();
    return;
  }
  if (event.key === "Escape" && !logSearch.classList.contains("hidden")) {
    event.preventDefault();
    closeLogSearch();
  }
});

layoutModeSelect.value = String(state.layoutMode);
paneSummary.textContent = getPaneSummaryLabel();

loadSnapshot().then(connectStream);
