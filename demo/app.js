(() => {
  "use strict";
  const $ = (selector) => document.querySelector(selector);
  const dom = {form: $("#searchForm"), question: $("#question"), button: $("#searchButton"), collectionTitle: $("#collectionTitle"), scopeCount: $("#scopeCount"), presetList: $("#presetList"), loading: $("#loadingCard"), loadingTitle: $("#loadingTitle"), loadingSteps: $("#loadingSteps"), workspace: $("#workspace"), answer: $("#answerCard"), noAnswer: $("#noAnswer"), videoTitle: $("#videoTitle"), reason: $("#reason"), watchRange: $("#watchRange"), coreStart: $("#coreStart"), watchButton: $("#watchButton"), watchButtonText: $("#watchButtonText"), toast: $("#toast")};
  let collectionId = "demo-mcp";
  let loadingTimer;
  const escapeHTML = (value) => String(value ?? "").replace(/[&<>'"]/g, (character) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"})[character]);
  async function api(path, options = {}) {
    const response = await demoFetch(path, {headers: {"Content-Type": "application/json"}, ...options, body: options.body && typeof options.body !== "string" ? JSON.stringify(options.body) : options.body});
    const value = await response.json();
    if (!response.ok) throw Object.assign(new Error(value.message || "请求失败"), {code: value.error});
    return value;
  }
  function formatTime(seconds) {
    const total = Math.max(0, Math.floor(Number(seconds) || 0));
    const hours = Math.floor(total / 3600), minutes = Math.floor((total % 3600) / 60), rest = total % 60;
    return hours ? `${hours}:${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}` : `${minutes}:${String(rest).padStart(2, "0")}`;
  }
  function toast(message) { dom.toast.textContent = message; dom.toast.hidden = false; window.setTimeout(() => { dom.toast.hidden = true; }, 2600); }
  function startLoading() {
    $("#learningMap").hidden = true;
    $("#reportGaps").hidden = true;
    $("#intentNote").hidden = true; $("#clarification").hidden = true;
    const labels = ["正在理解问题", "正在搜索收藏内容", "正在核实相关片段，宽泛主题需要检查多个学习方向"];
    let step = 0;
    dom.workspace.hidden = true; dom.loading.hidden = false; dom.loadingTitle.textContent = labels[step];
    [...dom.loadingSteps.children].forEach((item, index) => item.classList.toggle("active", index === step));
    loadingTimer = window.setInterval(() => { step = Math.min(step + 1, labels.length - 1); dom.loadingTitle.textContent = labels[step]; [...dom.loadingSteps.children].forEach((item, index) => item.classList.toggle("active", index <= step)); }, 900);
  }
  function stopLoading() { window.clearInterval(loadingTimer); dom.loading.hidden = true; }
  function render(response) {
    stopLoading(); dom.workspace.hidden = false; dom.answer.hidden = true; dom.noAnswer.hidden = true;
    $("#learningMap").hidden = true;
    const planned = ["intent_v2", "chapter_v1"].includes(response.version);
    if (planned) {
      const labels = {partial: "已找到部分讲解，尚有未覆盖方向。", supported: "已找到本次计划对应的讲解，不代表完整课程。", no_evidence: "本次未找到充分证据，不代表收藏里肯定没有。"};
      $("#intentNote").textContent = `本次理解：${response.plan.goal}。${labels[response.status] || ""}${response.plan.assumptions.length ? ` 本次默认：${response.plan.assumptions.join("；")}。` : ""}`;
      $("#intentNote").hidden = false;
    }
    if (response.learning_map?.groups?.length) {
      renderMap(response.learning_map, response.question);
      const missing = (response.coverage || []).filter(row => row.status !== "supported");
      const statusText = {not_found: "未找到充分证据", partial_evidence: "有相关内容，但不足以完整讲解", technical_failure: "检索或定位失败，可重试", dependency_missing: "前置方向缺少证据"};
      $("#reportGaps").hidden = !missing.length;
      $("#reportGaps").innerHTML = missing.length ? `<h3>尚未覆盖的方向</h3><ul>${missing.map(row => `<li>${escapeHTML(row.title)}：${escapeHTML(statusText[row.status] || row.status)}</li>`).join("")}</ul>` : "";
      dom.workspace.scrollIntoView({behavior: "smooth", block: "start"}); return;
    }
    if (response.intent?.mode === "topic" && !planned) {
      $("#intentNote").textContent = "已按学习主题检查相关方向，但当前收藏中未找到足够的片段证据。可修改上方问题。";
      $("#intentNote").hidden = false;
    }
    if (!response.answerable || !response.results?.length) {
      dom.noAnswer.querySelector("h2").textContent = planned ? "本次检索未找到足够回答问题的片段证据。" : "当前收藏的视频中，没有找到足够回答这个问题的内容。";
      dom.noAnswer.querySelector("div > p:last-child").textContent = planned ? "可补充你关心的具体问题，或查看收藏列表；未找到不等于内容不存在。" : "无需继续逐个视频查找。";
      dom.noAnswer.hidden = false; dom.workspace.scrollIntoView({behavior: "smooth", block: "start"}); return;
    }
    const item = response.results[0];
    dom.videoTitle.textContent = item.video_title; dom.reason.textContent = item.reason;
    dom.watchRange.textContent = `${formatTime(item.recommended_watch_start)}–${formatTime(item.recommended_watch_end)}`;
    const playbackStart = item.playback_start ?? item.core_start;
    dom.coreStart.textContent = formatTime(playbackStart); dom.watchButtonText.textContent = `从 ${formatTime(playbackStart)} 开始观看`;
    dom.watchRange.previousElementSibling.textContent = item.localization ? "答案片段" : "建议观看";
    $(".watch-note").textContent = item.localization ? "按原始字幕定位回答问题的片段；播放从片段起点开始，范围不再包含无关的相邻知识点。" : "推荐起点用于快速进入主题；建议观看范围包含理解完整解释所需的相邻上下文。";
    dom.watchButton.href = `${item.source_url}&t=${Math.floor(playbackStart)}s`; dom.answer.hidden = false;
    dom.workspace.scrollIntoView({behavior: "smooth", block: "start"});
  }
  function watchLink(item, start = item.recommended_watch_start) {
    const url = new URL(item.source_url);
    if (url.protocol !== "https:" || !["www.youtube.com", "youtube.com"].includes(url.hostname)) return "#";
    url.searchParams.set("t", `${Math.floor(start)}s`);
    return escapeHTML(url.href);
  }
  function renderMap(map, question) {
    $("#learningMap").hidden = false;
    const route = buildLearningRoute(map.groups);
    const seconds = route.reduce((total, item) => total + item.recommended_watch_end - item.recommended_watch_start, 0);
    $("#reportQuestion").textContent = question;
    const chapters = map.groups.some(group => group.clips.some(item => item.chapter));
    $("#reportSummary").textContent = `${map.groups.length} 个${chapters ? "内容单元" : "知识点"} · ${new Set(route.map(item => item.video_id)).size} 个来源视频 · 建议观看去重后约 ${formatTime(seconds)}`;
    $("#mapNotice").textContent = `以下按建议学习顺序整理，内容仅来自当前收藏的字幕，不代表完整课程。${map.failed_directions ? `另有 ${map.failed_directions} 个方向检索失败，本报告暂未纳入，可重试。` : ""}`;
    $("#reportOutline").innerHTML = map.groups.map((group, index) => `<a href="#report-section-${index}">${String(index + 1).padStart(2, "0")} ${escapeHTML(group.title)}</a>`).join("");
    $("#reportSections").innerHTML = map.groups.map((group, index) => `<section class="report-section" id="report-section-${index}" aria-labelledby="report-heading-${index}">
      <span class="report-number">${String(index + 1).padStart(2, "0")}</span><div><h3 id="report-heading-${index}">${escapeHTML(group.title)}</h3>
      <p class="report-focus">本节回答：${escapeHTML(group.question)}</p>${group.points ? group.points.map(point => `<p class="report-explanation">${escapeHTML(point.question)}<br>${escapeHTML(point.reason)}</p>`).join("") : ""}${group.clips.map(item => `${group.points ? "" : `<p class="report-explanation">${escapeHTML(item.reason)}</p>`}${item.chapter ? chapterSource(item) : `<div class="report-source"><span>视频依据</span><a href="${watchLink(item)}" target="_blank" rel="noopener">观看 ${formatTime(item.recommended_watch_start)}–${formatTime(item.recommended_watch_end)} ↗</a><small>${escapeHTML(item.video_title)}</small></div>`}`).join("")}</div></section>`).join("");
  }
  function chapterSource(item) {
    const c = item.chapter;
    const full = `<a href="${watchLink(item, c.start)}" target="_blank" rel="noopener">观看完整内容单元 ${formatTime(c.start)}–${formatTime(c.end)} · ${formatTime(c.end-c.start)} ↗</a>`;
    const key = `<a href="${watchLink(item, item.key_start)}" target="_blank" rel="noopener">直达关键点 ${formatTime(item.key_start)}–${formatTime(item.key_end)} ↗</a>`;
    return `<div class="report-source chapter-source"><span>${item.watch_mode === "chapter" ? "建议先看完整讲解" : "具体问题 · 优先直达关键点"}</span>${item.watch_mode === "chapter" ? full+key : key+full}<small>${escapeHTML(item.video_title)}</small></div><details class="chapter-details"><summary>本单元内容与内部导航</summary><p>${escapeHTML(c.summary)}</p><ul>${c.points.map(point => `<li><a href="${watchLink(item, point.start)}" target="_blank" rel="noopener">${formatTime(point.start)} ${escapeHTML(point.title)} ↗</a></li>`).join("")}</ul></details>`;
  }
  async function loadCollection() {
    const library = await api("/api/library"), collection = library.collections?.[0];
    collectionId = collection?.collection_id || "demo-mcp"; dom.collectionTitle.textContent = collection?.title || "收藏范围";
    dom.scopeCount.textContent = `· ${collection?.video_ids?.length || 0} 个视频已纳入检索`;
    $(".boundary span").textContent = `${String(collection?.video_ids?.length || 0).padStart(2, "0")} VIDEOS · LOCAL INDEX`;
    dom.question.placeholder = library.demo_questions?.[0]?.question || "描述你想从收藏视频中找到的知识";
    if (library.videos?.some(video => video.chapters?.length)) {
      $(".promise").textContent = "按视频内容整理相对独立的讲解单元。先看完整内容，也能直达问题关键点。";
    }
    dom.presetList.innerHTML = (library.demo_questions || []).map((item) => `<button type="button" data-question="${escapeHTML(item.question)}"><small>${escapeHTML(item.label)}</small>${escapeHTML(item.question)}</button>`).join("");
  }
  dom.presetList.addEventListener("click", (event) => { const button = event.target.closest("[data-question]"); if (!button) return; dom.question.value = button.dataset.question; dom.question.focus(); });
  $("#editQuestion").addEventListener("click", () => { dom.question.focus(); dom.question.scrollIntoView({behavior: "smooth", block: "center"}); });
  dom.form.addEventListener("submit", async (event) => {
    event.preventDefault(); dom.button.disabled = true; dom.button.querySelector("span").textContent = "正在查找"; startLoading();
    try { render(await api("/api/search", {method: "POST", body: {question: dom.question.value, collection_id: collectionId}})); }
    catch (error) {
      stopLoading();
      if (error.code === "clarification_required") {
        $("#clarificationText").textContent = error.message; $("#clarification").hidden = false;
        $("#editQuestion").focus();
      } else {
        dom.workspace.hidden = false; dom.answer.hidden = true; dom.noAnswer.hidden = true;
        $("#intentNote").textContent = `本次请求未能完成：${error.message} 这不代表收藏里没有答案，请重试。`;
        $("#intentNote").hidden = false;
      }
    }
    finally { dom.button.disabled = false; dom.button.querySelector("span").textContent = "从收藏中查找"; }
  });
  loadCollection().catch((error) => toast(error.message));
})();
