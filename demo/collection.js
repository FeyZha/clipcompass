"use strict";

function collectionVideos(library) {
  const ids = new Set(library.collections?.[0]?.video_ids || []);
  return (library.videos || []).filter(video => ids.has(video.video_id));
}
function filterVideos(videos, query) {
  const words = query.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
  return videos.filter(video => words.every(word => `${video.title || ""} ${video.creator || ""}`.toLocaleLowerCase().includes(word)));
}
function videoDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "时长未知";
  const total = Math.floor(seconds), minutes = Math.floor(total / 60);
  return minutes >= 60 ? `${Math.floor(minutes / 60)}:${String(minutes % 60).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}` : `${minutes}:${String(total % 60).padStart(2, "0")}`;
}
function safeVideoURL(value) {
  try {
    const url = new URL(value);
    return url.protocol === "https:" && ["youtube.com", "www.youtube.com", "youtu.be"].includes(url.hostname) && !url.username && !url.password ? url.href : null;
  } catch { return null; }
}

if (typeof module !== "undefined") module.exports = {collectionVideos, filterVideos, videoDuration, safeVideoURL};
if (typeof document !== "undefined") {
  const $ = selector => document.querySelector(selector);
  let videos = [];
  function render() {
    const matches = filterVideos(videos, $("#videoFilter").value);
    $("#listStatus").textContent = `显示 ${matches.length} / ${videos.length} 个视频`;
    $("#emptyLibrary").hidden = matches.length > 0;
    $("#emptyLibrary").textContent = videos.length ? "没有匹配的标题或作者。试试原视频标题中的词，或清空筛选查看全部。" : "当前收藏还没有视频。";
    const fragment = document.createDocumentFragment();
    // ponytail: render the small collection in full; paginate if thousands of videos make it slow.
    for (const video of matches) {
      const row = document.createElement("li"), number = document.createElement("span"), content = document.createElement("div");
      number.className = "video-number";
      number.textContent = String(videos.indexOf(video) + 1).padStart(2, "0");
      const title = document.createElement("h2"), meta = document.createElement("p"), url = safeVideoURL(video.source_url);
      title.textContent = video.title || "未命名视频";
      meta.textContent = `${video.creator || "作者未知"} · ${videoDuration(video.duration)} · YouTube`;
      content.append(title, meta);
      if (video.chapters?.length) {
        const details = document.createElement("details"), summary = document.createElement("summary"), list = document.createElement("ol");
        details.className = "chapter-details";
        summary.textContent = `查看 ${video.chapters.length} 个内容单元（按原字幕归纳）`;
        for (const chapter of video.chapters) {
          const item = document.createElement("li"), anchor = document.createElement(url ? "a" : "span");
          anchor.textContent = `${videoDuration(chapter.start)}–${videoDuration(chapter.end)} · ${chapter.title}`;
          if (url) { const target = new URL(url); target.searchParams.set("t", `${Math.floor(chapter.start)}s`); anchor.href = target.href; anchor.target = "_blank"; anchor.rel = "noopener noreferrer"; }
          item.append(anchor); list.append(item);
        }
        details.append(summary, list); content.append(details);
      }
      const link = document.createElement(url ? "a" : "span");
      link.className = "video-open";
      link.textContent = url ? "打开原视频 ↗" : "原视频链接不可用";
      if (url) { link.href = url; link.target = "_blank"; link.rel = "noopener noreferrer"; link.setAttribute("aria-label", `打开原视频：${video.title || "未命名视频"}（新标签页）`); }
      row.append(number, content, link);
      fragment.append(row);
    }
    $("#videoList").replaceChildren(fragment);
  }
  async function load() {
    $("#retryLibrary").hidden = true;
    $("#listStatus").textContent = "正在加载视频列表…";
    try {
      const response = await demoFetch("/api/library");
      if (!response.ok) throw new Error("无法读取收藏");
      const library = await response.json();
      if (!Array.isArray(library.collections) || !Array.isArray(library.videos)) throw new Error("收藏数据格式异常");
      videos = collectionVideos(library);
      $("#libraryTitle").textContent = library.collections[0]?.title || "当前收藏";
      $("#videoTotal").textContent = videos.length;
      const seconds = videos.reduce((sum, video) => sum + (Number.isFinite(video.duration) && video.duration >= 0 ? video.duration : 0), 0);
      $("#libraryDuration").textContent = `已知总时长 ${Math.floor(seconds / 3600)} 小时 ${Math.floor(seconds % 3600 / 60)} 分钟`;
      $("#videoFilter").disabled = $("#clearFilter").disabled = false;
      render();
    } catch {
      $("#libraryTitle").textContent = "暂时无法读取收藏";
      $("#listStatus").textContent = "收藏列表加载失败，请确认本地服务正在运行后重试。";
      $("#retryLibrary").hidden = false;
    }
  }
  $("#videoFilter").addEventListener("input", render);
  $("#clearFilter").addEventListener("click", () => { $("#videoFilter").value = ""; render(); $("#videoFilter").focus(); });
  $("#retryLibrary").addEventListener("click", load);
  load();
}
