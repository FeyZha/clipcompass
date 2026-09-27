export function videoId(value) {
  try {
    const url = new URL(value);
    if (url.protocol !== 'https:' || url.username || url.password || url.port) return null;
    let id;
    if (url.hostname === 'youtu.be') id = url.pathname.slice(1);
    else if (['www.youtube.com', 'youtube.com'].includes(url.hostname)) {
      id = url.pathname === '/watch' ? url.searchParams.get('v') : /^\/shorts\/([\w-]{11})$/.exec(url.pathname)?.[1];
    }
    return /^[\w-]{11}$/.test(id || '') ? id : null;
  } catch { return null; }
}
export function watchURL(value, start = 0) {
  const id = videoId(value);
  if (!id || !Number.isFinite(start) || start < 0) throw new Error('视频链接或时间无效。');
  return `https://www.youtube.com/watch?v=${id}&t=${Math.floor(start)}s`;
}
export function endpoint(port) {
  if (!/^\d{2,5}$/.test(String(port)) || Number(port) < 1024 || Number(port) > 65535) throw new Error('端口须为 1024–65535。');
  return `http://127.0.0.1:${Number(port)}`;
}
export function time(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '时间未知';
  const n = Math.floor(seconds);
  return `${Math.floor(n / 60)}:${String(n % 60).padStart(2, '0')}`;
}
export function indexedVideos(library) {
  const ids = new Set(library?.collections?.[0]?.video_ids || []);
  return (library?.videos || []).filter(v => ids.has(v.video_id));
}
export function pendingTasks(library, bookmarks = []) {
  const ready = new Set(indexedVideos(library).map(v=>v.video_id));
  return [...new Map([...bookmarks,...(library?.imports || [])].map(v=>[v.video_id,v])).values()]
    .filter(v=>!ready.has(v.video_id) && v.status !== 'ready');
}
// Only an active YouTube tab or the extension's own playback tab may be reused.
export function playbackTarget(tabs, activeId, managedId) {
  return tabs.find(t => t.id === activeId && videoId(t.url)) || tabs.find(t => t.id === managedId && videoId(t.url)) || null;
}
export async function playVideo(api, source, start, windowId) {
  const url = watchURL(source, start);
  const tabs = await api.tabs.query({windowId});
  const key = `playback:${windowId}`;
  const saved = await api.storage.session.get(key);
  const target = playbackTarget(tabs, tabs.find(t => t.active)?.id, saved[key]);
  if (target && videoId(target.url) === videoId(url)) {
    let jumped = false;
    try {
      const result = await api.scripting.executeScript({target: {tabId: target.id}, args: [videoId(url), start], func: (id, seconds) => {
        if (new URL(location.href).searchParams.get('v') !== id) return false;
        const player = document.querySelector('video');
        if (!player || player.readyState < 1 || !Number.isFinite(player.duration) || seconds >= player.duration) return false;
        player.currentTime = seconds;
        return true;
      }});
      jumped = result?.[0]?.result === true;
    } catch { /* Navigation or missing player: use YouTube's timestamp URL. */ }
    await api.tabs.update(target.id, jumped ? {active: true} : {url, active: true});
  } else if (target) await api.tabs.update(target.id, {url, active: true});
  const tab = target || await api.tabs.create({windowId, url, active: true});
  await api.storage.session.set({[key]: tab.id});
  return tab.id;
}
