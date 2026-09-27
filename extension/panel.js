import {videoId, watchURL, endpoint, time, indexedVideos, pendingTasks, playVideo} from './core.mjs';
const $ = s => document.querySelector(s);
const el = (tag, text, cls) => { const n = document.createElement(tag); if (text != null) n.textContent = text; if (cls) n.className = cls; return n; };
let state = {port:1485, question:'', report:null, view:'search', scroll:0, bookmarks:[]};
let library = null, connected = false, busy = false, current = null, windowId;
let saveQueue = Promise.resolve();
function save() {
  const snapshot = structuredClone(state);
  saveQueue = saveQueue.catch(() => {}).then(() => chrome.storage.local.set({compass:snapshot}));
  return saveQueue;
}
function notice(message) { $('#notice').textContent = message; $('#notice').hidden = !message; }
function show(view) {
  state.view = ['search','library','settings'].includes(view) ? view : 'search';
  for (const name of ['search','library','settings']) $(`#${name}View`).hidden = name !== state.view;
  document.querySelectorAll('[data-view]').forEach(b => { if(b.dataset.view === state.view) b.setAttribute('aria-current','page'); else b.removeAttribute('aria-current'); });
}
async function api(path, body) {
  const response = await fetch(endpoint(state.port) + path, {method:body ? 'POST':'GET', headers:body ? {'Content-Type':'application/json', ...(['/api/imports','/api/obsidian'].includes(path) ? {'X-ClipCompass-Import':library?.import_token || ''}:{})}:{}, body:body ? JSON.stringify(body):undefined, signal:AbortSignal.timeout(body ? 270000:8000), redirect:'error'});
  const data = await response.json();
  if (!response.ok) throw Object.assign(new Error(data.message || '请求失败'), {code:data.error});
  return data;
}
async function connect() {
  connected = false; $('#searchButton').disabled = true;
  $('#connection').textContent = '正在连接…';
  try {
    const data = await api('/api/library');
    if (!Array.isArray(data.videos) || !data.collections?.[0]?.collection_id) throw new Error('收藏格式不正确');
    library = data; connected = true;
    await chrome.storage.local.set({libraryCache:data});
    $('#connection').textContent = `已连接 · ${data.collections[0].title}`;
  } catch {
    $('#connection').textContent = '本地服务未连接。请启动服务后再检查连接；当前可查看上次缓存。';
  }
  $('#scope').textContent = `${indexedVideos(library).length} 条可检索视频${connected ? '':' · 离线缓存'}`;
  $('#searchButton').disabled = !connected || busy;
  $('#presets').replaceChildren(...(library?.demo_questions || []).map(p => button(p.question, () => { $('#question').value = state.question = p.question; save().catch(storageError); $('#question').focus(); }, 'quiet')));
  renderLibrary();
  if (connected && library.imports_enabled) await syncBookmarks();
}
let syncing = false, polling = false;
async function syncBookmarks() {
  if(syncing || !connected || !library?.imports_enabled) return;
  syncing = true;
  try {
    const known = new Set([...indexedVideos(library),...(library.imports || [])].map(v=>v.video_id));
    for(const bookmark of state.bookmarks) {
      if(known.has(bookmark.video_id)) continue;
      const result = await api('/api/imports',bookmark);
      library.imports ||= []; library.imports.push(result.job); known.add(bookmark.video_id);
    }
    renderLibrary();
  } catch(e) { notice(`链接已保留，等待提交整理：${e.message}`); }
  finally { syncing = false; }
}
async function refreshImports() {
  if(polling || busy) return;
  polling = true;
  const port = state.port;
  try {
    const data = await api('/api/library');
    if(port !== state.port) return;
    if(!Array.isArray(data.videos) || !data.collections?.[0]?.collection_id) throw new Error('收藏格式不正确');
    library = data; connected = true;
    $('#connection').textContent = `已连接 · ${data.collections[0].title}`;
    $('#scope').textContent = `${indexedVideos(library).length} 条可检索视频`;
    $('#searchButton').disabled = busy;
    await chrome.storage.local.set({libraryCache:data});
    await syncBookmarks(); renderLibrary();
  } catch {
    if(port !== state.port) return;
    connected = false; $('#searchButton').disabled = true;
    $('#connection').textContent = '本地服务连接中断，正在尝试重新连接。';
    $('#scope').textContent = '服务连接中断 · 收藏保留，连接恢复后继续整理';
    renderLibrary();
  } finally { polling = false; }
}
function storageError() { notice('浏览器保存失败，请检查可用空间；当前界面仍可使用。'); }
function button(text, action, cls) {
  const b = el('button',text,cls); b.type='button';
  b.addEventListener('click', () => Promise.resolve().then(action).catch(e => notice(e.message || '操作失败，请重试。')));
  return b;
}
async function watch(source, start) {
  state.scroll = window.scrollY;
  await save();
  await playVideo(chrome, source, start, windowId);
  notice('已定位到 YouTube。报告保留在这里；如视频暂停，请按播放。');
}
function chapterButtons(clip) {
  const box = el('div',null,'actions');
  const c = clip.chapter;
  const full = c && button(`看完整讲解 ${time(c.start)}–${time(c.end)}`, () => watch(clip.source_url,c.start), clip.watch_mode === 'chapter' ? '' : 'secondary');
  const start = clip.key_start ?? clip.playback_start ?? clip.recommended_watch_start;
  const key = button(`关键点 ${time(start)} ↗`, () => watch(clip.source_url,start), clip.watch_mode === 'chapter' ? 'secondary' : '');
  box.append(...(clip.watch_mode === 'chapter' ? [full,key] : [key,full]).filter(Boolean));
  return box;
}
function renderReport() {
  const root = $('#report'); root.replaceChildren();
  const r = state.report; $('#searchView').classList.toggle('has-report',Boolean(r)); if (!r) return;
  const head = el('div',null,'report-head');
  head.append(el('span','来自你的收藏','eyebrow'),el('h2',r.question),el('p',r.plan?.goal || '来自收藏视频的讲解','interpretation'));
  root.append(head);
  if (!r.results?.length) { root.append(el('p','本次未找到充分证据，不代表收藏中一定没有。可以补充问题后重试。')); return; }
  const groups = r.learning_map?.groups?.length ? r.learning_map.groups : r.results.map(c => ({title:c.video_title,clips:[c]}));
  head.append(el('p',`${groups.length} 个内容单元 · ${new Set(r.results.map(c=>c.video_id)).size} 个视频。建议顺序，不代表完整课程。`,'muted'));
  groups.forEach((g,index) => {
    const section = el('section',null,'chapter');
    section.append(el('span',String(index+1).padStart(2,'0'),'number'),el('h3',g.title));
    for (const clip of g.clips) {
      section.append(el('p',clip.reason),el('p',clip.video_title,'source'),chapterButtons(clip));
      if (clip.chapter) {
        const details = el('details'); details.append(el('summary','本单元讲什么 · 内部导航'),el('p',clip.chapter.summary));
        for (const point of clip.chapter.points || []) details.append(button(`${time(point.start)}  ${point.title}`,()=>watch(clip.source_url,point.start)));
        section.append(details);
      }
    }
    root.append(section);
  });
  const gaps = (r.coverage || []).filter(x=>x.status!=='supported');
  if(gaps.length) root.append(el('p',`尚未覆盖：${gaps.map(x=>x.title).join('；')}。`,'gaps'));
}
function renderLibrary() {
  renderObsidian();
  const rows = pendingTasks(library,state.bookmarks);
  const active = rows.filter(v=>['captions','chapters','indexing'].includes(v.status)).length;
  const failed = rows.filter(v=>v.status==='failed').length;
  $('#libraryCount').textContent = `${active} 条处理中 · ${rows.length-active-failed} 条等待 · ${failed} 条需处理${connected ? '' : ' · 离线状态，进度待更新'}`;
  $('#videos').replaceChildren(...rows.map(v=> {
    const box = el('article',null,'video');
    box.append(el('span',v.message || '等待连接服务后自动整理','tag'),el('h2',v.title));
    if(['captions','chapters','indexing'].includes(v.status)) {
      const progress = el('progress'); progress.setAttribute('aria-label',v.message); box.append(progress);
    }
    const actions = el('div',null,'actions');
    if(v.status==='failed') actions.append(button('重试整理',async()=> {
      await api('/api/imports',{...v,retry:true}); await refreshImports();
    },'secondary'));
    if(!v.status) actions.append(button('移除链接',async()=> { state.bookmarks = state.bookmarks.filter(x=>x.video_id!==v.video_id); await save(); renderLibrary(); },'remove'));
    if(actions.childElementCount) box.append(actions);
    return box;
  }));
  if(!rows.length) $('#videos').append(el('p',connected ? '暂无待整理任务。完成的内容已保留，可直接在「找知识」中搜索。' : '暂无缓存任务。连接恢复后更新整理状态。','muted'));
  renderCurrent();
}
function renderCurrent() {
  const id = videoId(current?.url);
  const exists = id && [...indexedVideos(library),...state.bookmarks,...(library?.imports || [])].some(v=>v.video_id===id);
  $('#currentVideo').textContent = id ? current.title || '当前 YouTube 视频' : '切到 YouTube 视频页，即可收藏当前视频。';
  $('#saveCurrent').disabled = !id || exists;
  $('#saveCurrent').textContent = exists ? '已在收藏中' : '收藏当前视频 ＋';
}
async function refreshCurrent() {
  [current] = await chrome.tabs.query({active:true,windowId}); renderCurrent();
}
document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>{show(b.dataset.view);state.scroll=0;save().catch(storageError);window.scrollTo(0,0);}));
$('#question').addEventListener('input',()=>{state.question=$('#question').value;save().catch(storageError);});
$('#saveCurrent').addEventListener('click',async()=>{
  try {
    await refreshCurrent(); const id=videoId(current?.url); if(!id) throw new Error('请先打开一个 YouTube 视频。');
    if(![...indexedVideos(library),...state.bookmarks].some(v=>v.video_id===id)) state.bookmarks.unshift({video_id:id,title:current.title || 'YouTube 视频',source_url:watchURL(current.url),saved_at:Date.now()});
    await save();renderLibrary();
    notice(connected && library?.imports_enabled ? '已收藏，正在交给后台整理。关闭侧栏不影响已提交的任务。' : '已收藏，连接支持自动整理的本地服务后将自动提交。');
    await syncBookmarks();
  }catch(e){notice(e.message);}
});
$('#searchForm').addEventListener('submit',async event=>{
  event.preventDefault(); if(busy || !connected) return;
  const question=$('#question').value.trim(); if(!question) return;
  busy=true;state.question=question;state.report=null;state.scroll=0;notice('');renderReport();$('#progress').hidden=false;$('#searchButton').disabled=true;
  try {
    await save();
    const result=await api('/api/search',{question,collection_id:library.collections[0].collection_id});
    if(!Array.isArray(result.results)) throw new Error('服务返回的报告格式不正确。');
    state.report=result;await save();renderReport();
  }catch(e){notice(e.code==='clarification_required' ? `请补充：${e.message}` : `本次请求未完成：${e.message}。这不代表收藏中没有答案。`);}
  finally{busy=false;$('#progress').hidden=true;$('#searchButton').disabled=!connected;}
});
$('#settingsForm').addEventListener('submit',async event=>{
  event.preventDefault();if(busy){notice('请等待当前搜索完成后再切换连接。');return;}
  try{endpoint($('#port').value);state.port=Number($('#port').value);library=null;await chrome.storage.local.remove('libraryCache');await save();await connect();}catch(e){notice(e.message);}
});
let obsidianBusy = false;
function renderObsidian() {
  const status = library?.obsidian;
  const field = $('#vaultPath');
  if(!field.dataset.edited && document.activeElement !== field) field.value = status?.vault || '';
  $('#syncObsidian').disabled = !connected || !status || obsidianBusy;
  $('#disableObsidian').disabled = !connected || !status?.enabled || obsidianBusy;
  $('#obsidianStatus').textContent = !status ? '当前服务未提供 Obsidian 同步。' :
    `${status.enabled ? '自动同步已开启' : '自动同步已关闭'} · ${status.synced} 个视频已导出${connected ? '' : ' · 离线缓存'}。` +
    (status.error || '') + (status.errors || []).map(e=>`${e.title}：${e.message}`).join('；');
}
$('#vaultPath').addEventListener('input',()=>{$('#vaultPath').dataset.edited='true';});
async function configureObsidian(enabled) {
  if(obsidianBusy) return;
  obsidianBusy=true;renderObsidian();
  try {
    const result = await api('/api/obsidian',{enabled,vault:$('#vaultPath').value.trim()});
    library.obsidian=result.obsidian;delete $('#vaultPath').dataset.edited;
    notice(enabled ? '同步设置已保存，请查看下方同步结果。' : '自动同步已关闭，已导出的笔记保留。');
  } catch(e) {notice(e.message);}
  finally {obsidianBusy=false;renderObsidian();}
}
$('#obsidianForm').addEventListener('submit',event=>{event.preventDefault();configureObsidian(true);});
$('#disableObsidian').addEventListener('click',()=>configureObsidian(false));
$('#clearReport').addEventListener('click',async()=>{
  if(busy){notice('请等待当前请求完成后再清除。');return;}
  try{state.report=null;state.question='';state.scroll=0;$('#question').value='';await save();renderReport();notice('已清除本浏览器中的问题与报告；收藏保留。本地服务的诊断日志不受影响。');}catch{storageError();}
});
let scrollTimer;
window.addEventListener('scroll',()=>{clearTimeout(scrollTimer);scrollTimer=setTimeout(()=>{state.scroll=window.scrollY;save().catch(storageError);},180);},{passive:true});
async function init(){
  windowId=(await chrome.windows.getCurrent()).id;
  const stored=await chrome.storage.local.get(['compass','libraryCache']);
  if(stored.compass)state={...state,...stored.compass};library=stored.libraryCache || null;
  $('#question').value=state.question;$('#port').value=state.port;show(state.view);renderReport();renderLibrary();
  const scroll=state.scroll;await refreshCurrent();await connect();window.scrollTo(0,scroll);
  chrome.tabs.onActivated.addListener(info=>{if(info.windowId===windowId)refreshCurrent().catch(()=>{});});
  chrome.tabs.onUpdated.addListener((_id,info,tab)=>{if(tab.windowId===windowId && tab.active && (info.url || info.title || info.status==='complete'))refreshCurrent().catch(()=>{});});
  setInterval(refreshImports,5000);
}
init().catch(e=>notice(`侧栏初始化失败：${e.message}。请重新打开扩展。`));
