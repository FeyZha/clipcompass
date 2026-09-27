import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {videoId,watchURL,endpoint,indexedVideos,pendingTasks,playbackTarget,playVideo} from './core.mjs';
const a='bHcJCp2Fyxs', b='zzbr1h9sF54', url=id=>`https://www.youtube.com/watch?v=${id}`;
assert.equal(videoId(url(a)),a);
assert.equal(videoId(`https://youtu.be/${a}`),a);
assert.equal(videoId(`https://www.youtube.com/shorts/${a}`),a);
for(const bad of ['javascript:alert(1)','https://youtube.com.evil.test/watch?v='+a,'https://evil@youtube.com/watch?v='+a,'https://youtube.com:444/watch?v='+a,'https://youtube.com/watch?v=x']) assert.equal(videoId(bad),null);
for(const n of [NaN,Infinity,-1])assert.throws(()=>watchURL(url(a),n));
assert.equal(watchURL(url(a),33.8),url(a)+'&t=33s');
for(const bad of ['https://evil.test',80,65536,'1485/path'])assert.throws(()=>endpoint(bad));
assert.equal(endpoint(1485),'http://127.0.0.1:1485');
assert.deepEqual(indexedVideos({collections:[{video_ids:[a]}],videos:[{video_id:a},{video_id:b}]}),[{video_id:a}]);
const tasksLibrary = {collections:[{video_ids:[a]}],videos:[{video_id:a}],imports:[
  {video_id:a,status:'ready'},{video_id:b,status:'chapters'},{video_id:'failed',status:'failed'},
  {video_id:'ready-only',status:'ready'},{video_id:'queued',status:'queued'}]};
assert.deepEqual(pendingTasks(tasksLibrary,[{video_id:a},{video_id:b},{video_id:'offline'}]).map(v=>v.video_id),
  [b,'offline','failed','queued']);
assert.deepEqual(pendingTasks(null),[]);
assert.deepEqual(pendingTasks({...tasksLibrary,imports:[{video_id:b,status:'ready'}]},[{video_id:b}]),[]);
assert.equal(playbackTarget([{id:1,url:'https://example.org',active:true},{id:2,url:url(a)}],1,3),null);
assert.equal(playbackTarget([{id:1,url:'https://example.org'},{id:2,url:url(a)}],1,2).id,2);
const calls=[];let tabs=[{id:1,url:'https://example.org',active:true}], saved={};
const api={tabs:{query:async()=>tabs,create:async opts=>{calls.push(['create',opts]);return{id:3};},update:async(id,opts)=>calls.push(['update',id,opts])},storage:{session:{get:async()=>saved,set:async v=>Object.assign(saved,v)}},scripting:{executeScript:async()=>[{result:true}]}};
await playVideo(api,url(a),30,7);
assert.equal(calls[0][0],'create');assert.equal(calls[0][1].windowId,7);assert.equal(saved['playback:7'],3);
tabs=[{id:1,url:'https://example.org',active:true},{id:3,url:url(a)}];calls.length=0;
await playVideo(api,url(a),60,7);assert.deepEqual(calls,[['update',3,{active:true}]]);
await playVideo(api,url(b),10,7);assert.equal(calls.at(-1)[2].url,url(b)+'&t=10s');
tabs=[{id:1,url:'https://example.org',active:true},{id:3,url:'https://other.org'}];calls.length=0;
await playVideo(api,url(a),0,7);assert.equal(calls[0][0],'create');
tabs=[{id:1,url:url(a),active:true}];api.scripting.executeScript=async()=>{throw Error('no player');};calls.length=0;
await playVideo(api,url(a),10,7);assert.equal(calls[0][2].url,url(a)+'&t=10s');
const manifest=JSON.parse(readFileSync(new URL('./manifest.json',import.meta.url)));
assert.equal(manifest.name,'ClipCompass | 视频知识库与学习助手');
for (const size of [16,32,48,128]) {
  assert.equal(manifest.action.default_icon[size],manifest.icons[size]);
  const png=readFileSync(new URL(manifest.icons[size],import.meta.url));
  assert.equal(png.subarray(0,8).toString('hex'),'89504e470d0a1a0a');
  assert.equal(png.readUInt32BE(16),size);
  assert.equal(png.readUInt32BE(20),size);
}
for (const path of ['./panel.html','../library/static/index.html','../library/static/collection.html']) {
  const html=readFileSync(new URL(path,import.meta.url),'utf8');
  assert(html.includes('ClipCompass | 视频知识库与学习助手'));
  assert(!html.includes('片刻'));
}
assert.equal(manifest.side_panel.default_path,'panel.html');
assert(!manifest.permissions.includes('tabs'));assert(!manifest.host_permissions.includes('<all_urls>'));
console.log('Extension checks passed: URL/port boundaries, collection scope, non-YouTube tab preservation, managed tab reuse, seek fallback, least permissions.');
