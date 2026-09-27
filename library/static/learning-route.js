"use strict";

// Keep the suggested learning order; overlapping footage is watched only once.
function buildLearningRoute(selected) {
  const route = [];
  selected.forEach(group => group.clips.forEach(item => {
    let clip = {...item, titles: [group.title], questions: [group.question]};
    let position = route.length;
    while (true) {
      const i = route.findIndex(old => old.video_id === clip.video_id && old.recommended_watch_start <= clip.recommended_watch_end && clip.recommended_watch_start <= old.recommended_watch_end);
      if (i === -1) break;
      const old = route[i];
      clip.recommended_watch_start = Math.min(clip.recommended_watch_start, old.recommended_watch_start);
      clip.recommended_watch_end = Math.max(clip.recommended_watch_end, old.recommended_watch_end);
      clip.titles = [...new Set([...old.titles, ...clip.titles])];
      clip.questions = [...new Set([...old.questions, ...clip.questions])];
      position = Math.min(position, i);
      route.splice(i, 1);
    }
    route.splice(position, 0, clip);
  }));
  return route;
}
