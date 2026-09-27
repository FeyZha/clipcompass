"""Paired chapter/short-span functional checks, same eight-video scope, not accuracy gold."""
import json
import time
from evaluation.english_v1 import ROOT, save, read, digest
from evaluation.chapter_pilot import DATA, VIDEO_IDS, check
from evaluation.mixed50_v1 import MixedPipeline
from library.intent_pipeline_v2 import IntentPipeline
from library.chapter_pipeline import ChapterPipeline
from library.server import understand_query, AppError
from llm.provider import LLMError

RUN=ROOT/'evaluation_runs/chapter_pilot_v1'
CASES=[
    ('C01','介绍梯度下降','learn'),
    ('C02','我想了解动量法怎样帮助神经网络训练','learn'),
    ('C03','我想学习英语冠词 a、an 和 the','learn'),
    ('C04','我想了解面包发酵的过程','learn'),
    ('C05','介绍黑洞','learn'),
    ('C06','为什么政府不能无限印钱？','specific'),
    ('C07','我想了解睡眠和记忆的关系','learn'),
    ('C08','介绍 MCP 如何连接工具和数据','learn'),
    ('S01','大批次和小批次训练在优化上有什么区别？','specific'),
    ('S02','梯度下降的参数更新公式是什么？','specific'),
    ('S03','发酵产生的二氧化碳如何让面包膨胀？','specific'),
    ('N01','我想学习 LoRA 低秩适配','absent'),
    ('N02','Kubernetes PodDisruptionBudget 如何配置？','absent'),
    ('Q01','这个是怎么工作的？','clarify')]


def run():
    check()
    cases=[{'id':i,'question':q,'expected':e} for i,q,e in CASES]
    save(DATA/'functional_cases.json',cases)
    files=[ROOT/'library/chapter_pipeline.py',ROOT/'evaluation/chapter_check.py',DATA/'functional_cases.json',DATA/'manifest.json']
    manifest={'purpose':'paired functional development check; chapter duration is not quality or accuracy',
              'scope':VIDEO_IDS,'criteria':{'scope':'all citations within pilot videos',
              'boundaries':'keys inside source chapter and original cues',
              'negative':'no outside answers; clarify missing referents',
              'quality':'manual content-unit audit, no minimum duration target'},
              'hashes':{str(f.relative_to(ROOT)):digest(f) for f in files}}
    save(RUN/'manifest.json',manifest)
    base=MixedPipeline(understand_query)
    short=IntentPipeline(base,ROOT/'runtime/chapter_pilot_v1/comparison-short-traces')
    chapter=ChapterPipeline(base,ROOT/'runtime/chapter_pilot_v1/comparison-chapter-traces')
    summary=[]
    for case in cases:
        for name,pipeline in (('short',short),('chapter',chapter)):
            target=RUN/name/(case['id']+'.json')
            if target.exists():
                raise FileExistsError('Do not overwrite or silently mix comparison runs')
            start=time.perf_counter()
            try:
                response=pipeline.search(case['question'],set(VIDEO_IDS))
                error=None
            except (LLMError,AppError) as exc:
                response={}; error={'code':getattr(exc,'code','llm_error'),'message':str(exc)}
            row={**case,'response':response,'error':error,'latency_ms':round((time.perf_counter()-start)*1000,1)}
            save(target,row)
            clips=response.get('results',[])
            assert all(c['video_id'] in VIDEO_IDS for c in clips)
            if name=='chapter':
                for c in clips:
                    assert c['chapter']['start']<=c['key_start']<c['key_end']<=c['chapter']['end']
                    assert c['recommended_watch_start']<=c['recommended_watch_end']
            expected=case['expected']
            passed=(error and error['code']=='clarification_required') if expected=='clarify' else (
                not error and (not clips if expected=='absent' else bool(clips)))
            summary.append({'id':case['id'],'variant':name,'pass':bool(passed),'count':len(clips),
                            'watch_seconds':sum(c['recommended_watch_end']-c['recommended_watch_start'] for c in clips),
                            'latency_ms':row['latency_ms']})
            print(case['id'],name,'PASS' if passed else 'NOT PASSED',len(clips),flush=True)
    for f,h in manifest['hashes'].items(): assert digest(ROOT/f)==h,f
    check(); save(RUN/'summary.json',summary)


if __name__=='__main__': run()
