import unittest
from unittest.mock import Mock, patch
import threading
import numpy as np

from evaluation.chapter_pilot import units, materialize
from library.chapter_pipeline import ChapterPipeline, make_clip
from llm.provider import LLMError, GenerationResult


class ChapterTests(unittest.TestCase):
    def test_content_partition_and_key_boundaries(self):
        cues=[{'start':i*10,'end':i*10+9,'text':str(i)} for i in range(6)]
        self.assertEqual(units(cues,[0,3]),[(0,2),(3,5)])
        for starts in ([1],[0,0],[0,6],[0,4,2]):
            with self.assertRaises(LLMError): units(cues,starts)
        label=lambda a:{'start_cue':a,'title':'章节','summary':'描述','english_summary':'description',
                        'kind':'teaching','points':[{'title':'要点','start_cue':a}]}
        rows={'chapters':[label(0),label(3)]}
        chapters=materialize('v',cues,[0,3],rows)
        self.assertEqual(chapters[0]['end'],29)
        rows['chapters'][0]['points'][0]['start_cue']=3
        with self.assertRaises(LLMError): materialize('v',cues,[0,3],rows)

    def test_full_chapter_and_precise_anchor_are_separate(self):
        cues=[{'start':i*10,'end':i*10+9,'text':'evidence'} for i in range(6)]
        c={'start_cue':0,'end_cue':5,'start':0,'end':59,'chapter_id':'c'}
        row={'key_start_cue':2,'key_end_cue':3,'reason':'有依据'}
        video={'video_id':'v','title':'t','channel':'a','url':'https://www.youtube.com/watch?v=v'}
        broad=make_clip(c,row,video,cues,True)
        precise=make_clip(c,row,video,cues,False)
        self.assertEqual((broad['recommended_watch_start'],broad['recommended_watch_end']),(0,59))
        self.assertEqual((precise['recommended_watch_start'],precise['recommended_watch_end']),(20,39))
        self.assertEqual(broad['key_start'],20)
        with self.assertRaises(LLMError): make_clip(c,{**row,'key_end_cue':6},video,cues,True)

    def test_dual_retrieval_scope_and_original_constraints(self):
        engine=ChapterPipeline.__new__(ChapterPipeline)
        c={'video_id':'v','chapter_id':'c','kind':'teaching','english_summary':'topic','start':0,'end':29,
           'start_cue':0,'end_cue':2,'title':'title'}
        other={**c,'video_id':'outside','chapter_id':'outside'}
        engine.chapters=[c,other];engine.by_id={'c':c,'outside':other}
        engine.chapter_vectors=np.array([[1.,0.],[0.,1.]])
        engine.parents={'w':['c'],'x':['outside']};engine.logger=Mock()
        p=Mock();engine.pipeline=p;p.lock=threading.Lock()
        p.chunks=[{'chunk_id':'w','video_id':'v'},{'chunk_id':'x','video_id':'outside'}]
        p.vectors=engine.chapter_vectors
        p.captions={'v':[{'start':i*10,'end':i*10+9,'text':'evidence'} for i in range(3)]}
        p.metadata={'v':{'video_id':'v','title':'t','channel':'a','url':'https://www.youtube.com/watch?v=v'}}
        plan={'task_type':'compare','english_query':'compare','core_query':'topic','directions':[],
              'constraints':[{'text':'不要推导','quote':'不要推导'}]}
        answer={'items':[{'chapter_id':'c','reason':'evidence','key_start_cue':1,'key_end_cue':2,'covered_directions':[]}]}
        result=GenerationResult(answer,'deepseek-flash',1,1,0,1)
        with patch('library.chapter_pipeline.plan_request',return_value=(plan,{})), \
             patch('library.chapter_pipeline.encode',return_value=np.array([[1.,0.]])), \
             patch('library.chapter_pipeline.score_pairs',return_value=[1.]) as rank, \
             patch('library.chapter_pipeline.generate_json',return_value=result) as gate:
            r=engine.search('比较A和B，不要推导',{'v'})
            self.assertEqual(r['results'][0]['recommended_watch_start'],10)
            payload=gate.call_args.args[1]
            self.assertIn('比较A和B，不要推导',payload)
            self.assertNotIn('outside',payload)
            self.assertEqual(gate.call_args.kwargs['schema']['properties']['items']['maxItems'],1)
            r=engine.search('比较A和B，不要推导',set())
            self.assertFalse(r['answerable'])
            self.assertEqual(gate.call_count,1)
            # A full explanation ranked fourth must reach the same comparison as an early recap.
            engine.chapters=[{**c,'chapter_id':f'c{i}'} for i in range(6)]
            engine.by_id={c['chapter_id']:c for c in engine.chapters}
            engine.chapter_vectors=np.array([[1.,0.]]*6)
            engine.parents['w']=list(engine.by_id)
            rank.return_value=[6.,5.,4.,3.,2.,1.]
            answer['items'][0]['chapter_id']='c3'
            r=engine.search('比较A和B，不要推导',{'v'})
            self.assertEqual(r['results'][0]['segment_id'],'c3')
            self.assertIn('c5',gate.call_args.args[1])


if __name__=='__main__': unittest.main()
