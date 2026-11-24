"""Run two real target-conditioned Reasoning Graph searches for scene 003."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from copy import deepcopy
from PIL import Image

from high_level_planner.perception.sam3_client import Sam3PerceptionClient
from high_level_planner.scene_graph.constructor import SceneGraphConstructor
from high_level_planner.scene_graph.graph import SceneGraph
from high_level_planner.perception.base import PerceptionResult
from high_level_planner.reasoning_graph.language_models.openai_compatible import OpenAICompatibleLanguageModel
from high_level_planner.reasoning_graph.state import ImageSceneState
from high_level_planner.reasoning_graph.target_conditioned import generate_between_scenes
from high_level_planner.planner.ucs import uniform_cost_search
from high_level_planner.ucs_path_output import write_ucs_path_output
from script.goal_config import dynamic_reasoning_config, load_api_settings

INSTRUCTION = ("Organize the desktop compactly near the monitor, straighten and stack the book "
               "and folder neatly, align the keyboard, phone and mouse, and preserve all objects.")

class ImageConditionedModel(OpenAICompatibleLanguageModel):
    """Give both photographs to every real model expansion in the graph search."""
    def __init__(self, config, first, last):
        super().__init__(config)
        self.visual_evidence = [ImageSceneState(str(Path(first).resolve())),
                                ImageSceneState(str(Path(last).resolve()))]
    def generate(self, prompt, *, num_responses=1, images=None):
        # The underlying graph algorithm may pass an image-primary current
        # state. Deduplicate only exact paths; target evidence is always sent.
        evidence=[]
        seen=set()
        for item in [*(images or []),*self.visual_evidence]:
            if item.source not in seen:
                seen.add(item.source);evidence.append(item)
        return super().generate(prompt,num_responses=num_responses,images=evidence)


def write(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n")


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_image(source, destination):
    image=Image.open(source).convert('RGB')
    image.thumbnail((960,720))
    image.save(destination,format='JPEG',quality=88,optimize=True)
    return destination


def sam3_pair(initial_image,target_image,output,sam3_endpoint,initial_saved=None,target_saved=None):
    backend=Sam3PerceptionClient(endpoint=sam3_endpoint)
    constructor=SceneGraphConstructor(perception_backend=backend)
    initial=(SceneGraph.from_json(initial_saved) if initial_saved else
             constructor.construct(initial_image,instruction=INSTRUCTION))
    raw_graph=(SceneGraph.from_json(target_saved) if target_saved else
               constructor.construct(target_image,instruction=INSTRUCTION))
    raw_graph.to_json(output/'terminal_sam3_raw.json')
    base_ids={o.id for o in initial.objects}; raw_ids={o.id for o in raw_graph.objects}
    supplement=None
    target=raw_graph.copy()
    if base_ids!=raw_ids and base_ids-raw_ids=={'book_sam3_2'}:
        # SAM3 itself sees a second nested book/stack detection at lower
        # confidence. Preserve its real confidence; no manual geometry.
        book_detector=Sam3PerceptionClient(endpoint=sam3_endpoint,
            prompt_catalog=['book'],prompt_catalog_path=None,
            confidence_threshold=.25,mask_cluster_iou_threshold=.95)
        books=[d for d in book_detector.detect(target_image).detections if d.label=='book']
        books.sort(key=lambda d:d.score,reverse=True)
        if len(books)>=2:
            supplement_graph=constructor._from_perception_result(
                PerceptionResult('sam3',[books[1]],[Image.open(target_image).width,Image.open(target_image).height]),
                image_path=str(target_image),instruction=INSTRUCTION)
            second=supplement_graph.objects[0]
            second.id='book_sam3_2'
            target.add_object(second)
            supplement={'method':'SAM3 book-only second detection','score':books[1].score,
                        'bbox':books[1].bbox,'identity_confident':False,
                        'reason':'raw target SAM3 missed book_sam3_2 at default threshold'}
    target.metadata['supplemental_sam3_detection']=supplement
    initial.to_json(output/'initial_sam3_scene_graph.json')
    target.to_json(output/'terminal_sam3_scene_graph.json')
    write(output/'sam3_identity_audit.json',{'initial_ids':sorted(base_ids),'raw_target_ids':sorted(raw_ids),
        'final_target_ids':sorted(o.id for o in target.objects),'supplement':supplement,
        'same_unique_ids':base_ids=={o.id for o in target.objects},
        'saved_initial_sam3_graph':str(initial_saved) if initial_saved else None,
        'saved_target_sam3_graph':str(target_saved) if target_saved else None,
        'warning':'SAM3 labels physical folder as book and terminal second detection has low confidence'})
    return initial,target


def one_mode(mode,initial,target,first,last,output,settings):
    dest=output/mode;dest.mkdir(parents=True,exist_ok=True)
    config=dynamic_reasoning_config(settings,dest/'model_audit')
    config=replace(config,max_retries=1)
    if mode=='images':
        compressed_first=model_image(first,dest/'model_initial.jpg')
        compressed_last=model_image(last,dest/'model_terminal.jpg')
        model=ImageConditionedModel(config,compressed_first,compressed_last)
    else:
        model=OpenAICompatibleLanguageModel(config)
    report={'mode':mode,'status':'running','reasoning_graph_generated':False,
            'model_inputs':'SAM3 graph search plus two photographs on every expansion' if mode=='images'
                           else 'SAM3 initial and terminal graphs only',
            'initial_image_sha256':sha(first),'target_image_sha256':sha(last),
            'model_generated_goal_image':False,'kimi_target_graph_extraction':False}
    write(dest/'run_summary.json',report)
    begun=time.monotonic()
    try:
        graph=generate_between_scenes(initial,target,INSTRUCTION,model=model,
            max_states=10,max_depth=6,beam_width=1,root_beam_width=1,num_branches=3,
            position_tolerance=.035)
        graph.to_json(dest/'reasoning_graph.json')
        search=uniform_cost_search(graph)
        write_ucs_path_output(graph,search,dest,enabled=True)
        path=[{'from_state':edge.from_state,'to_state':edge.to_state,
               'action':edge.to_dict()} for edge in search.path]
        write(dest/'selected_path.json',path)
        report.update(status='completed',reasoning_graph_generated=True,
            node_count=len(graph.nodes),edge_count=len(graph.edges),
            goal_ids=sorted(graph.goal_ids),goal_state=search.goal_state,
            selected_path_length=len(search.path),model_execution=graph.metadata.get('model_execution'),
            graph_sha256=sha(dest/'reasoning_graph.json'))
    except Exception as exc:
        report.update(status='generation_failed',error=f'{type(exc).__name__}: {exc}')
    finally:
        report['elapsed_seconds']=round(time.monotonic()-begun,3)
        report['http_attempts']=len(list((dest/'model_audit').glob('request_*.json')))
        write(dest/'run_summary.json',report)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--initial-image',type=Path,required=True)
    parser.add_argument('--target-image',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--api-config',type=Path,required=True)
    parser.add_argument('--initial-sam3-graph',type=Path)
    parser.add_argument('--target-sam3-graph',type=Path)
    parser.add_argument('--sam3-endpoint',default='http://127.0.0.1:8003')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    initial,target=sam3_pair(args.initial_image,args.target_image,args.output_dir,args.sam3_endpoint,
                            args.initial_sam3_graph,args.target_sam3_graph)
    settings=load_api_settings(args.api_config)
    results=[]
    for mode in ('images','scene_graphs'):
        outcome=one_mode(mode,initial,target,args.initial_image,args.target_image,args.output_dir,settings)
        results.append(outcome);write(args.output_dir/'comparison.json',{'results':results})
        print(json.dumps({k:outcome.get(k) for k in ('mode','status','node_count','edge_count','selected_path_length','error')},ensure_ascii=False),flush=True)
    return 0
if __name__=='__main__':raise SystemExit(main())
