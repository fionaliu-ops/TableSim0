#!/usr/bin/env python3
"""Rearrange existing TableSim assets and render index-preserving paired scenes."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'organized-pairs-v3'
LIQUIDS = {'cup', 'bottle', 'tumbler'}
REAR = {'laptop', 'monitor', 'printer'}
DECOR = {'plant', 'plant pot', 'globe', 'clock', 'alarm clock', 'blackboard', 'chalkboard'}
GAP = .014

def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def extents(obj, yaw):
    x, y = obj['footprint']
    t = math.radians(yaw)
    return np.array([abs(math.cos(t))*x+abs(math.sin(t))*y,
                     abs(math.sin(t))*x+abs(math.cos(t))*y])

def overlaps(xy, half, other_xy, other_half, gap=0):
    return np.all(np.abs(np.asarray(xy)-other_xy) < half+other_half+gap-1e-8, axis=-1)

def plan_scene(scene, attrs):
    objects = {o['label']: o for o in scene['placed_objects']}
    if len(objects) != len(scene['placed_objects']):
        raise ValueError('Duplicate labels require explicit instance identities')
    fixed = {k for k in objects if attrs.get(k, {}).get('movable') is False}
    yaw = {}
    for k, o in objects.items():
        yaw[k] = float(o['yaw_degrees']) if k in fixed else 0.
        if k not in fixed and k in {'pen','book','notebook','marker'}:
            yaw[k] = 0. if o['footprint'][0] >= o['footprint'][1] else 90.
        if k == 'phone':
            yaw[k] = 90. if o['footprint'][0] > o['footprint'][1] else 0.
        if k == 'laptop':
            yaw[k] = 90.
    dims = {k: extents(o, yaw[k]) for k,o in objects.items()}
    parents = {}
    if 'pen' in objects and 'notebook' in objects and np.all(dims['pen']+.01 <= dims['notebook']):
        parents['pen'] = 'notebook'
    if 'book' in objects and 'notebook' in objects and np.all(dims['notebook']+.01 <= dims['book']):
        parents['notebook'] = 'book'
    roots = [k for k in objects if k not in parents]
    anchor_x = next((float(objects[k]['xy'][0]) for k in fixed if k in REAR), 0.)
    target = {}
    for k in objects:
        category = attrs.get(k, {}).get('category', 'others')
        t = [.46, -.19]
        if category == 'document': t = [-.42, -.06]
        elif category == 'electronic_device': t = [.23, .03]
        elif category == 'container': t = [.47, .25]
        if k in DECOR: t = [-.48, .27]
        if k in LIQUIDS: t = [.48, .17]
        if k == 'spoon': t = [.48, .0]
        if k in REAR: t = [anchor_x, min(.23, .35-dims[k][1]/2)]
        if k == 'keyboard': t = [anchor_x, -.12]
        if k == 'mouse': t = [anchor_x+.24, -.12]
        target[k] = np.array(t)
    fixed_plan = {}
    for k in fixed:
        o = objects[k]
        relation=o.get('relation')
        if isinstance(relation,dict) and relation.get('predicate') in {'on','inside'}:
            raise ValueError('Immovable object has a source support relation: '+k)
        fixed_plan[k] = {'xy':o['xy'], 'yaw_degrees':yaw[k], 'z_offset':o['z_offset'], 'parent':None}
    workspace_candidates = [(0.,-.275),(-.24,-.275),(.24,-.275),(0.,-.24),(-.32,-.26),(.32,-.26)]
    priorities = {'laptop':0,'monitor':0,'printer':0,'keyboard':1,'mouse':2}
    for ws in workspace_candidates:
        wsxy, wshalf = np.array(ws), np.array([.12,.065])
        if any(overlaps(wsxy,wshalf,np.array(p['xy']),dims[k]/2,GAP) for k,p in fixed_plan.items()):
            continue
        for mode in range(3):
            result = copy.deepcopy(fixed_plan)
            movable = [k for k in roots if k not in fixed]
            if mode == 0:
                movable.sort(key=lambda k:(priorities.get(k,3),-float(np.prod(dims[k])),k))
            elif mode == 1:
                movable.sort(key=lambda k:(priorities.get(k,3),-float(max(dims[k])),k))
            else:
                movable.sort(key=lambda k:(priorities.get(k,3),0 if k in LIQUIDS else 1,-float(np.prod(dims[k])),k))
            failed = False
            for k in movable:
                half = dims[k]/2
                lo = np.array([-.615,-.37])+half+.006
                hi = np.array([.615,.37])-half-.006
                if np.any(lo>hi): failed=True; break
                xs = list(np.arange(lo[0],hi[0]+1e-6,.02))+[float(np.clip(target[k][0],lo[0],hi[0]))]
                ys = list(np.arange(lo[1],hi[1]+1e-6,.02))+[float(np.clip(target[k][1],lo[1],hi[1]))]
                for j,p in result.items():
                    xy=np.array(p['xy']); h=dims[j]/2
                    xs += [xy[0],xy[0]-h[0]-GAP-half[0],xy[0]+h[0]+GAP+half[0]]
                    ys += [xy[1],xy[1]-h[1]-GAP-half[1],xy[1]+h[1]+GAP+half[1]]
                xs=np.unique(np.clip(xs,lo[0],hi[0])); ys=np.unique(np.clip(ys,lo[1],hi[1]))
                xx,yy=np.meshgrid(xs,ys); points=np.column_stack([xx.ravel(),yy.ravel()])
                valid=~overlaps(points,half,wsxy,wshalf,GAP)
                for j,p in result.items():
                    electronic_k=attrs.get(k,{}).get('electronic',False)
                    electronic_j=attrs.get(j,{}).get('electronic',False)
                    gap=.055 if (k in LIQUIDS and electronic_j) or (j in LIQUIDS and electronic_k) else GAP
                    valid &= ~overlaps(points,half,np.array(p['xy']),dims[j]/2,gap)
                points=points[valid]
                if not len(points): failed=True; break
                delta=points-target[k]
                score=delta[:,0]**2*2+delta[:,1]**2
                # Prefer functional groups, aligned rows, and rear placement for tall items.
                cat=attrs.get(k,{}).get('category','others')
                if cat=='document': score += np.maximum(points[:,0]+.18,0)*3
                if k in LIQUIDS: score += np.maximum(.25-points[:,0],0)*3
                if k in DECOR or objects[k]['height']>.14: score += np.maximum(.12-points[:,1],0)*.5
                result[k]={'xy':points[int(np.argmin(score))].tolist(),'yaw_degrees':yaw[k],'z_offset':0.,'parent':None}
            if failed: continue
            pending=dict(parents)
            while pending:
                progressed=False
                for k,p in list(pending.items()):
                    if p not in result: continue
                    result[k]={'xy':list(result[p]['xy']),'yaw_degrees':yaw[k],
                               'z_offset':result[p]['z_offset']+objects[p]['height']+.004,'parent':p}
                    del pending[k]; progressed=True
                if not progressed: raise ValueError('Support cycle')
            return {'objects':result,'workspace':{'xy':list(ws),'size':[.24,.13]},
                    'fixed_labels':sorted(fixed),'minimum_root_clearance':GAP,'strategy':mode}
    raise ValueError('No collision-free organized placement retaining all objects and a free workspace')

def identity(scene):
    return Counter(tuple(str(o[k]) for k in ('label','asset','source','kind','model_id')) for o in scene['placed_objects'])

def render_one(original, plan, output, specs, variants, attrs, scene):
    import sapien
    import scence_generator as g
    idx=original['scene_index']; image=original['image']
    try:
        scene.set_timestep(1/250)
        g.add_table_scene(scene,np.random.default_rng(20261009+idx))
        loaded={}
        for record in original['placed_objects']:
            label=record['label']; spec=specs[record['asset']]
            if spec['source']!=record['source'] or spec['kind']!=record['kind']:
                raise ValueError('Asset map does not match source manifest')
            variant=variants[(record['asset'],str(record['model_id']))]
            obj=g.load_object(scene,g.ObjectChoice(label,spec,variant))
            if not np.allclose(obj.footprint,record['footprint'],atol=.0001):
                raise ValueError(f'{label}: asset footprint changed since original generation')
            loaded[label]=obj
        pending=dict(plan['objects'])
        done=set()
        while pending:
            for label,p in list(pending.items()):
                parent=p['parent']
                if parent and parent not in done: continue
                obj=loaded[label]
                z=p['z_offset']
                if parent:
                    support=loaded[parent]
                    z=support.z_offset+support.height+.004
                    obj.relation={'subject':label,'predicate':'on','object':parent}
                g.place_loaded_object(obj,np.array(p['xy']),z,math.radians(p['yaw_degrees']))
                del pending[label]; done.add(label)
        for _ in range(4): scene.step()
        scene.update_render()
        validation=g.validate_physical_layout(loaded)
        # Original poses are serialized to 5 decimal places and yaw to 2.
        # Preserve immovable poses exactly; allow only their sub-0.05 mm
        # rounding difference against the generator's 4 mm inset boundary.
        tolerated=[]
        for label in plan['fixed_labels']:
            violation=f'{label}:outside_table_bounds'
            if violation not in validation['violations']: continue
            obj=loaded[label]
            half=g.rotated_half_extents(obj)+g.PLACEMENT_CLEARANCE
            low=np.array([g.TABLE_XLIM[0],g.TABLE_YLIM[0]])+half
            high=np.array([g.TABLE_XLIM[1],g.TABLE_YLIM[1]])-half
            excess=float(max(np.max(low-obj.xy),np.max(obj.xy-high),0.))
            if original['physical_validation']['valid'] and excess<=.00005:
                validation['violations'].remove(violation)
                tolerated.append({'label':label,'inner_boundary_excess_metres':excess,
                                  'reason':'source_manifest_pose_rounding'})
        validation['valid']=not validation['violations']
        if tolerated:
            validation['source_pose_rounding_tolerance_metres']=.00005
            validation['source_pose_rounding_adjustments']=tolerated
        if not validation['valid']: raise ValueError(str(validation['violations']))
        camera_position=[0.,-1.12,1.42]; camera_target=[0.,0.,.82]
        w,h=original['resolution']
        camera=scene.add_camera('organized_desk_camera',w,h,math.radians(54),.01,10.)
        camera.entity.set_pose(g.camera_pose(np.array(camera_position),np.array(camera_target)))
        scene.update_render()
        temp=output/(Path(image).stem+'.tmp.png')
        g.render_camera(camera,temp)
        temp.replace(output/image)
        new=copy.deepcopy(original)
        new['placed_objects']=[g.object_record(loaded[o['label']]) for o in original['placed_objects']]
        new['on_relationships']=[o.relation for o in loaded.values() if o.relation]
        new['relationships_sampled']=copy.deepcopy(new['on_relationships'])
        new['occlusion']=None
        new['physical_validation']=validation
        new['layout_policy']={'name':'organized_functional_layout','version':VERSION,
                              'camera_side':'negative_y_front','free_workspace':plan['workspace'],
                              'fixed_labels':plan['fixed_labels'],'valid':True,'violations':[],
                              'principles':['retain_all_original_assets','immovable_objects_unchanged',
                              'aligned_devices','documents_grouped_left','liquids_separated_from_electronics',
                              'writing_tools_on_notebook','clear_front_workspace','no_unintended_overlap']}
        new['camera']={'position':camera_position,'target':camera_target,'fovy_degrees':54,'near':.01,'far':10.}
        new['source_provenance']={k:original.get(k) for k in ('source_batch_index','source_seed','source_scene_index','source_manifest','attempt_index')}
        new['source_manifest']='../scenes_new_0929_1000/manifest.json'
        new['source_scene_index']=idx
        new['source_image']='../scenes_new_0929_1000/'+image
        new['organization']={'version':VERSION,'retained_object_count':len(loaded),
                             'image_sha256':digest(output/image),'same_camera_as_source':False,
                             'camera_note':'Source manifest omitted camera extrinsics; organized pairs use a fixed recorded camera.'}
        if identity(new)!=identity(original): raise ValueError('Object identity mismatch')
        for label in plan['fixed_labels']:
            a=next(o for o in original['placed_objects'] if o['label']==label)
            b=next(o for o in new['placed_objects'] if o['label']==label)
            for key in ('xy','yaw_degrees','z_offset'):
                if not np.allclose(a[key],b[key],atol=1e-5): raise ValueError('Immovable object moved')
        dump(output/'records'/f'scene_{idx:03d}.json',new)
        return new
    finally:
        scene.clear()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=ROOT/'image_collectors/scenes_new_0929_1000')
    p.add_argument('--output',type=Path,default=ROOT/'image_collectors/scenes_new_0929_1000_organized')
    p.add_argument('--mode',choices=['plan','render','run','finalize'],required=True)
    p.add_argument('--start',type=int,default=0); p.add_argument('--end',type=int,default=1000)
    p.add_argument('--batch-size',type=int,default=20)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    source=json.loads((a.source/'manifest.json').read_text())
    scenes=source['scenes']; attrs=json.loads((ROOT/'image_collectors/object_attributes_with_category.json').read_text())['objects']
    assert len({s['image'] for s in scenes})==len(scenes)
    assert [s['scene_index'] for s in scenes]==list(range(len(scenes)))
    if a.mode=='plan':
        plans={}; failures=[]
        for s in scenes:
            try: plans[str(s['scene_index'])]=plan_scene(s,attrs)
            except Exception as e: failures.append({'scene_index':s['scene_index'],'error':str(e)})
        dump(a.output/'layout_plans.json',{'version':VERSION,'source_sha256':digest(a.source/'manifest.json'),'plans':plans,'failures':failures})
        print(json.dumps({'planned':len(plans),'failures':failures},ensure_ascii=False),flush=True)
        return int(bool(failures))
    data=json.loads((a.output/'layout_plans.json').read_text())
    assert data['version']==VERSION and data['source_sha256']==digest(a.source/'manifest.json')
    if a.mode=='render':
        sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'script'))
        import scence_generator as g
        import sapien
        scene=sapien.Scene([sapien.physx.PhysxCpuSystem(),sapien.render.RenderSystem(a.device)])
        specs={s['output']:s for s in g.load_specs()}
        needed={o['asset'] for s in scenes[a.start:a.end] for o in s['placed_objects']}
        variants={(name,v.display_id):v for name in needed for v in g.discover_variants(specs[name])}
        failures=[]
        for s in scenes[a.start:a.end]:
            idx=s['scene_index']; record_path=a.output/'records'/f'scene_{idx:03d}.json'
            if record_path.exists() and (a.output/s['image']).exists():
                existing=json.loads(record_path.read_text())
                if existing.get('organization',{}).get('version')==VERSION and existing['organization']['image_sha256']==digest(a.output/s['image']):
                    print(f'SKIP {idx:03d}',flush=True); continue
            try:
                render_one(s,data['plans'][str(idx)],a.output,specs,variants,attrs,scene)
                print(f'OK {idx:03d}',flush=True)
            except Exception as e:
                failures.append({'scene_index':idx,'error':str(e),'traceback':traceback.format_exc()})
                print(f'FAIL {idx:03d}: {e}',flush=True)
        dump(a.output/'logs'/f'failures_{a.start:03d}_{a.end:03d}.json',failures)
        return int(bool(failures))
    if a.mode=='run':
        for start in range(a.start,a.end,a.batch_size):
            end=min(start+a.batch_size,a.end)
            log=a.output/'logs'/f'render_{start:03d}_{end:03d}.log';log.parent.mkdir(exist_ok=True)
            cmd=[sys.executable,str(Path(__file__).resolve()),'--mode','render','--device',a.device,'--source',str(a.source),'--output',str(a.output),'--start',str(start),'--end',str(end)]
            with log.open('a') as f: result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
            count=len(list((a.output/'records').glob('scene_*.json')))
            dump(a.output/f'progress_{a.device.replace(":","_")}.json',{'completed':count,'total':len(scenes),'last_batch':[start,end],'last_exit_code':result.returncode,'updated_at_unix':time.time()})
            print(f'BATCH {start}:{end} exit={result.returncode}; completed={count}/{len(scenes)}',flush=True)
        return 0
    if a.mode=='finalize':
        from PIL import Image
        records=[]; errors=[]
        for s in scenes:
            idx=s['scene_index']; record=a.output/'records'/f'scene_{idx:03d}.json'
            try:
                r=json.loads(record.read_text()); assert r['scene_index']==idx and r['image']==s['image']
                assert identity(r)==identity(s) and r['physical_validation']['valid']
                assert r['organization']['version']==VERSION
                by_label={o['label']:o for o in r['placed_objects']}
                original_by_label={o['label']:o for o in s['placed_objects']}
                plan=data['plans'][str(idx)]
                roots=[]
                for label,o in by_label.items():
                    expected=plan['objects'][label]
                    assert np.all(np.isfinite([*o['xy'],o['z_offset'],o['yaw_degrees']]))
                    assert np.allclose(o['xy'],expected['xy'],atol=1e-5)
                    assert abs(o['yaw_degrees']-expected['yaw_degrees'])<1e-5
                    if label in plan['fixed_labels']:
                        for key in ('xy','yaw_degrees','z_offset'):
                            assert np.allclose(o[key],original_by_label[label][key],atol=1e-5)
                    parent=expected['parent']
                    if parent:
                        support=by_label[parent]
                        assert o['relation']['predicate']=='on' and o['relation']['object']==parent
                        assert np.allclose(o['xy'],support['xy'],atol=1e-5)
                        assert np.all(extents(o,o['yaw_degrees'])<=extents(support,support['yaw_degrees'])+1e-5)
                        assert abs(o['world_bounds'][0][2]-support['world_bounds'][1][2])<.01
                    else:
                        roots.append(o)
                        assert not overlaps(np.array(o['xy']),extents(o,o['yaw_degrees'])/2,
                                            np.array(plan['workspace']['xy']),np.array(plan['workspace']['size'])/2)
                for i,o in enumerate(roots):
                    for other in roots[i+1:]:
                        assert not overlaps(np.array(o['xy']),extents(o,o['yaw_degrees'])/2,
                                            np.array(other['xy']),extents(other,other['yaw_degrees'])/2)
                assert r['organization']['image_sha256']==digest(a.output/r['image'])
                with Image.open(a.output/r['image']) as im: assert list(im.size)==r['resolution']; im.verify()
                records.append(r)
            except Exception as e: errors.append({'scene_index':idx,'error':repr(e),'traceback':traceback.format_exc()})
        actual_images=sorted(x.name for x in a.output.glob('scene_*.png'))
        expected_images=sorted(s['image'] for s in scenes)
        if actual_images!=expected_images:
            errors.append({'error':'Image filename set does not match source exactly'})
        report={'expected':len(scenes),'validated':len(records),'errors':errors,'source_sha256':digest(a.source/'manifest.json'),
                'checks':['exact_image_filename_set','one_to_one_indices','same_object_models_and_counts',
                          'immutable_source_manifest','unchanged_immovable_objects','poses_match_layout_plans',
                          'physical_layout_valid','supported_stack_containment','clear_front_workspace',
                          'nonintersecting_root_footprints','image_sha256_and_png_integrity']}
        dump(a.output/'validation.json',report)
        if errors: print(json.dumps(report));return 1
        result=copy.deepcopy(source)
        result.update(generator='script/organize_paired_scenes.py',scene_generator='script/organize_paired_scenes.py',scenes=records,failures=[],source_manifest='../scenes_new_0929_1000/manifest.json',source_manifests=['../scenes_new_0929_1000/manifest.json'],incomplete_batches=[])
        result['generation_config']={'scenes':len(records),'width':1024,'height':768,'layout_version':VERSION,'preserve_object_identity':True,'preserve_scene_indices':True}
        result['runtime']={'python_executable':sys.executable,'python_version':sys.version,'generated_at_unix':time.time()}
        result['pairing']={'source_directory':'../scenes_new_0929_1000','key':'scene_index',
                           'same_image_filenames':True,'same_object_models_and_counts':True,
                           'source_manifest_sha256':report['source_sha256'],
                           'same_camera_as_source':False,'render_seed_base':20261009}
        result['summary']={'requested_scenes':len(scenes),'candidate_attempts':len(records),'max_attempts':len(records),'incomplete_batches':0,'rendered_scenes':len(records),'failures':0,'images':[s['image'] for s in records],'paired_source_sha256':report['source_sha256']}
        dump(a.output/'manifest.json',result)
        print(json.dumps(report),flush=True)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
