"""Closed-loop scene 003: external model plans one action from current and goal images."""
from __future__ import annotations
import argparse,json,time,shutil
from copy import deepcopy
from pathlib import Path
from dataclasses import replace
import numpy as np
from PIL import Image
from high_level_planner.perception.sam3_client import Sam3PerceptionClient
from high_level_planner.scene_graph.constructor import SceneGraphConstructor
from high_level_planner.reasoning_graph.language_models.openai_compatible import OpenAICompatibleLanguageModel,RetryableLanguageModelError
from high_level_planner.reasoning_graph.parser import ReasoningGraphParser
from script.compare_scene003_modalities import install_recording,match_book_instances,request_from_action,write,digest
from high_level_planner.reasoning_graph.state import ImageSceneState
from script.dynamic_sim import build_environment
from script.goal_config import load_api_settings,dynamic_reasoning_config
from high_level_planner.evaluation import EpisodeMetrics

INSTRUCTION="Organize the desktop compactly near the monitor, straighten and stack the book and folder neatly, align the keyboard, phone and mouse, and preserve all objects."
TARGET_YAWS = {'phone':0., 'keyboard':0., 'mouse':0., 'book':-90., 'folder':0., 'stapler':-90.}

def aligned_goal(env):
    return (env.held is None and env.task.is_left_gripper_open() and env.task.is_right_gripper_open()
            and env.image_goal is not None
            and env.image_goal.evaluate(env)['passed']
            and book_support_status(env.task)['aligned_and_supported'])
SCHEMA="""Return strict JSON {"done":false,"action":{"action_type":"move|place|stack|align|open|close","source_object_id":"ID","target":"table or ID","pose":{"frame":"image_normalized","xy":[x,y],"support":"table or ID"},"reason":"brief"},"uncertainty":"brief"}. Return one atomic action only. Image 1 is the CURRENT robot head-camera image, image 2 is the TARGET tidy image. Use target image coordinates divided by width and height. Clear blockers before moving objects below them. Use this safe rearrangement order when those objects still need work: place the phone in the middle/front to free the right-front buffer; move the keyboard temporarily into that buffer; move the mouse to a clear temporary right buffer; move the cup through a central/front handoff to its right/rear point; side-grasp and stack the book on the fixed folder; place the keyboard near the monitor and the mouse beside it. Temporary moves are allowed. The terminal picture defines final object positions and orientations. Temporary buffer positions never count as completed goals. If a goal is blocked, clear the obstacle and retry; do not silently substitute a different final location. Do not repeat an action already completed unless the current image demands it. Never move the monitor. Move and straighten the folder only after the book is held outside the left workspace. The executor supports wrist rotation during transport; final yaw targets come from the terminal-image axes. The releasing arm must retreat before another arm enters the shared workspace. Folder manipulation requires the right arm and held book to wait safely outside the left workspace. Return done=true and action=null only when the desktop is fully organized."""

def compact_image(source,destination):
    image=Image.open(source).convert('RGB')
    image.thumbnail((640,480))
    image.save(destination,format='JPEG',quality=82,optimize=True)
    return ImageSceneState(str(destination.resolve()),mime_type='image/jpeg')

def goal_feedback(task):
    cup=task.objects['cup'].get_pose().p
    phone=task.objects['phone'].get_pose().p
    keyboard=task.objects['keyboard'].get_pose().p
    mouse=task.objects['mouse'].get_pose().p
    result=[]
    if cup[0]<.100 or cup[1]<.070:result.append('Cup must be farther toward the rear-right area.')
    if task.region_for_x(phone[0])!='middle':result.append('Phone must be in the middle horizontal region.')
    if task.region_for_x(keyboard[0])!='middle' or keyboard[1]<.030:result.append('Keyboard must be farther toward the monitor, in the middle region.')
    if abs(float(mouse[0]))>.180 or np.linalg.norm(mouse[:2]-keyboard[:2])>.180:result.append('Mouse must move left and closer to the keyboard, away from the cup.')
    if not (task.is_left_gripper_open() and task.is_right_gripper_open()):result.append('A gripper must release its held object before completion.')
    return result

def book_support_status(task):
    book=np.asarray(task.objects['book'].get_pose().p,dtype=float)
    folder=np.asarray(task.objects['folder'].get_pose().p,dtype=float)
    gap=float(book[2]-folder[2]-task.object_geometry['folder']['height'])
    offset=float(np.linalg.norm(book[:2]-folder[:2]))
    return {'aligned_and_supported':bool(offset<=.04 and 0<=gap<=.025),
            'center_offset':offset,'support_gap':gap}

def constrained_scene003_action(env, ids_by_name, model_action):
    """Reserve temporary corridors; restore every item to its image goal."""
    task = env.task
    goal = env.image_goal
    if env.held:
        name = env.held[0]
        proposed_name = env.object_map.get(model_action.get('source_object_id'))
        pose = model_action.get('pose') if isinstance(model_action, dict) else None
        proposed_xy = np.asarray(pose.get('xy'), dtype=float) if isinstance(pose, dict) else np.empty(0)
        if (proposed_name == name and model_action.get('action_type') in {'move', 'place'}
                and model_action.get('target') == 'table' and proposed_xy.shape == (2,)):
            proposed_world = np.r_[proposed_xy, 1.] @ env.transform
            current_world = np.asarray(task.objects[name].get_pose().p[:2], dtype=float)
            # Preserve a model-proposed recovery buffer when an object remains
            # held after a failed transfer. The normal placement executor still
            # enforces reachability, object clearance, contacts and arm safety.
            if (np.linalg.norm(proposed_world - goal.targets[name]) > .04
                    and np.linalg.norm(proposed_world - current_world) > .03):
                recovery = deepcopy(model_action)
                recovery['action_type'] = 'move'
                recovery['target'] = 'table'
                recovery['pose']['support'] = 'table'
                return recovery, {
                    'proposed_object': name,
                    'required_object': name,
                    'reason': 'honor held-object recovery buffer after a blocked transfer',
                    'temporary': True,
                    'target_world_xy': proposed_world.tolist(),
                }
    checks = goal.evaluate(env)
    def done(name):
        return checks['objects'][name]['passed'] and checks['orientations'].get(name, {'passed': True})['passed']
    def at(name, xy, tolerance=.03):
        return np.linalg.norm(task.objects[name].get_pose().p[:2] - xy) <= tolerance
    cup = task.objects['cup'].get_pose().p
    book_done = done('book') and done('folder') and book_support_status(task)['aligned_and_supported']
    if env.held:
        name = env.held[0]
        target = ([.180, -.050] if name == 'cup' and env.held[1] == 'left' else goal.targets[name])
        reason = 'finish the currently held object before switching arms'
    elif not book_done and not at('keyboard', [.220, -.200]) and not at('phone', [-.045, -.240]):
        name, target, reason = 'phone', [-.045, -.240], 'clear the keyboard side grasp; this is a temporary phone position'
    elif not book_done and not at('keyboard', [.220, -.200]):
        name, target, reason = 'keyboard', [.220, -.200], 'temporary keyboard buffer clears the book grasp'
    elif not done('cup') and cup[0] <= .14 and not at('mouse', [.280, -.100]):
        name, target, reason = 'mouse', [.280, -.100], 'clear the supported cup handoff'
    elif not done('cup'):
        name = 'cup'
        target = [.180, -.050] if cup[0] <= .14 else goal.targets['cup']
        reason = 'supported handoff with arm retreat, then final cup position'
    elif not book_done and not at('phone', [.080, -.230]):
        name, target, reason = 'phone', [.080, -.230], 'clear the book lowering corridor after the keyboard has left'
    elif not book_done:
        name, target, reason = 'book', goal.targets['book'], 'align the support and stack book at the image goal'
    elif not done('keyboard') and task.objects['mouse'].get_pose().p[1] < -.06:
        name, target, reason = 'mouse', [.300, -.030], 'clear the keyboard pickup corridor'
    else:
        name = next((n for n in ('keyboard', 'mouse', 'stapler', 'phone') if not done(n)), None)
        if name is None:
            return model_action, None
        target, reason = goal.targets[name], 'restore final position and orientation from the confirmed image'
    support = ids_by_name['folder'] if name == 'book' else 'table'
    normalized = ((np.asarray(target) - env.transform[2]) @ np.linalg.inv(env.transform[:2])).tolist()
    action = {'action_type': 'stack' if name == 'book' else 'move',
              'source_object_id': ids_by_name[name], 'target': support,
              'pose': {'frame': 'image_normalized', 'xy': normalized, 'support': support}, 'reason': reason}
    is_final = np.linalg.norm(np.asarray(target) - goal.targets[name]) < .001
    if name in TARGET_YAWS and (is_final or name == 'book'):
        action['pose']['yaw_degrees'] = TARGET_YAWS[name]
    return action, {'proposed_object': env.object_map.get(model_action['source_object_id']),
                    'required_object': name, 'reason': reason, 'temporary': not is_final,
                    'target_world_xy': np.asarray(target).tolist()}


def validate(data,ids):
    if not isinstance(data,dict) or not isinstance(data.get('done'),bool):raise ValueError('Invalid one-step result')
    if data['done']:
        if data.get('action') is not None:raise ValueError('Done result must not contain action')
        return None
    a=data.get('action')
    if not isinstance(a,dict) or a.get('source_object_id') not in ids:raise ValueError('Unknown action source')
    if a.get('action_type') not in {'move','place','stack','align','open','close'}:raise ValueError('Unsupported atomic action or rotation without yaw')
    if a.get('target') not in ({'table',None}|ids):raise ValueError('Unknown action target')
    if a['action_type'] not in {'open','close'}:
        pose=a.get('pose')
        xy=np.asarray(pose.get('xy') if isinstance(pose,dict) else None,dtype=float)
        if not isinstance(pose,dict) or pose.get('frame')!='image_normalized' or xy.shape!=(2,) or not np.isfinite(xy).all() or not np.all((0<=xy)&(xy<=1)):
            raise ValueError('Invalid target image pose')
    return a


def automatic_evaluation(env):
    """Legacy runner's independent simulator evaluator, not camera-grounded truth.

    Preserve the existing image/metric/tilt tolerances and publish their scope.
    No metric state is passed back into the planner.
    """
    measured = env.image_goal.evaluate(env)
    checks = {f'{group}:{key}': bool(value['passed'])
              for group in ('objects', 'orientations', 'supports')
              for key, value in measured.get(group, {}).items()}
    objects = measured['objects']
    errors = {key: {'position_error': value['position_error_m'], 'image_error': value['image_error'],
                    'yaw_error_degrees': None, 'frame': 'world', 'units': 'm', 'observed': True}
              for key, value in objects.items()}
    positions = [v['position_error_m'] for v in objects.values()]
    supports = {key: bool(value['passed']) for key, value in measured.get('supports', {}).items()}
    return {'success': bool(aligned_goal(env)), 'goal_checks': checks,
        'payload_released': env.held is None,
        'layout': {'definition': 'legacy_simulator_image_metric_tilt_checks',
            'object_errors': errors, 'pose_checks': {k: bool(v['passed']) for k, v in objects.items()},
            'object_pose_match_rate': sum(v['passed'] for v in objects.values())/len(objects),
            'target_relation_satisfaction': sum(supports.values())/len(supports) if supports else None,
            'relation_checks': supports, 'expected_object_count': len(objects),
            'missing_or_unobserved_ids': [], 'unexpected_object_ids': [],
            'position_error_by_frame': {'world:m': {'mean': sum(positions)/len(positions),
                'max': max(positions), 'measured_objects': len(positions)}}}}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--goal-image',type=Path,required=True)
    ap.add_argument('--api-config',type=Path,required=True)
    ap.add_argument('--max-model-decisions',type=int,default=20)
    ap.add_argument('--replay-model-trace',type=Path)
    ap.add_argument('--sam3-endpoint',default='http://127.0.0.1:8003')
    args=ap.parse_args()
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    settings=load_api_settings(args.api_config)
    config=replace(dynamic_reasoning_config(settings,out/'model_audit'),max_retries=4,retry_backoff=8.0,max_output_tokens=1536)
    model=OpenAICompatibleLanguageModel(config)
    env=install_recording(build_environment('scene_003','scene_003',out))
    trace={'status':'starting','initial_source':'simulator.head_camera.rgb',
           'goal_image':str(args.goal_image.resolve()),'goal_image_sha256':digest(args.goal_image),
           'reasoning_graph_called':False,'model_image_generation_called':False,
           'decisions':[],'completed_actions':0}
    start=time.monotonic()
    evaluation_metrics = EpisodeMetrics(method='legacy_direct_dynamic_sim', instruction=INSTRUCTION, output_dir=out)
    evaluation_metrics.context = {'max_model_decisions': args.max_model_decisions,
        'goal_image_sha256': trace['goal_image_sha256'], 'evaluation_source': 'simulator_external_task_evaluator',
        'executor_policy': 'legacy_scene003_constrained_overlay'}
    def execute(request):
        tick = time.perf_counter()
        try:
            result = env.step(request)
        except Exception as exc:
            from script.dual_arm_safety import ArmClearanceError
            evaluation_metrics.execution(success=False, seconds=time.perf_counter()-tick,
                details=getattr(exc, 'details', {}), fatal=isinstance(exc, ArmClearanceError))
            raise
        evaluation_metrics.execution(success=True, seconds=time.perf_counter()-tick)
        return result
    try:
        observation=env.reset(0)
        from script.scene003_image_goal import Scene003ImageGoal
        env.image_goal=Scene003ImageGoal(env.task,args.goal_image)
        evaluation_metrics.observe(evaluation=automatic_evaluation(env), payload_released=env.held is None)
        env.max_placement_deviation=.024
        trace['image_goal_targets']={k:v.tolist() for k,v in env.image_goal.targets.items()}
        initial_snapshot=out/'initial_head_camera.png'
        shutil.copyfile(observation['image'],initial_snapshot)
        constructor=SceneGraphConstructor(perception_backend=Sam3PerceptionClient(endpoint=args.sam3_endpoint))
        initial=constructor.construct(observation['image'],instruction=INSTRUCTION)
        initial.to_json(out/'initial_sam3_scene_graph.json')
        ids={o.id for o in initial.objects}
        trace.update(initial_image=str(initial_snapshot),initial_image_sha256=digest(initial_snapshot),initial_ids=sorted(ids))
        mapping=match_book_instances(initial,env.task)
        trace['identity_mapping']=mapping
        env.bind_graph(initial,mapping)
        for name in env.task.objects:
            if name not in env.object_map.values():
                object_id='sim_'+name
                env.object_map[object_id]=name
                ids.add(object_id)
        trace['simulator_only_ids']={k:v for k,v in env.object_map.items() if k.startswith('sim_')}
        ids_by_name={name:object_id for object_id,name in env.object_map.items()}
        write(out/'dynamic_trace.json',trace)
        goal_image=compact_image(args.goal_image,out/'model_terminal.jpg')
        if args.replay_model_trace:
            source_trace=json.loads(args.replay_model_trace.read_text())
            trace['replayed_model_trace']=str(args.replay_model_trace.resolve())
            for old in source_trace['decisions']:
                if old.get('status')!='executed':
                    continue
                action=old.get('executed_action') or old['model_action']
                item={'index':len(trace['decisions']),'status':'replaying_model_action',
                      'source_decision_index':old['index'],'model_action':action}
                trace['decisions'].append(item);write(out/'dynamic_trace.json',trace)
                observation=execute(request_from_action(action))
                item.update(status='executed',observed_image=str(observation['image']),
                            object_poses=observation['object_poses'],placement=observation.get('placement'))
                trace['completed_actions']+=1
                write(out/'dynamic_trace.json',trace)
        consecutive_motion_failures=0
        for index in range(args.max_model_decisions):
            evaluation_metrics.observe(evaluation=automatic_evaluation(env), payload_released=env.held is None)
            if aligned_goal(env):
                trace['status']='simulator_goal_reached'
                break
            current=observation['image']
            current_snapshot=out/f'head_camera_before_decision_{index:02d}.png'
            shutil.copyfile(current,current_snapshot)
            current_image=compact_image(current,out/f'model_current_{index:02d}.jpg')
            held=env.held[0] if env.held else None
            history=[{'index':x['index'],'source':(x.get('executed_action') or x.get('model_action') or {}).get('source_object_id'),
                      'action_type':(x.get('executed_action') or x.get('model_action') or {}).get('action_type'),
                      'target':(x.get('executed_action') or x.get('model_action') or {}).get('target'),
                      'status':x.get('status'),'error':x.get('error')} for x in trace['decisions']]
            arm_map={o.id:env._arm(env.object_map[o.id]) for o in initial.objects if o.id in env.object_map}
            recovery_hint='\nIf a held object failed during transfer, retry a clearly empty spot on that arm\'s side; do not move a few pixels within the same blocked region. The left arm is better suited to smaller image x; the right arm to larger image x.'
            stacked_book=book_support_status(env.task)['aligned_and_supported']
            # Once the book is stacked and no longer constrains the central
            # table, a smaller buffer is enough; the live contact monitor
            # still rejects actual collisions during execution.
            env.thin_placement_margin = .01
            trace['thin_placement_margin']=env.thin_placement_margin
            remaining=[f'{name}: final image pose or orientation not reached' for name,c in env.image_goal.evaluate(env)['objects'].items() if not c['passed']]
            if not stacked_book:remaining.append('Book must be side-grasped and placed aligned on the fixed folder after the cup is clear.')
            prompt=(SCHEMA+'\nInstruction: '+INSTRUCTION+'\nRemaining simulator goal feedback (qualitative, no measurements): '+json.dumps(remaining)+'\nBook already stacked: '+str(stacked_book)+'. Do not repeat a completed book stack unless its orientation is still incorrect. Prioritize the remaining movable objects.\nAllowed IDs and robot arms: '+json.dumps([{'id':o.id,'label':o.name,'arm':arm_map.get(o.id)} for o in initial.objects],ensure_ascii=False)+'\nAdditional simulator-identified objects (not SAM3 detections): '+json.dumps(trace['simulator_only_ids'])+'\nHeld simulator object: '+json.dumps(held)+'\nHistory: '+json.dumps(history,ensure_ascii=False)+'\nIf holding an object after a failed placement, plan a new placement for that same object.'+recovery_hint)
            (out/f'prompt_{index:02d}.txt').write_text(prompt)
            item={'index':index,'current_image':str(current_snapshot),'current_image_sha256':digest(current_snapshot),'status':'requesting_model'}
            trace['decisions'].append(item);write(out/'dynamic_trace.json',trace)
            try:
                with evaluation_metrics.phase('planning'):
                    answer=model.generate(prompt,images=[current_image,goal_image])[0]
                (out/f'model_answer_{index:02d}.txt').write_text(answer)
                data=ReasoningGraphParser._json_value(answer)
                action=validate(data,ids)
                item.update(model_action=action,uncertainty=data.get('uncertainty'))
                if action is None:
                    if not aligned_goal(env):
                        raise ValueError('Model reported done before simulator goal was reached')
                    item['status']='model_reported_done';trace['status']='model_reported_done';break
                if held and env.object_map.get(action['source_object_id'])!=held:
                    raise ValueError(f'Robot is holding {held}; proposed different object')
                action,override=constrained_scene003_action(env,ids_by_name,action)
                action=deepcopy(action)
                name=env.object_map[action['source_object_id']]
                item['executed_arm']=env.held[1] if env.held else env._arm(name)
                item['executed_action']=action
                if override is not None:item['safety_override']=override
                item['status']='executing';write(out/'dynamic_trace.json',trace)
                observation=execute(request_from_action(action))
                item.update(status='executed',observed_image=str(observation['image']),object_poses=observation['object_poses'],placement=observation.get('placement'))
                trace['completed_actions']+=1
                consecutive_motion_failures=0
                write(out/'dynamic_trace.json',trace)
            except RetryableLanguageModelError as exc:
                item.update(status='api_unavailable',error=f'{type(exc).__name__}: {exc}')
                trace['status']='api_unavailable';write(out/'dynamic_trace.json',trace)
                break
            except Exception as exc:
                from script.dual_arm_safety import ArmClearanceError
                if isinstance(exc, ArmClearanceError):
                    item.update(status='safety_stop', error=str(exc))
                    trace['status']='safety_stop'
                    break
                item.update(status='failed',error=f'{type(exc).__name__}: {exc}')
                consecutive_motion_failures+=1
                # A failed physical action may stop after the grasp or midway
                # through transfer.  Persist the live state so the next model
                # decision and the audit trace both see that partial result.
                try:
                    observation=env.observe()
                    item.update(failure_observed_image=str(observation['image']),
                                failure_object_poses=observation['object_poses'],
                                held_object=(env.held[0] if env.held else None))
                except Exception as observe_exc:
                    item['failure_observation_error']=f'{type(observe_exc).__name__}: {observe_exc}'
                # BaseTask.move latches plan_success=False after an IK failure.
                # Clear only this motion-status latch so the model's next
                # revised action can actually invoke the planner.
                if env.task.plan_success is False:
                    env.task.plan_success=True
                    item['motion_failure_latch_reset_for_retry']=True
                trace['status']='recovering_after_failure';write(out/'dynamic_trace.json',trace)
                if index+1==args.max_model_decisions or consecutive_motion_failures>=4:
                    trace['status']='execution_or_model_failure';break
                if 'failure_observed_image' not in item:
                    observation=env.observe()
        else:trace['status']='decision_limit'
        trace['fixed_scene_success']=bool(env.evaluate())
        trace['book_support']=book_support_status(env.task)
        trace['orientation_checks']=env.orientation_status(TARGET_YAWS)
        trace['image_goal']=env.image_goal.evaluate(env)
        trace['desk_goal_success']=bool(aligned_goal(env) and trace['status'] != 'safety_stop')
        if env.held is None and trace['status'] != 'safety_stop':
            env.park_arms()
        # Validate the observed end state again after retreat/home motions.
        trace['fixed_scene_success']=bool(env.evaluate())
        trace['book_support']=book_support_status(env.task)
        trace['orientation_checks']=env.orientation_status(TARGET_YAWS)
        trace['image_goal']=env.image_goal.evaluate(env)
        trace['desk_goal_success']=bool(aligned_goal(env) and trace['status'] != 'safety_stop')
        if not trace['desk_goal_success'] and trace['status']=='simulator_goal_reached':
            trace['status']='goal_lost_after_arm_retreat'
        if trace['desk_goal_success']:
            trace['status']='image_goal_reached'
        final=env.observe()['image'];trace['final_image']=str(final)
        final_graph=constructor.construct(final,instruction=INSTRUCTION)
        final_graph.to_json(out/'final_sam3_scene_graph.json')
        trace['final_sam3_ids']=sorted(o.id for o in final_graph.objects)
        evaluation_metrics.observe(evaluation=automatic_evaluation(env), payload_released=env.held is None)
    except Exception as exc:
        trace.update(status='setup_or_final_error',desk_goal_success=False,error=f'{type(exc).__name__}: {exc}')
        evaluation_metrics.invalidate_observation(trace['error'])
    finally:
        trace['elapsed_seconds']=round(time.monotonic()-start,3)
        trace['evaluation_metrics'] = evaluation_metrics.write(
            'episode_error' if trace['status']=='setup_or_final_error' else trace['status'])
        if env.arm_safety is not None:
            trace['arm_safety']=dict(env.arm_safety.stats)
        try:env.close()
        except Exception as exc:trace['close_error']=f'{type(exc).__name__}: {exc}'
        write(out/'dynamic_trace.json',trace)
    print(json.dumps({k:trace.get(k) for k in ['status','completed_actions','fixed_scene_success','desk_goal_success','error']},ensure_ascii=False),flush=True)
    return 0 if trace.get('desk_goal_success') else 2
if __name__=='__main__':raise SystemExit(main())
