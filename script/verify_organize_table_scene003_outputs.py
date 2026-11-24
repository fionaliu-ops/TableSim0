"""Validate both collect_data episodes, including every stored image/video frame."""
import hashlib
import json
import pickle
from pathlib import Path

import cv2
import h5py
import numpy as np

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('output', type=Path)
args=parser.parse_args()
out=args.output.resolve()
root=Path(__file__).resolve().parents[1]
seeds = list(map(int, (out / 'seed.txt').read_text().split()))
assert seeds == [0, 1], seeds
info = json.loads((out / 'scene_info.json').read_text())
assert set(info) == {'episode_0', 'episode_1'}
plan_digest = hashlib.sha256((root / 'planner_outputs/scene_003_kimi_api_unique_goal_20260919/action_flow.json').read_bytes()).hexdigest()
report = {'entrypoint': 'script/collect_data.py', 'task': 'organize_table_scene003',
          'config': out.name, 'seeds': seeds,
          'mode': 'contact_physics', 'physical_grasp_verified': True, 'episodes': []}
for ep in range(2):
    meta = info[f'episode_{ep}']
    assert meta['execution']['plan_sha256'] == plan_digest
    assert meta['execution']['mode']=='contact_physics'
    assert meta['execution']['physical_grasp_verified']
    adapted=meta['execution']['plan_adaptation']
    assert adapted['removed_object']=='book_sam3_2'
    assert adapted['cup_final_support']=='book_sam3'
    grasps=meta['execution']['physical_grasps']
    for stage in ('01_move_cup','05_move_mouse','07_place_cup','09_move_phone'):
        assert all(grasps[stage][key] for key in ('two_finger_contact','lift_verified','transfer_verified'))
    for stage in ('02_move_book','06_move_keyboard'):
        assert grasps[stage]['mode']=='contact_push' and grasps[stage]['error_xy']<.025
    if adapted.get('layout')=='compact_in_front_of_monitor':
        assert meta['goal_checks']['compact_layout']
        layout=meta['layout_metrics']
        assert layout['minimum_table_edge_margin']>=.17
        assert layout['occupied_area']<=.14
        assert layout['mean_distance_to_monitor']<=.26
        assert all(grasps['08_close_stapler/position'][key] for key in ('two_finger_contact','lift_verified','transfer_verified'))
    stages = meta['execution']['stages']
    assert [s['step'] for s in stages] == [1,2,5,6,7,8,9]
    assert all(s['success'] and not s.get('return_home_failed') for s in stages)
    assert all(meta['goal_checks'].values())
    instructions = json.loads((out / 'instructions' / f'episode{ep}.json').read_text())
    assert instructions['seen'] and instructions['unseen']
    with (out / '_traj_data' / f'episode{ep}.pkl').open('rb') as f:
        trajectory = pickle.load(f)
    assert trajectory['left_joint_path'] and trajectory['right_joint_path']
    with h5py.File(out / 'data' / f'episode{ep}.hdf5', 'r') as h5:
        rgb = h5['observation/head_camera/rgb']
        frames = len(rgb)
        assert frames > 0
        shapes = {}
        def inspect(name, node):
            if isinstance(node, h5py.Dataset):
                assert node.shape[0] == frames, (name, node.shape, frames)
                shapes[name] = list(node.shape)
                if node.dtype.kind in 'fiu':
                    assert np.isfinite(node[:]).all(), name
        h5.visititems(inspect)
        joint = h5['joint_action/vector'][:]
        assert np.any(np.ptp(joint, axis=0) > .01), 'Robot trajectory has no motion'
        assert h5['endpose/left_endpose'].shape[0] == frames
        assert h5['endpose/right_endpose'].shape[0] == frames
        for i, encoded in enumerate(rgb):
            image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
            assert image is not None and image.shape == (480, 640, 3), (ep, i)
            if i in (0, frames-1):
                suffix = 'initial' if i == 0 else 'final'
                cv2.imwrite(str(out / f'episode{ep}_{suffix}.png'), image)
    capture = cv2.VideoCapture(str(out / 'video' / f'episode{ep}.mp4'))
    assert capture.isOpened()
    fps = capture.get(cv2.CAP_PROP_FPS)
    decoded = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        assert frame.shape[:2] == (480, 640)
        decoded += 1
    capture.release()
    assert decoded == frames, (decoded, frames)
    result = dict(episode=ep, seed=seeds[ep], stages_passed=7, all_goal_checks_passed=True,
                  hdf5_frames=frames, video_frames=decoded, video_fps=fps,
                  instruction_counts={key:len(instructions[key]) for key in ('seen','unseen')},
                  trajectory_segments={key:len(trajectory[key]) for key in ('left_joint_path','right_joint_path')},
                  datasets=shapes)
    result['layout_metrics']=meta.get('layout_metrics')
    report['episodes'].append(result)
baseline=json.loads((root/'data/organize_table_scene003/physical_review/object_parameters_before.json').read_text())
changed=[p for p,h in baseline['files_sha256'].items() if hashlib.sha256((root/p).read_bytes()).hexdigest()!=h]
assert not changed, changed
report['unchanged_parameter_files']=len(baseline['files_sha256'])
report['passed'] = True
report['source_sha256'] = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (root/'envs/organize_table_scene003.py', root/'task_config'/f'{out.name}.yml')}
(out / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(report, ensure_ascii=False))
