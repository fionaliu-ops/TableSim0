"""Bounded scene_003 smoke run; use --play for contact-driven execution."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault('SAPIEN_RENDERER', 'opengl')
os.environ.setdefault('SAPIEN_NO_DISPLAY', '1')

import imageio.v2 as imageio
import collect_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--play', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/organize_table_scene003_smoke')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    def run(task, settings):
        settings.update(need_plan=args.play, save_data=False, render_freq=0)
        task.setup_demo(now_ep_num=0, seed=0, **settings)
        try:
            def save_frame(name):
                task.scene.update_render()
                for camera in task.cameras.static_camera_list:
                    camera.take_picture()
                imageio.imwrite(args.output / name, task.cameras.get_rgb()['head_camera']['rgb'])
            save_frame('initial.png')
            report = {'task': type(task).__name__, 'mode': 'contact_physics',
                      'physical_grasp_verified': False, 'plan_sha256': task.plan_sha256,
                      'initial_success': bool(task.check_success()),
                      'stage_targets': task.stage_targets, 'play_requested': args.play}
            if report['initial_success']:
                raise AssertionError('Initial state incorrectly satisfies the terminal goal')
            if args.play:
                task.play_once()
                save_frame('final.png')
                report.update(plan_success=bool(task.plan_success), goal_checks=task.goal_checks(),
                              objects={name:{'pose':[*map(float,obj.get_pose().p),*map(float,obj.get_pose().q)],
                                             'footprint':task.object_geometry[name]['footprint'].tolist(),
                                             'bottom':task._bottom(name),
                                             'angle':float(obj.angle) if hasattr(obj,'angle') else None}
                                       for name,obj in task.objects.items()},
                              completed_stages=task.completed_stages, stage_log=task.stage_log,
                              physical_grasps=task.physical_grasp_log, failure_reason=task.failure_reason,
                              success=bool(task.check_success()), score=task.get_score())
            (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
            print(json.dumps(report, ensure_ascii=False))
            if args.play and (not report['success'] or not report['plan_success']):
                raise SystemExit('Physical execution did not finish successfully; see report.json')
        finally:
            task.close_env(clear_cache=True)
    collect_data.run = run
    collect_data.main('organize_table_scene003', 'organize_table_scene003')


if __name__ == '__main__':
    main()
