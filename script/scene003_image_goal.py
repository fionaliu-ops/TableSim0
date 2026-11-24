"""Measured layout of the user-confirmed terminal_image_v2.png.

Image centres were read from that image, not from the legacy region predicate.
A calibrated camera ray maps each visible centre to its physical centre plane.
"""
import hashlib
import numpy as np

IMAGE_SHA256 = '7d342baf409ae0b34324f9c1615719af8cd6ce9b7db81bad5a44d9ffc019f2fe'
CENTERS = {'folder': [.336, .387], 'book': [.338, .381],
           'phone': [.450, .420], 'keyboard': [.560, .428],
           'mouse': [.669, .430], 'cup': [.687, .324], 'stapler': [.337, .521]}
YAWS = {'folder': 0., 'book': -90., 'phone': 0., 'keyboard': 0., 'mouse': 0., 'stapler': -90.}


class Scene003ImageGoal:
    def __init__(self, task, image_path):
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != IMAGE_SHA256:
            raise ValueError('Goal annotations do not match the supplied image hash')
        self.task = task
        cfg = task.cameras.get_config()['head_camera']
        self.k = np.asarray(cfg['intrinsic_cv'])
        self.extrinsic = np.asarray(cfg['extrinsic_cv'])
        if self.extrinsic.shape == (3, 4):
            self.extrinsic = np.vstack((self.extrinsic, [0., 0., 0., 1.]))
        self.camera_to_world = np.linalg.inv(self.extrinsic)
        rgb = task.get_obs()['observation']['head_camera']['rgb']
        self.size = np.array([rgb.shape[1], rgb.shape[0]])
        self.targets = {}
        for name, xy in CENTERS.items():
            height = task._table_z() + .006 + task.object_geometry[name]['height'] / 2
            if name == 'book':
                height += task.object_geometry['folder']['height']
            ray = self.camera_to_world[:3, :3] @ np.linalg.solve(self.k, np.r_[np.array(xy) * self.size, 1.])
            origin = self.camera_to_world[:3, 3]
            self.targets[name] = (origin + ray * ((height - origin[2]) / ray[2]))[:2]
        # A stack shares its support's centre even though perspective and height
        # shift the two visible centres slightly.
        self.targets['book'] = self.targets['folder'].copy()

    def evaluate(self, env):
        checks = {}
        for name, target in self.targets.items():
            pose = self.task.objects[name].get_pose()
            position = np.asarray(pose.p)
            tilt = (float(np.degrees(np.arccos(np.clip(pose.to_transformation_matrix()[2, 1], -1., 1.))))
                    if hasattr(pose, 'to_transformation_matrix') else 0.)
            center = position.copy()
            center[2] += self.task.object_geometry[name]['height'] / 2
            camera = self.extrinsic @ np.r_[center, 1.]
            pixel = self.k @ camera[:3]
            xy = pixel[:2] / pixel[2] / self.size
            pixel_error = float(np.linalg.norm(xy - CENTERS[name]))
            metric_error = float(np.linalg.norm(position[:2] - target))
            checks[name] = {'target_xy': target.tolist(), 'actual_xy': position[:2].tolist(),
                            'target_image_xy': CENTERS[name], 'actual_image_xy': xy.tolist(),
                            'image_error': pixel_error, 'position_error_m': metric_error,
                            'tilt_degrees': tilt,
                            'passed': pixel_error <= .035 and metric_error <= .025 and tilt <= 10.}
        orientations = env.orientation_status(YAWS)
        supports = {}
        if {'book', 'folder'} <= set(self.task.objects):
            book = np.asarray(self.task.objects['book'].get_pose().p)
            folder = np.asarray(self.task.objects['folder'].get_pose().p)
            gap = float(book[2] - folder[2] - self.task.object_geometry['folder']['height'])
            offset = float(np.linalg.norm(book[:2] - folder[:2]))
            supports['book_on_folder'] = {'support_gap_m': gap, 'center_offset_m': offset,
                                           'passed': 0 <= gap <= .015 and offset <= .025}
        return {'passed': all(v['passed'] for v in checks.values()) and all(v['passed'] for v in orientations.values())
                and all(v['passed'] for v in supports.values()), 'supports': supports,
                'position_tolerance_m': .025, 'image_tolerance': .035,
                'objects': checks, 'orientations': orientations,
                'goal_image_sha256': IMAGE_SHA256}
