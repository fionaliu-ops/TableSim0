"""Versioned, image-first goal revision driven by observed execution failures."""
import json
from pathlib import Path


class AdaptiveGoalRevision:
    def __init__(self, generator, initial, target, goal_image, output_dir, *, max_revisions=3):
        self.generator = generator
        self.initial = initial
        self.original_target = target.copy()
        self.active_target = target.copy()
        self.image = Path(goal_image)
        self.output_dir = Path(output_dir) / 'goal_revisions'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_revisions = max_revisions
        self.history = []

    def should_revise(self, record, *, held=False):
        activated = sum(item['status']=='activated' for item in self.history)
        if held or activated >= self.max_revisions or len(self.history) >= 3*self.max_revisions:
            return False
        category = record.get('feedback', {}).get('failure_category')
        feedback = record.get('feedback', {})
        if category == 'motion_execution_failed' and feedback.get('phase') in {
                'park_other_arm', 'grasp_approach', 'grasp_contact'}:
            return False  # Pickup failed before transport; changing the goal cannot fix contact.
        source = record.get('request', {}).get('object_target', {}).get('source')
        from high_level_planner.reasoning_graph.geometry import pose
        desired = pose(self.active_target, source) if source else None
        requested = feedback.get('requested_pose')
        if desired and requested and desired.get('frame') == requested.get('frame') == 'image_normalized':
            if (sum((a-b)**2 for a,b in zip(desired['xy'],requested['xy'])) > .02**2
                    or desired.get('support','table') != requested.get('support','table')):
                return False  # A failed buffer needs another route, not another terminal layout.
        if category == 'grasp_occluded':
            blockers = feedback.get('blocking_object_ids', [])
            measured = feedback.get('blocking_visual_poses', {})
            # If the obstructing objects are already at their terminal poses,
            # this is a conflict in the desired layout, not ordinary unstacking.
            return bool(blockers) and all(
                pose(self.active_target,key) and measured.get(key)
                and sum((a-b)**2 for a,b in zip(pose(self.active_target,key)['xy'],
                                               measured[key]['xy'])) <= .02**2
                for key in blockers)
        if category == 'destination_occupied':
            feedback = record.get('feedback', {})
            source = record.get('request', {}).get('object_target', {}).get('source')
            blockers = set(feedback.get('blocking_object_ids', []))
            for previous in reversed(self.history):
                if previous['status'] != 'activated':
                    continue
                same_cause = (source == previous['failed_request']['object_target'].get('source')
                    and blockers and blockers == set(previous['reason'].get('blocking_object_ids', [])))
                if same_cause:
                    before = previous.get('blocking_visual_poses', {})
                    after = feedback.get('blocking_visual_poses', {})
                    changed = any(key in before and key in after and before[key] and after[key]
                        and sum((a-b)**2 for a,b in zip(before[key]['xy'],after[key]['xy'])) > .03**2
                        for key in blockers)
                    if not changed:
                        return False  # Keep the revised goal; clear the unchanged blocker first.
                break
        # Grasp occlusion changes action ordering first; a rejected destination
        # or unreachable transport changes the desired layout itself.
        return category in {'destination_occupied', 'destination_obstructed',
                            'trajectory_clearance', 'motion_execution_failed'}

    @staticmethod
    def support_contract(scene):
        return {(r.source, 'inside' if r.relation == 'in' else r.relation, r.target)
                for r in scene.relations if r.relation in {'on','in','inside','attached_to'}}

    @staticmethod
    def retain_rotation_reference(current, revised):
        """The image reader returns current-to-target deltas, not initial-to-target."""
        for obj in revised.objects:
            observed = current.get_object(obj.id)
            if observed is None:
                continue
            if observed.metadata.get('visual_yaw_observable') is False:
                obj.attributes.yaw_degrees = None
                obj.metadata.get('planning_pose', {}).pop('yaw_degrees', None)
                obj.metadata['visual_yaw_observable'] = False
                continue
            delta = obj.attributes.yaw_degrees
            previous = observed.attributes.yaw_degrees
            if delta is not None and previous is not None:
                composed = (float(previous) + float(delta) + 180.) % 360. - 180.
                obj.attributes.yaw_degrees = composed
                obj.metadata.setdefault('planning_pose', {})['yaw_degrees'] = composed
                obj.metadata['revision_yaw_reference'] = {
                    'observed_initial_to_current':previous, 'model_current_to_target':delta}

    def revise(self, current, target, record, instruction, planner):
        from high_level_planner.reasoning_graph.target_conditioned import (
            prompt_graph, TargetGraphEvaluator, normalize_executable_target)
        from script.online_rearrangement import refine_visual_rotations

        revision_id = len(self.history) + 1
        folder = self.output_dir / f'revision_{revision_id:03d}'
        folder.mkdir(parents=True, exist_ok=False)
        # This allowlist deliberately excludes low-level world diagnostics.
        feedback = {key:value for key,value in record.get('feedback', {}).items()
                    if key in {'failure_category','blocking_object_ids','phase','object_moved','error_type'}}
        request = record['request']
        edge = request.get('metadata', {}).get('reasoning_edge', {})
        failed_request = {'instruction':request.get('instruction'),
                          'object_target':request.get('object_target', {}),
                          'action_type':edge.get('action_type'),
                          'parameters':edge.get('parameters', {})}
        audit = {'revision':revision_id, 'status':'requested',
                 'reason':feedback, 'source_node':record.get('node_localization'),
                 'failed_request':failed_request, 'current_image':str(current.image_path),
                 'blocking_visual_poses':{key:value for key,value in record.get('feedback', {}).get(
                     'blocking_visual_poses', {}).items() if value and value.get('frame')=='image_normalized'},
                 'previous_goal_image':str(self.image), 'physical_success_claimed':False}
        self.history.append(audit)
        def save():
            (folder/'revision.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
        current.to_json(folder/'current_scene.json')
        target.to_json(folder/'previous_target.json')
        save()
        try:
            prompt = (
                'Revise the terminal layout after the observed execution failure below. '
                'Reference image 1 is the CURRENT robot camera image; image 2 is the PREVIOUS '
                'desired terminal layout, not a current observation. Keep the camera viewpoint, '
                'table, fixed equipment, object sizes, identities and counts unchanged. '
                'Preserve the original organizing intent and support/containment relationships. '
                'Change only placements necessary to resolve the failure and dependent conflicts; '
                'keep already completed placements where possible. Choose reachable, visibly free '
                'tabletop locations with room for the gripper and object footprint. Do not merely '
                'copy the current unfinished arrangement, hide an object, or erase remaining goals. '
                'The previous image coordinates are revisable preferences, not immutable constraints. '
                'Generate a NEW pictorial terminal layout, not actions or a graph.\n'
                'ORIGINAL TASK: '+instruction+'\n'
                'CURRENT OBSERVED NODE: '+json.dumps(prompt_graph(current), ensure_ascii=False)+'\n'
                'PREVIOUS TERMINAL LAYOUT: '+json.dumps(prompt_graph(target), ensure_ascii=False)+'\n'
                'EXECUTION FEEDBACK: '+json.dumps(audit, ensure_ascii=False))
            (folder/'revision_prompt.txt').write_text(prompt)
            prediction = self.generator.revise(current.image_path, current_scene_graph=current,
                instruction=prompt, previous_goal_image=self.image)
            audit.update(candidate_goal_image=str(prediction.image.image_path),
                         candidate_bundle=str(prediction.bundle_path))
            save()
            if not prediction.ready_for_planning:
                raise ValueError('Revised layout has unresolved visual identities')
            revised = prediction.scene_graph
            if {o.id for o in revised.objects} != {o.id for o in current.objects}:
                raise ValueError('Revised goal changed the observed inventory')
            if self.support_contract(revised) != self.support_contract(self.original_target):
                raise ValueError('Revised goal changed the original support/containment intent')
            if target.metadata.get('yaw_semantics') == 'visual_delta_from_initial_reference_degrees':
                # Keep the same yaw origin across all revisions and live frames.
                self.retain_rotation_reference(current, revised)
                refine_visual_rotations(self.initial, revised, self.initial.image_path,
                    prediction.image.image_path, planner.scene_constructor.perception_backend,
                    audit_path=folder/'rotation_audit.json')
                # A new 2D illustration must not restore an axis already found
                # unobservable by the current RGB-D camera observation.
                for obj in revised.objects:
                    if current.get_object(obj.id).metadata.get('visual_yaw_observable') is False:
                        obj.attributes.yaw_degrees = None
                        obj.metadata.get('planning_pose', {}).pop('yaw_degrees', None)
                        obj.metadata['visual_yaw_observable'] = False
            # Fixed equipment cannot follow incidental drawing shifts. Use the
            # SAME normalized contract in the planner and final live evaluator.
            revised = normalize_executable_target(current, revised)
            revised.metadata['target_revision'] = revision_id
            revised.to_json(folder/'candidate_target.json')
            # Detect a no-op candidate before spending a reasoning-model call.
            if TargetGraphEvaluator(current, revised).evaluate(target, instruction).terminal:
                raise ValueError('Regenerated image did not change the blocked target layout')
            output = planner.revise_terminal_target(current, revised, instruction, revision=audit)
            audit.update(status='activated', goal_image=str(prediction.image.image_path),
                         new_graph_start=output.reasoning_graph.start_id)
            self.image = Path(prediction.image.image_path)
            self.active_target = revised.copy()
            output.reasoning_graph.metadata['target_revision'] = dict(audit)
            output.reasoning_graph.to_json(folder/'reasoning_graph.json')
            revised.to_json(folder/'active_target.json')
            save()
            return revised, output
        except Exception as exc:
            audit.update(status='rejected', error=f'{type(exc).__name__}: {exc}')
            save()
            raise
