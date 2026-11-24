"""Execute one model edge at a time, repairing from RGB and execution feedback."""
import json


def run_feedback_loop(env, planner, initial, target, graph, search, instruction, max_steps=20,
                      goal_revision=None):
    from script.dynamic_sim import LoopResult
    from script.dual_arm_safety import ArmClearanceError, TrajectoryClearanceError
    from high_level_planner.reasoning_graph.node_localization import locate_reasoning_node
    from high_level_planner.reasoning_graph.target_conditioned import (
        TargetGraphEvaluator, build_target_conditioned_generator)
    from high_level_planner.reasoning_graph.geometry import pose

    if planner._target_reasoning_generator is None:
        planner._target_reasoning_generator = build_target_conditioned_generator(
            initial, target, generator=planner.reasoning_generator,
            max_depth=4, max_states=8, max_repair_expansions=4)
        planner._target_scene_graph = target.copy()
        planner._target_position_tolerance = .02
    planner.memory.update(initial, graph, preserve_execution=False)
    evaluator = TargetGraphEvaluator(initial, target)
    current, records = initial, []
    visual_rotation_mode = target.metadata.get('yaw_semantics') == 'visual_delta_from_initial_reference_degrees'
    if hasattr(env, 'symbolic_grasp_controller'):
        env.symbolic_grasp_controller.yaw_semantics = target.metadata.get('yaw_semantics', 'world_absolute_degrees')
    path = env.output_dir/'dynamic_trace.json'
    def save(status, terminal=False):
        path.write_text(json.dumps({'status':status, 'execution_model':'symbolic_gripper_follow',
            'closed_loop':True, 'steps':records, 'observed_goal':terminal,
            'active_goal_revision':target.metadata.get('target_revision', 0),
            'goal_revisions':goal_revision.history if goal_revision else [],
            'simulation_truth_scope':('none_rgbd_and_robot_encoders' if getattr(env, 'visual_only_execution', False)
                                      else 'low_level_execution_and_failure_detection_only'),
            'robot_motion_executed':True, 'robot_grasp_verified':False}, ensure_ascii=False, indent=2))
        return LoopResult(status, len(records), terminal, terminal, path)
    def identity_ready(scene):
        return not getattr(env, 'visual_only_execution', False) or all(
            obj.metadata.get('visual_identity_confirmed') is not False for obj in scene.objects)
    def best_partial_edge(candidates):
        # A bounded graph may have no GOAL yet. Choose the model-proposed
        # branch whose reachable states satisfy most target predicates, not
        # the cheapest first edge (which can repeatedly move a solved object).
        def unsatisfied(node_id):
            scene = graph.get_node(node_id).scene_graph
            if scene is None:
                return float('inf')
            report = evaluator.evaluate(scene, instruction)
            return sum(not value for value in report.goal_checks.values())
        def score(edge):
            frontier, visited = [edge.to_state], set()
            best = unsatisfied(edge.to_state)
            while frontier:
                node = frontier.pop()
                if node in visited:
                    continue
                visited.add(node)
                best = min(best, unsatisfied(node))
                frontier.extend(e.to_state for e in graph.neighbors(node)
                    if e.metadata.get('dependencies', {}).get('validity') != 'invalid'
                    and e.metadata.get('physical_execution', {}).get('status')
                        not in {'failed','failed_in_prior_attempt'})
            return best, unsatisfied(edge.to_state), edge.cost
        return min(candidates, key=score)
    for index in range(max_steps):
        if evaluator.evaluate(current, instruction).terminal and env.held is None and identity_ready(current):
            return save('symbolic_grasp_visually_verified', True)
        edge = search.next_subtask
        if edge is None:
            cursor = planner.memory.current_node_id or graph.start_id
            candidates = [candidate for candidate in graph.neighbors(cursor)
                if candidate.metadata.get('dependencies', {}).get('validity') != 'invalid'
                and candidate.metadata.get('physical_execution', {}).get('status')
                    not in {'failed', 'failed_in_prior_attempt'}]
            if not candidates:
                save('extending_observed_frontier')
                try:
                    repaired = planner.update_target_conditioned_scene(current, instruction,
                        observation_source='head_camera.observed_frontier',
                        allow_partial_execution=True)
                    current, graph, search = repaired.scene_graph, repaired.reasoning_graph, repaired.search_result
                    graph.to_json(env.output_dir/f'reasoning_graph_frontier_{index:03d}.json')
                    cursor = planner.memory.current_node_id or graph.start_id
                    candidates = [candidate for candidate in graph.neighbors(cursor)
                        if candidate.metadata.get('dependencies', {}).get('validity') != 'invalid'
                        and candidate.metadata.get('physical_execution', {}).get('status')
                            not in {'failed', 'failed_in_prior_attempt'}]
                except Exception as exc:
                    if records:
                        records[-1]['frontier_replan_error'] = str(exc)
                    return save('feedback_replan_failed')
                if not candidates and search.next_subtask is None:
                    return save('no_executable_path')
            edge = search.next_subtask or best_partial_edge(candidates)
        record = {'index':index, 'reasoning_from':edge.from_state, 'reasoning_to':edge.to_state,
                  'goal_revision':target.metadata.get('target_revision', 0),
                  'request':planner.vla_interface.format_subtask(edge).to_dict()}
        record['status'] = 'requested'
        records.append(record)
        save('executing_graph_edge')
        error = None
        try:
            if getattr(env, 'visual_only_execution', False):
                env.bind_graph(current)
            observation = env.step(planner.vla_interface.format_subtask(edge))
            record.update(status='executed', placement=env.last_placement)
        except Exception as exc:
            error = exc
            # A preflight rejection has not collided; runtime safety is fatal.
            fatal = isinstance(exc, ArmClearanceError) and not isinstance(exc, TrajectoryClearanceError)
            details = dict(getattr(exc, 'details', {}))
            details.setdefault('failure_category', 'trajectory_clearance' if isinstance(exc, TrajectoryClearanceError)
                               else 'motion_execution_failed')
            # Coordinates from low-level diagnostics are deliberately not fed
            # into the visual planning scene. Only action outcome and identities.
            details.update(error_type=type(exc).__name__, held_object=bool(env.held))
            if getattr(edge, 'source_object_id', None):
                details['source_visual_pose'] = pose(current, edge.source_object_id)
            details['requested_pose'] = (getattr(edge, 'parameters', {}).get('pose') or
                getattr(edge, 'parameters', {}).get('placement_target', {}).get('parameters', {}).get('pose'))
            if details.get('blocking_object_ids'):
                details['blocking_visual_poses'] = {key:pose(current, key) for key in details['blocking_object_ids']}
            record.update(status='execution_failed', error=str(exc), feedback=details)
            record['execution_evidence'] = planner.record_execution_result(
                edge, success=False, details=details, fatal=fatal)
            save('observing_execution_failure')
            if env.held is not None and not fatal:
                try:
                    env.symbolic_grasp_controller.recover_held()
                    record['recovery'] = 'robot_returned_payload_to_pickup_and_opened'
                except Exception as recovery_exc:
                    record['recovery_error'] = str(recovery_exc)
            observation = env.observe()
            if fatal:
                record['failure_observation'] = str(observation['image'])
                return save('safety_stop')
        record['observation'] = str(observation['image'])
        try:
            # Both successful and failed actions are localized from a fresh RGB.
            if record.get('feedback', {}).get('failure_category') == 'visual_identity_unresolved':
                current.metadata['force_visual_recheck_ids'] = [edge.source_object_id]
            current = planner.scene_updater.update_scene_graph(current, observation['image'], instruction=instruction,
                moved_object_id=edge.source_object_id if error is None else None)
            current.metadata.pop('force_visual_recheck_ids', None)
            if visual_rotation_mode:
                from script.online_rearrangement import refine_visual_rotations
                for obj in current.objects:
                    if obj.attributes.xy is not None and obj.metadata.get('observed_in_current_frame', True):
                        obj.metadata['planning_pose'] = {'frame':'image_normalized',
                            'xy':list(obj.attributes.xy), 'yaw_degrees':0.}
                refine_visual_rotations(initial, current, initial.image_path, observation['image'],
                    planner.scene_constructor.perception_backend,
                    audit_path=env.output_dir/f'visual_rotation_{index+1:03d}.json')
            # This is a runner policy, not a visual measurement. Perception
            # replaces scene metadata, so explicitly retain the execution mode.
            current.metadata['graph_validation_ignore_physical_feasibility'] = True
            current.to_json(env.output_dir/f'scene_graph_{index+1:03d}.json')
            metrics = getattr(env, '_episode_metrics', None)
            if metrics is not None:
                metrics.observe(current, payload_released=env.held is None)
            localization = locate_reasoning_node(current, graph,
                candidate_ids={edge.from_state, edge.to_state})
            if not localization.conclusive:
                global_location = locate_reasoning_node(current, graph)
                if global_location.conclusive:
                    localization = global_location
            record['node_localization'] = {'selected_node_id':localization.node_id,
                'expected_node_id':edge.to_state, 'conclusive':localization.conclusive,
                'matches_expected':localization.conclusive and localization.node_id == edge.to_state,
                'source':'camera_rgb_scene_graph'}
            if not error:
                matched = record['node_localization']['matches_expected']
                record['execution_evidence'] = planner.record_execution_result(edge, success=matched,
                    details={'failure_category':None if matched else 'visual_postcondition_mismatch',
                             'motion_completed':True, 'visual_transition_verified':matched})
            revised_goal = False
            if goal_revision and goal_revision.should_revise(record, held=env.held is not None):
                save('regenerating_terminal_layout_from_feedback')
                try:
                    new_target, repaired = goal_revision.revise(current, target, record, instruction, planner)
                    target = new_target
                    evaluator = TargetGraphEvaluator(initial, target)
                    current, graph, search = repaired.scene_graph, repaired.reasoning_graph, repaired.search_result
                    target.to_json(env.output_dir/'active_target_scene_graph.json')
                    record['repair'] = {'mode':'regenerated_terminal_layout',
                                        'revision':target.metadata['target_revision']}
                    revised_goal = True
                except Exception as revision_exc:
                    record['goal_revision_error'] = f'{type(revision_exc).__name__}: {revision_exc}'
                    # A rejected candidate never replaces the active goal.
            if revised_goal:
                pass
            elif not error and record['node_localization']['matches_expected']:
                planner.memory.current_node_id = localization.node_id
                planner.memory.scene_graph = current
                search = planner.graph_planner.plan(graph, start_id=localization.node_id)
                record['repair'] = {'mode':'reuse_verified_remaining_graph'}
            else:
                save('replanning_from_execution_feedback')
                # This API attaches execution evidence to the scene planning
                # context consumed by the external reasoning model.
                repaired = planner.update_target_conditioned_scene(current, instruction,
                    observation_source='head_camera.after_execution',
                    coverage={'kind':'camera_frame','image':str(observation['image'])},
                    allow_partial_execution=True)
                current, graph, search = repaired.scene_graph, repaired.reasoning_graph, repaired.search_result
                record['repair'] = graph.metadata.get('closed_loop', {}).get('latest')
            graph.to_json(env.output_dir/f'reasoning_graph_{index+1:03d}.json')
            if env.held is not None:
                # Failure is already delivered and replanned; do not start a
                # new pick while a failed carry still owns an object.
                return save('held_object_recovery_required')
        except Exception as exc:
            record['replan_or_observation_error'] = f'{type(exc).__name__}: {exc}'
            return save('feedback_replan_failed')
        save('running_symbolic_feedback')
    if evaluator.evaluate(current, instruction).terminal and env.held is None and identity_ready(current):
        return save('symbolic_grasp_visually_verified', True)
    return save('step_limit')
