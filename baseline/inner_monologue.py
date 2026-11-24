"""Inner Monologue tabletop adaptation (Huang et al., CoRL 2022)."""
import json

from .paper_common import ACTION_FORMAT, PaperPlanner, describe_scene


class InnerMonologuePlanner(PaperPlanner):
    """Object+Scene+Success feedback as a growing textual interaction history.

    The scene descriptor is shared sensor perception, translated to language.
    There is no reasoning graph, symbolic rollout, or local action repair.
    """
    method = 'inner_monologue'

    def reset(self):
        super().reset()
        self.dialogue = []

    def decide(self, observation):
        if observation.scene is None:
            raise ValueError('Inner Monologue needs a sensor-derived passive scene description')
        self.dialogue.append({'Scene': describe_scene(observation.scene),
            'Success': self.history[-1]['success'] if self.history else None,
            'PreviousAction': self.history[-1]['action'] if self.history else None})
        if self.initial_plan_active:
            # Recovery-only protocol: all methods follow a common supplied
            # prefix until the first checker failure, then use their own method.
            if self.plan and not self.failed:
                return self.output('action', action=self.plan[0])
            self.initial_plan_active = False
        prompt = ('Choose the next tabletop skill from the language feedback and task. '
            'Keep track of achieved subgoals and what remains. Account for visible and '
            'previously observed occluded objects. A failed skill may need retrying or '
            'a different preceding action. Return one action, never an entire sequence.\n'
            + ACTION_FORMAT + '\nFor a skill return status=plan and exactly one ACTION.\n'
            + 'Example feedback: the previous move failed and a different visible object '
              'blocks the source. Next skill: move that blocker before retrying. '
              'Example feedback: previous move succeeded and one object remains misplaced. '
              'Next skill: move the remaining object, preserving completed placements.\n'
            + json.dumps({**self.context(), 'interaction_history': self.dialogue}, ensure_ascii=False))
        status, actions = self.load_plan(self.ask('next_skill', prompt))
        if len(actions) > 1:
            from high_level_planner.reasoning_graph.language_models.base import LanguageModelError
            raise LanguageModelError('Inner Monologue must choose exactly one skill')
        self.plan = actions
        self.dialogue.append({'RobotAction': actions[0].to_dict() if actions else status})
        return self.output('action' if actions else status, action=actions[0] if actions else None)
