"""Failure-triggered Look -> Explain -> Replan (Pchelintsev et al., 2025)."""
import json

from .paper_common import ACTION_FORMAT, PaperPlanner, image_state


class LERaPlanner(PaperPlanner):
    """Success advances the plan with zero model calls; failure invokes 3 stages.

    Look sees current RGB and the failed action. Explain and Replan consume text.
    The supplied scene graph and rich controller failure details never enter them.
    An optional shared initial plan isolates replanning from initial-plan quality.
    """
    method = 'lera'

    def reset(self):
        super().reset()
        self.initialized = False

    def decide(self, observation):
        if not self.initialized:
            if not self.plan:
                if observation.image is None: raise ValueError('LERa initial planner requires RGB')
                images = [image_state(observation.image)]
                if self.target_image is not None: images.append(self.target_image)
                prompt = ('External initial planner: propose a full ordered task plan. '
                    'Image 1 is the current observation; image 2, if supplied, is the fixed goal.\n'
                    + ACTION_FORMAT + json.dumps(self.context(), ensure_ascii=False))
                status, self.plan = self.load_plan(self.ask('initial_plan', prompt, images))
                if not self.plan:
                    self.initialized = True
                    return self.output(status)
            self.initialized = True
        elif self.failed:
            if observation.image is None: raise ValueError('LERa Look requires fresh RGB after failure')
            failed_action = self.history[-1]['action']
            remaining = [a.to_dict() for a in self.plan]
            successful_actions = [item['action'] for item in self.history if item['success']]
            look_prompt = ('Inspect this current RGB image and the failed task action. Describe '
                'the scene, possible reasons this action failed, and visible evidence. '
                'State uncertainty rather than inventing hidden state. Give a short textual description.\n'
                + json.dumps({'failed_action': failed_action, 'object_catalog': self.object_catalog}, ensure_ascii=False))
            look = self.ask('look', look_prompt, [image_state(observation.image)])
            explain = self.ask('explain', 'Explain the failure using the visual description. '
                'Propose a corrected conceptual sequence, explaining the purpose of each step. '
                'Use free-form text.\n' + json.dumps({**self.context(), 'visual_description': look,
                'failed_action': failed_action, 'remaining_plan': remaining,
                'successful_actions': successful_actions}, ensure_ascii=False))
            prompt = ('Translate the corrective explanation into a revised remaining action plan. '
                'Preserve the original goal and output only executable action messages.\n'
                + ACTION_FORMAT + json.dumps({**self.context(), 'corrective_explanation': explain,
                                             'previous_remaining_plan': remaining,
                                             'successful_actions': successful_actions}, ensure_ascii=False))
            status, revised = self.load_plan(self.ask('replan', prompt))
            self.plan = revised
            self.failed = False
            self.replans += 1
            if not self.plan: return self.output(status)
        return self.output('action', action=self.plan[0]) if self.plan else self.output('done', 'plan_exhausted')
