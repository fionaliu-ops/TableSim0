"""Check artifact consistency only; no visual/physical execution claims."""
import copy
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
plan = json.loads((root / 'scene_003_action_plan.json').read_text())
checks = []

def check(condition, label):
    if not condition:
        raise ValueError(label)
    checks.append(label)

check(plan['provenance']['external_model_called'] is False, 'No external call recorded')
for item in plan['provenance']['inputs']:
    check(hashlib.sha256((root / item['path']).read_bytes()).hexdigest() == item['sha256'], f"Input hash: {item['role']}")
ids = {o['id'] for o in plan['objects']}
check(len(ids) == len(plan['objects']) == 8, 'Eight unique object identities')
actions = plan['actions']
check(len({a['id'] for a in actions}) == len(actions), 'Unique action identities')
check(len(plan['state_sequence']) == len(actions) + 1, 'One predicted state per action')
state = copy.deepcopy(plan['state_sequence'][0]['objects'])
initial_monitor = copy.deepcopy(state['monitor'])
seen = set()
for i, a in enumerate(actions, 1):
    check(a['object_id'] in ids and set(a['state_updates']) <= ids, f"{a['id']}: valid object references")
    check(set(a['depends_on']) <= seen, f"{a['id']}: dependencies precede action")
    if a['destination']:
        check(all(0 <= x <= 1 for x in a['destination']['approximate_center_uv']), f"{a['id']}: normalized coordinates")
    for obj, update in a['state_updates'].items():
        state[obj].update(update)
    recorded = plan['state_sequence'][i]
    check(recorded['objects'] == state and recorded['after_action'] == a['id'], f"{a['id']}: symbolic state update matches")
    check(set(state) == ids and state['monitor'] == initial_monitor, f"{a['id']}: objects preserved and monitor fixed")
    seen.add(a['id'])
check(state['book_light_blue']['support'] == 'book_dark_blue' and state['book_light_blue']['articulation'] == 'closed', 'Predicted closed book stack')
check(actions[4]['destination']['support_confirmed'] is False, 'Cup support uncertainty retained')
check(state['stapler']['articulation'] == 'unresolved_unchanged', 'No unsupported stapler close action')
report = dict(status='passed', checks_passed=len(checks), checks=checks,
              scope='Artifact consistency, dependency order and predicted symbolic state updates only',
              visual_accuracy_verified=False, physical_execution_verified=False,
              external_model_calls=0, required_action_stages=7, conditional_stages=2)
(root / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k:v for k,v in report.items() if k != 'checks'}, ensure_ascii=False))
