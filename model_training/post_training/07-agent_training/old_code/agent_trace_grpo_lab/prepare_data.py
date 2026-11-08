"""Download real trajectories and create a leakage-resistant training split.

The preparation intentionally keeps gold outcome fields ONLY as internal reward
columns, never in the model-visible prompt. This is a *static audit* baseline.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from datasets import load_dataset

def visible_messages(messages, max_messages=12, max_chars=1400):
    lines = []
    for i, m in enumerate((messages or [])[-max_messages:]):
        if not isinstance(m, dict):
            continue
        role = str(m.get('role', 'unknown'))
        content = m.get('content')
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        calls = m.get('tool_calls') or []
        if calls:
            content += '\nTOOL_CALLS: ' + json.dumps(calls, ensure_ascii=False, default=str)
        # Explicit allowlist: never serialize test_result/resolved from source row.
        lines.append(f'[{i} | {role}] ' + content[:max_chars])
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--max-rows',type=int,default=1200)
    p.add_argument('--train-issues',type=int,default=160)
    p.add_argument('--eval-issues',type=int,default=40)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    raw=load_dataset('SWE-Gym/OpenHands-Sampled-Trajectories',split='train')
    data=list(raw.select(range(min(a.max_rows,len(raw)))))
    by_issue=defaultdict(list)
    for r in data:
        if r.get('instance_id') and isinstance(r.get('resolved'), bool):
            by_issue[r['instance_id']].append(r)
    issues=sorted(by_issue)
    random.Random(a.seed).shuffle(issues)
    split_ids={'train':set(issues[:a.train_issues]),
               'eval':set(issues[a.train_issues:a.train_issues+a.eval_issues])}
    outdir=Path('data');outdir.mkdir(exist_ok=True)
    for split,ids in split_ids.items():
        with (outdir/f'{split}.jsonl').open('w') as f:
            for issue in sorted(ids):
                for row in by_issue[issue]:
                    prompt='Audit the following coding-agent trace. Return only JSON with keys solved (boolean), failure_type (string), evidence_step (integer or null). Do not trust the assistant self-report as a verifier.\\n'+visible_messages(row.get('messages'))
                    out={'instance_id':issue, 'run_id':str(row.get('run_id')),
                         'prompt':[{'role':'user','content':prompt}],
                         'resolved':row['resolved']}
                    # Hidden labels are kept for external metrics and the reward callback.
                    f.write(json.dumps(out,ensure_ascii=False)+'\n')
        print(split,'issues',len(ids),'rows',sum(len(by_issue[x]) for x in ids))
    print('Use a random seed and issue-level split. All runs of any issue stay in one partition.')

if __name__=='__main__':main()
