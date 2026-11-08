"""Baseline TRL GRPO: classify real coding-agent outcomes.

Runs only after prepare_data.py and GPU environment setup.
The reward uses private labels supplied by the dataset, never shown in prompts.
This is reward training of a *trace verifier*, not a bug fixer.
"""
import argparse
import json
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer


def parse_prediction(x):
    if isinstance(x, list):
        x = x[-1].get('content', '') if x else ''
    if not isinstance(x, str):
        return None
    try:
        obj=json.loads(x.strip())
    except (json.JSONDecodeError,TypeError):
        return None
    if not isinstance(obj,dict) or type(obj.get('solved')) is not bool:
        return None
    if not isinstance(obj.get('failure_type'),str):
        return None
    if 'evidence_step' not in obj or (obj['evidence_step'] is not None and type(obj['evidence_step']) is not int):
        return None
    return obj


def outcome_reward(completions, resolved, **kwargs):
    """Outcome agreement from independently recorded evaluation labels."""
    out=[]
    for completion,gold in zip(completions,resolved):
        pred=parse_prediction(completion)
        out.append(1.0 if pred is not None and pred['solved']==gold else 0.0)
    return out


def schema_reward(completions, **kwargs):
    return [0.05 if parse_prediction(x) is not None else 0.0 for x in completions]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model', default='Qwen/Qwen3-0.6B')
    p.add_argument('--max-steps',type=int,default=60)
    p.add_argument('--output-dir',default='outputs/trace-verifier-grpo')
    a=p.parse_args()
    with open('data/train.jsonl') as f:
        rows=[json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError('No data; run prepare_data.py')
    train=Dataset.from_list(rows)
    conf=GRPOConfig(output_dir=a.output_dir,
                    learning_rate=2e-6,
                    max_steps=a.max_steps,
                    per_device_train_batch_size=4,
                    gradient_accumulation_steps=4,
                    num_generations=4,
                    max_completion_length=160,
                    temperature=0.9,
                    logging_steps=5,
                    chat_template_kwargs={'enable_thinking':False},
                    report_to='none')
    trainer=GRPOTrainer(model=a.model,args=conf,train_dataset=train,
                        reward_funcs=[outcome_reward,schema_reward])
    trainer.train()
    trainer.save_model(a.output_dir)

if __name__=='__main__':main()
