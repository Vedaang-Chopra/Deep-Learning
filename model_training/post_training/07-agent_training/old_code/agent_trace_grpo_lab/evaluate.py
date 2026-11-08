"""Evaluate saved predictions. Input JSONL: {instance_id, resolved, solved, evidence_valid?}."""
import argparse
import json
from collections import Counter


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--predictions',required=True)
    args=p.parse_args()
    with open(args.predictions) as f:
        rows=[json.loads(s) for s in f if s.strip()]
    if not rows:
        raise ValueError('Empty predictions file')
    cm=Counter((bool(r['resolved']),bool(r['solved'])) for r in rows)
    tp=cm[(True,True)]; fn=cm[(True,False)]; tn=cm[(False,False)]; fp=cm[(False,True)]
    sens=tp/(tp+fn) if (tp+fn) else float('nan')
    spec=tn/(tn+fp) if (tn+fp) else float('nan')
    fpr=fp/(fp+tn) if (fp+tn) else float('nan')
    print(json.dumps({'n':len(rows),'tp':tp,'fn':fn,'tn':tn,'fp':fp,
                      'balanced_accuracy':(sens+spec)/2,
                      'false_positive_success_rate':fpr},indent=2))

if __name__=='__main__':main()
