# Research diagnosis

## Data audit
- Number of unique `instance_id`s:
- Runs per issue and class balance:
- Observed failure taxonomy (with 3 real trace IDs):
- Potential leakage paths:

## Failure analysis
- One false-positive success judgment:
- One false-negative judgment:
- One case where failure attribution is not observable:

## Research ablations
- A: reward based only on resolved outcome vs outcome + grounded evidence.
- B: full capped trace vs last 12 messages.
- Controls: same issue-held-out split, seed, model, rollout budget, optimizer configuration.

## Final metrics
- Balanced accuracy:
- False-positive success rate:
- Fraction of zero-advantage groups:
- Evidence validity:
- Tokens/rollout; wall time; GPU hours:
