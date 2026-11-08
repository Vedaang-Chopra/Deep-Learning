# Post-Train Scheduler: Full Explanation in Simple Words

This doc explains the whole project. It uses one story: **a student and a teacher**.

Folder map:
- `EXPLANATION.md`: this file.
- `conversations/`: the 5 Claude Code chat sessions from this project, saved as readable text.

---

## 1. The story

A **student** is learning 5 subjects: planning, coding, reasoning, tool_use, math.
Each subject has a skill level from 0 to 1. At the start the student is weak.

The **teacher** has a pile of **120 homework questions** (these are the "tasks").
Each question needs some of the 5 subjects. A coding question may also need a little math.

The teacher has **5,760 tries** in total. One try = the student attempts one question once, and gets **pass** or **fail**.

**The teacher's job:** pick which question to give next, 5,760 times, so the student ends up as good as possible.

**The score:** the student's pass rate on a **secret exam** the teacher never sees. The exam uses a different kind of question than the homework.

In the real world, the "student" is an AI model, a "try" is a rollout (a costly model run), and the "teacher" is the scheduler we wrote.

---

## 2. What the teacher knows and does not know

| Teacher knows | Teacher does NOT know |
|---|---|
| The list of questions | How hard each question is |
| Which subjects each question uses | The student's real skill level |
| The exact learning rule | Which questions are poison (Part 2) |
| Pass or fail of each try | The exam questions |

The teacher also cannot peek at the simulator. Peeking = automatic fail.

---

## 3. How the student learns (the key rule)

Tries come in **groups of 8** on the same question.

- If **all 8 pass**: the question is too easy. The student learns **nothing**.
- If **all 8 fail**: the question is too hard. The student learns **nothing**.
- If results are **mixed** (say 4 pass, 4 fail): the student **learns a lot**.

Why? The method is called GRPO. It learns by comparing the good tries against the bad tries in a group. If all tries are the same, there is nothing to compare.

The learning amount:
- `spread = k × (8 − k) / 64`, where `k` is the number of passes.
- It is biggest at k = 4, and zero at k = 0 or k = 8.
- The gain is split across the subjects the question uses.
- The gain lands in the student's skill only every 60 tries (a "cadence" boundary).
- Gains shrink as the skill gets near 1: `skill += credit × (1 − skill)`.

**One sentence:** a good teacher gives questions the student gets right about half the time.

---

## 4. How hard a question is

A question has a hidden **difficulty** for each subject it needs.
- Chance to pass = the product, over its subjects, of a smooth S-curve of (skill − difficulty).
- If skill equals difficulty on a subject, that part passes 95% of the time.
- The teacher only sees the **shape** of a question: how much it needs each subject, compared with its biggest need. It does not see the overall **scale**, which is how hard it really is.

So the unknown for each question is just one number: its **scale**.

---

## 5. The baseline (what we need to beat)

The shipped baseline is a lazy teacher:
1. Give every question exactly 3 groups of 8 (6 groups per question over the whole run).
2. Go through all questions in order. Repeat.

What goes wrong (from `runs/baseline` and notebook 03):
- 720 groups were closed. Only 343 taught anything.
- About **52% of the budget** went to groups where all 8 agreed. Wasted.
- Baseline scores about 0.89 on a clean bank in my local proxy test.

Student version: the teacher keeps giving the same easy questions the student already knows, and the same impossible ones.

---

## 6. The idea behind my solution

Three simple ideas:

1. **Guess each question's difficulty from results.** Start by saying "it could be anything." After each group of 8, update the guess. If it passed 7 of 8, the question is probably easy. If 0 of 8, probably hard. (Idea from IRT adaptive testing, the way some exams pick questions to fit you.)
2. **Guess the student's skill by replaying the learning rule.** We know the rule. So after each group we can recompute the student's approximate skill ourselves. We start at a guess of 0.30 for every subject. The starting error shrinks with every update.
3. **Pick the question with the most expected learning.** Score = (chance of mixed results now) × (room left to improve on its subjects). Give that question a full group of 8. Repeat.

Good things this gives:
- Too-easy questions score low. Mastered subjects score low.
- Too-hard questions score low now, but come back later, because the score is recomputed from the student's current skill.
- No random exploration is needed. The group results themselves teach the teacher.

Ideas I read about but did not build (see `conversations/` chat 5a34da0b): Thompson sampling, trajectory difficulty, adaptive rollouts.

---

## 7. Part 2: poison questions

In Part 2, some questions may be **poison**. A poison question works backwards. Instead of raising the student's skill, it **lowers** it. Nothing marks which ones. The bank may have poison or not, and the teacher is not told.

How to catch one, in simple terms:
- Inside one group of 8, a poison question looks normal.
- But over time, a normal question gets **easier** as the student trains on it. A poison question gets **harder**, because the student is getting worse.
- So for every question we keep **two stories**: "it is clean" and "it is poison". We check which story explains the pass/fail history better.
- That gives a **chance of being poison** per question.

How the teacher acts on it:
- A question's value is reduced by its chance of being poison. If the chance is high, its value goes negative.
- A new, unchecked question gets only **one group per 60-try window**. This is because we only see the effect after the window ends.
- We also estimate the **share of poison in the whole bank**. On a clean bank this goes down, so the teacher stops being scared. On a bank with poison it goes up.
- If the teacher thinks it does harm, it fixes its own picture of the student's skill.

The cost: some of the budget is spent testing whether questions are safe.

---

## 8. Where everything is (code map)

| File | What it is |
|---|---|
| `curriculum/__init__.py` | `Scheduler` (Part 1) and `PoisonScheduler` (Part 2). The two classes the grader runs. `choose()` and `observe()`. |
| `curriculum/skill_model.py` | The maths. `TaskModel` (difficulty guess), `SkillTracker` (replays the student's skill), `SignedTask` (clean vs poison stories), `estimate_poison_rate`. |
| `tests/test_scheduler.py` | 12 tests. Dev only. |
| `local_scoring.py` | Makes a stand-in exam and scores runs. Dev only. Never imported by the scheduler. |
| `notebooks/01_inspect_tasks.ipynb` | Look at the task bank. |
| `notebooks/02_inspect_random_scheduler.ipynb` | Look at the random scheduler. |
| `notebooks/03_inspect_baseline.ipynb` | Baseline run and score. Prints the "number to beat". |
| `gymsim.py` | The simulator. Given. Not edited by us. |
| `bin/train`, `bin/check` | Given. Run training and self-check. |

Layers (as in the global rules): the notebooks only inspect. The two classes are the public interface. The maths is core logic. No logic lives in the notebooks.

How a scheduler turn works:
1. `choose()`: if the current group is finished, pick the highest-value question. Then hand out one try from it. (A group of 8 always stays on one question.)
2. The simulator runs the try and returns pass or fail.
3. `observe()`: count the result. When 8 are in, close the group, update that question's difficulty guess, and bank the credit. Every 60 tries, land the credit on the skill estimate.

Speed: only questions that changed get re-scored. The cost does not grow with history, as grading requires.

---

## 9. Checks I ran

1. **Unit tests: 12 pass.** They check:
   - My pass-chance formula matches the simulator's exactly.
   - The difficulty guess gets close to the true value.
   - A question that keeps failing loses priority, and comes back when skill rises.
   - On a clean bank, my replay of the student's skill matches the real skill exactly.
   - A question that gets worse with training is flagged as poison, and one that gets better is not.
   - Same seed gives the same run. Each group stays on one question.
2. **`./bin/check`: all pass.** Imports, 5,760 rollouts, fast enough, files written, two identical runs match.
3. **Local score comparison** (4 fresh banks, stand-in exam, a PROXY not the real score):

| World | Baseline | Mine |
|---|---|---|
| Part 1, no poison | 0.896 | 0.918 |
| Part 2, poison present | 0.806 (worst bank 0.638) | 0.904 (worst bank 0.873) |

The bigger gain is in Part 2, and the worst case is much better.

---

## 10. The last edit

Commit `578bf6e`: "drop lower skill clamp to match simulator".
- Before: `self.tracker.prof[skill] = min(1.0, max(0.0, value))`
- After: `self.tracker.prof[skill] = min(1.0, value)`
- Why: the real simulator lets skill go below 0 under poison. My extra floor made my picture of the student differ from the real one.
- Same change in `_land`, in `SignedTask._skills_under`, and in the matching test.

---

## 11. Honest limits

- The start skill of 0.30 is a guess, not tuned.
- Scores are from a proxy exam on 4 banks. The real exam is a different distribution.
- Poison detection needs several groups per question. A question tried once or twice cannot be judged.
- No random exploration. Ties go by bank order. This is repeatable, but not adaptive in that sense.
- Not done: more banks and poison levels, learning the start skill from early results, deliberately testing suspected poison.
- `git status` shows `tasks/train.jsonl` and two notebooks as modified. Check that this is intended before submitting.

---

## 12. Answers to the 5 Litmus questions (short)

1. **Walk through it:** teacher picks the question most likely to give mixed results. Section 6 and 8. Poison handling is section 7.
2. **The line I changed:** section 10.
3. **How I checked:** tests, `bin/check`, local score. Section 9.
4. **What I'd want to know first:** there is no printed score, so build a proxy first. Never read simulator state inside the scheduler. Keep per-call cost flat. Only 75 minutes, so choose ideas.
5. **What was not covered:** section 11.

---

## 13. Word list

- **Task / question:** one item the model can try.
- **Rollout / try:** one attempt. Costs budget.
- **Group:** 8 tries on the same task.
- **GRPO:** learns by comparing tries in a group.
- **Spread:** how mixed a group is. Bigger = more learning.
- **Skill (`prof`):** the student's level in one subject, 0 to 1.
- **Shape:** how much a task needs each subject, relative to its largest.
- **Scale:** how hard the task really is. Hidden.
- **Cadence:** gains land only every 60 tries.
- **Poison task:** a task that lowers skill instead of raising it.
- **Proxy score:** my own stand-in exam score. Not the graded one.

---

## 14. The saved chats (in `conversations/`)

| File | What it covers |
|---|---|
| `session_2026-11-04T00-54_78d03ba3.md` | Reading the README. Explaining the problem step by step, in simple words. |
| `session_2026-11-04T01-16_4e91655e.md` | The random scheduler: a one-file edit to `curriculum/__init__.py`, plus the notebook-writing skill. |
| `session_2026-11-04T01-21_61824180.md` | The inspection notebooks. Baseline explained in steps. |
| `session_2026-11-04T01-30_5a34da0b.md` | Which scheduler methods exist. IRT adaptive testing, Thompson sampling, trajectory difficulty, adaptive rollouts. |
| `session_2026-11-04T01-43_1401e146.md` | Questions about poison tasks, and a slow step-by-step walkthrough of how Scheduler 1 and the poison scheduler work. |

These are text-only versions: your messages and Claude's replies. Tool output and hidden system text are left out.
