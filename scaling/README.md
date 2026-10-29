# Held-out data scaling of SuperMimic

Does more motion data make [SuperMimic](https://arxiv.org/abs/2610.28480) better on motions it has never seen?
The paper's data sweep evaluates on training motions only. I trained it on 1, 7, 40 and 254 hours of BONES-SEED and
tested it on three held-out sets.

![Tracking return vs training data](figures/fig1_heldout_vs_data.png)

*Unseen BONES-SEED motions and performers flatten after 40 h. LAFAN1 keeps rising.*

**Takeaways**

- Unseen BONES-SEED motions: no gain beyond 40 h of data (at equal compute).
- A different motion-capture source (LAFAN1): large gains, best with all the data.
- With 1 h of data, the world model is far worse on unseen transitions, and it overfits while the policy does not.

| Training data | Unseen motions | Unseen performers | LAFAN1 |
|---|---:|---:|---:|
| 1 h | 0.867 | 0.872 | 0.624 |
| 7 h | 0.909 | 0.918 | 0.706 |
| 40 h | **0.926** | 0.929 | 0.755 |
| 254 h | **0.926** | 0.930 | **0.779** |

Mean return (1.0 = perfect tracking), averaged over the last 3 checkpoints. Full numbers:
[`results/results.md`](results/results.md).

Built on [LaoYang1994/SuperMimic-Rep](https://github.com/LaoYang1994/SuperMimic-Rep), a fork of
[CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced) (MIT).

## Findings

### 1. Unseen BONES-SEED motions: no gain beyond 40 h

Return on unseen motions: 0.867 → 0.909 → 0.926 → 0.926 (1 h → 254 h).

<details>
<summary>Details</summary>

- Unseen performers show the same pattern.
- The paper found the same plateau on training motions.
- 1 h of data gives a lower ceiling, not a collapse. It plateaus at about 0.865 by 20k iterations and does not decline.
  But it never reaches 0.90 and fails 4% of unseen-performer segments (0.1% at 40 h and 254 h).
- The gap between training and unseen motions shrinks: 0.093 (1 h), 0.031 (7 h), 0.016 (40 h), about 0 (254 h).
  The training score uses a sample of each run's own data, so it is measured on different clips for each run.

</details>

![Learning curves](figures/fig2_learning_curves.png)

*Learning curves on each evaluation set.*

### 2. A different capture source (LAFAN1): keeps improving

Return 0.62 → 0.78 and failures 16% → 7% (1 h → 254 h).

<details>
<summary>Details</summary>

- The last step (+0.024, 40 h → 254 h) is within seed noise (see [Caveats](#caveats)).
- 254 h is ahead of 40 h on LAFAN1 at 9 of 10 checkpoints.

</details>

### 3. The world model: small data overfits

Error on unseen transitions (8-step, ×10⁻³): 44.5 → 21.3 → 17.9 → 17.7 (1 h → 254 h).

![World-model error](figures/fig3_wm_error.png)

*Left: error vs data. Right: the 1 h world model gets worse with more training.*

<details>
<summary>Details</summary>

- On its own training rollouts, the 1 h model's error is only 7.2. It is 6× worse on unseen transitions (1.4× at 254 h).
- Its unseen-transition error is lowest at 8k iterations (31) and rises to 47 by 40k. Its policy's held-out return
  does not drop.
- Error on its own rollouts rises with data (7.2 → 13.0), a trend similar to the paper's Fig. 3(c).
- The error is the paper's metric: 8 steps open-loop on logged actions. Every run's world model is scored on the same
  transitions, recorded while all four policies played the held-out sets. At each checkpoint, those come from that
  checkpoint's policies.

</details>

### 4. Compute to target: the paper's ranking reverses

Iterations to reach a return of 0.85:

- **On training motions,** 1 h is fastest (5.7k vs 9.3k for 254 h), as in the paper's Fig. 3(b).
- **On unseen motions,** 1 h is slowest (10.4k vs 7.3k for 254 h), and it never reaches 0.90.

![Iterations to target](figures/fig4_compute_to_target.png)

<details>
<summary>Details</summary>

| Training data | Training motions | Unseen motions |
|---|---:|---:|
| 1 h | 5.7k | 10.4k |
| 7 h | 7.4k | 7.8k |
| 40 h | 6.7k | 7.4k |
| 254 h | 9.3k | 7.3k |

Crossings are interpolated between checkpoints, which are 4,000 iterations apart. GPU-hours are in
[`results/results.md`](results/results.md).

</details>

## How this was built

34 hours from idea to results: about 12 hours of my time and 27 H100-hours (about $120). I used an AI agent (Claude)
for most of the reading, coding and debugging.

| Phase | When (Toronto time) | Who |
|---|---|---|
| Idea and literature check | Tue 21:00 – 23:00 | agent researches, I choose |
| Experiment design and plan | Wed 11:30 – 13:30 | together; I set the scope |
| Code and tests | Wed 13:30 – 14:40 | agent, during my lunch |
| Accounts, GPUs, data pipeline | Wed 14:40 – 18:00 | I run commands, agent debugs |
| Smoke test and launch | Wed 18:00 – 19:00 | same |
| Training: 4 runs on 2 H100s | Wed 19:00 – Thu 06:00 | unattended |
| Evaluation and analysis | Thu 07:00 – 07:45 | agent writes, I run |
| Write-up | Thu 07:45 – evening | agent drafts, I edit |

<details>
<summary>Problems caught along the way</summary>

Checking real output against expectations caught:

- **Mirrored motions leaking into training.** BONES-SEED stores each mirrored clip under its own motion name. Found by
  inspecting the metadata before splitting.
- **A command-line flag** that silently dropped two of the three held-out sets.
- **A wrong metadata path** in the download step.
- **An interrupted run** (40 h), resumed from its checkpoint.

Commit timestamps and the run logs record the times above.

</details>

## Setup

- **Method:** SuperMimic with the paper's networks and hyperparameters, via SuperMimic-Rep. One H100 per run.
- **Training:** 40,000 iterations per run on the paper's 320,000-iteration schedule. Checkpoints every 4,000.
- **Data:** nested subsets of BONES-SEED of 1, 7, 40 and 254 h. Each subset contains the smaller ones.
- **Evaluation:** three held-out sets, cut into 10-second segments, each played once from its first frame.

<details>
<summary>Definitions and how the held-out sets were built</summary>

**Definitions**

- **Return:** reward summed over a segment, divided by the largest value it can take. 1.0 is perfect tracking.
- **Failure:** the robot falls (pelvis height or tilt beyond the training limits), which ends the segment.
- **Iteration:** one gradient update, about 2.5 simulator steps across 4,096 parallel environments.
- **254 h:** the paper's filters leave 263 h (129,511 clips, within 3 clips of the paper); 8.7 h of that is held out.

**Held-out sets** ([`make_splits.py`](make_splits.py)), carved out before any training subset:

- **Unseen performers:** every clip of randomly chosen performers (2,293 clips, 4.3 h).
- **Unseen motions:** every clip of randomly chosen motions, by any performer (2,006 clips, 4.4 h). Mirrored copies are
  held out with their originals. Very common motions (more than 100 clips) are never held out.
- **LAFAN1:** all 40 clips (2.45 h), a different capture source.

**Other measurements**

- **Training motions:** a fixed sample of up to 2,000 clips from each run's own subset.
- **Evaluation noise:** re-evaluating one checkpoint changed its return by 0.0007. Between checkpoints of one run,
  return varies by about ±0.005 on BONES-SEED and more on LAFAN1.

</details>

## Caveats

- **One seed per data size,** as in the paper. Differences between 40 h and 254 h are within seed noise.
- **40,000 of 320,000 iterations.** Mid-training results, not converged ones.
- **"Unseen motion" relies on the dataset's labels.** Two differently named motions can be similar. LAFAN1 is the clean
  test.
- **The 40 h run was interrupted** at iteration 6,400 and resumed from its 4,000 checkpoint.

<details>
<summary>More on the caveats</summary>

- Other reproductions report seed-to-seed spreads of 0.007–0.03. The LAFAN1 step from 40 h to 254 h (+0.024) is that
  size. The 40 h run's LAFAN1 score moved between 0.725 and 0.771 over its last three checkpoints.
- The resumed run refilled its replay buffer, which checkpoints do not store.
- The 254 h run briefly shared its GPU with a stray process. This changed its wall-clock time, not its iterations.

</details>

<details>
<summary>Related results</summary>

Checked on 2026-10-29 against [awesome-supermimic](https://github.com/miao-org/awesome-supermimic) and the
reproductions it links. Held-out results exist only for models trained on the full dataset. None report world-model
error on held-out data.

- [realmimic](https://huggingface.co/kevinhenryz/realmimic) (full BONES-SEED, two seeds, training length not stated):
  LAFAN1 10-second segments, return 0.821–0.828, 4–6% failures. Our 254 h run: 0.779, 7% failures, at 40,000 iterations.
- [ScaleBFM_Plus](https://github.com/Hxxxz0/ScaleBFM_Plus): 99.5% tracking success on 2,575 held-out motions, with its
  own split and success definition.

</details>

<details>
<summary>Reproduce</summary>

[`RUNBOOK.md`](RUNBOOK.md) has every command, from download to figures:

1. Build the BONES-SEED library (section 1).
2. Make the splits and subsets (2), then the libraries (3).
3. Train and evaluate each run with `--schedule.stop-iteration 40000` (5–6).
4. Run `post_runs.sh` (7). `analyze.py` regenerates every table and figure from the logs.

CPU tests: `python -m pytest scaling/tests`.

</details>
