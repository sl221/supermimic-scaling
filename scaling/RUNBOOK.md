# Scaling study runbook

Held-out data-scaling curve for SuperMimic: four runs (1, 7, 40 h and all remaining BONES-SEED), one seed each,
paper networks and hyperparameters, 40,000 iterations on the paper's 320,000-iteration learning-rate schedule.

`$DATA` is the shared network volume (same region as the GPUs). Every command runs from the repository root.

## 0. Environment (every machine)

```bash
apt-get update -qq && apt-get install -y -qq tmux libegl1 libgl1   # minimal images: MuJoCo needs libEGL to import
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
uv sync --group dev
export DATA=/path/to/shared/volume
export HF_TOKEN=...          # never commit this
export WANDB_API_KEY=...     # never commit this
```

## 1. BONES-SEED (CPU machine, at least 128 GB RAM)

Accept the license at https://huggingface.co/datasets/bones-studio/seed first.

```bash
uv run hf download bones-studio/seed g1.tar.gz metadata/seed_metadata_v004.csv --repo-type dataset --local-dir $DATA/raw
uv run python scripts/data/stream_bones_seed.py --archive $DATA/raw/g1.tar.gz --out $DATA/shards --workers 14   # 16-vCPU pod: leave cores for the archive reader
uv run python scripts/data/filter_bones_seed.py --shards $DATA/shards --parts $DATA/parts --workers 8
uv run python scripts/data/merge_library_parts.py --parts $DATA/parts --merged-manifest $DATA/merged.csv
uv run python scripts/data/filter_bones_seed.py --metadata $DATA/raw/metadata/seed_metadata_v004.csv --merged-manifest $DATA/merged.csv --out $DATA/keep.txt
uv run python scripts/data/merge_library_parts.py --parts $DATA/parts --clip-list $DATA/keep.txt --out $DATA/lib/bones_all.pt
```

Check: about 129,500 clips and 263 h.

## 2. Held-out splits and nested subsets

```bash
uv run python scaling/make_splits.py --library $DATA/lib/bones_all.pt --metadata $DATA/raw/metadata/seed_metadata_v004.csv --out-dir $DATA/splits
uv run python scaling/make_subsets.py --library $DATA/lib/bones_all.pt --train-pool $DATA/splits/train_pool.txt --out-dir $DATA/subsets --hours 1 7 40
```

Defaults: performer = `actor_uid`, motion group = `content_name` with the mirror suffix `_M` stripped (a take and its
mirror are held out together), motions with more than 100 clips never held out.

Check: no clip in two sets (the script guarantees it; `splits_report.json` shows the counts); subsets nested.

## 3. Libraries

```bash
for s in D1 D7 D40 Dall; do
  uv run python scripts/data/merge_library_parts.py --parts $DATA/parts --clip-list $DATA/subsets/$s.txt --out $DATA/lib/train_$s.pt
done
for h in heldout_motion heldout_performer; do
  uv run python scripts/data/merge_library_parts.py --parts $DATA/parts --clip-list $DATA/splits/$h.txt --out $DATA/lib/$h.pt
  uv run python scaling/segment_library.py --library $DATA/lib/$h.pt --out $DATA/lib/${h}_seg10.pt
done
# LAFAN1 (40 clips, never trained on): G1 retargets from lvhaidong/LAFAN1_Retargeting_Dataset, folder g1/
uv run python scripts/data/csv_to_npz_batch.py --input-dir $DATA/raw/lafan1/g1 --output-dir $DATA/npz/lafan1 --input-fps 30 --workers 16
uv run python scripts/data/build_library.py --npz-dir $DATA/npz/lafan1 --out $DATA/lib/lafan1.pt
uv run python scaling/segment_library.py --library $DATA/lib/lafan1.pt --out $DATA/lib/lafan1_seg10.pt
```

## 4. Smoke test (one H100)

```bash
uv run sm-train --run-name smoke --data.library $DATA/lib/train_D1.pt --env.obs-mode beyondmimic_seq \
    --schedule.stop-iteration 1000 --schedule.checkpoint-interval 500
uv run python scaling/eval_heldout.py --ckpt logs/smoke/ckpt_0001000.pt --out-dir results/smoke \
    --library motion=$DATA/lib/heldout_motion_seg10.pt lafan1=$DATA/lib/lafan1_seg10.pt
uv run python scaling/wm_error.py --ckpt logs/smoke/ckpt_0001000.pt --windows "results/smoke/*_windows.pt" --out results/smoke/wm_error.json
```

Record iterations per second and peak GPU memory from the training log before launching the real runs.

## 5. The four runs (one H100 each)

```bash
S=D40   # D1, D7, D40 or Dall
uv run sm-train --run-name $S --seed 0 --data.library $DATA/lib/train_$S.pt --env.obs-mode beyondmimic_seq \
    --schedule.stop-iteration 40000 --schedule.checkpoint-interval 4000 --log-root $DATA/logs
```

## 6. Evaluation (same GPU, after training)

```bash
for c in $DATA/logs/$S/ckpt_00*.pt; do
  uv run python scaling/eval_heldout.py --ckpt $c --out-dir $DATA/results/$S/$(basename $c .pt) \
    --library motion=$DATA/lib/heldout_motion_seg10.pt performer=$DATA/lib/heldout_performer_seg10.pt \
              lafan1=$DATA/lib/lafan1_seg10.pt
done
# once all four runs are evaluated, per checkpoint:
uv run python scaling/wm_error.py --ckpt $DATA/logs/{D1,D7,D40,Dall}/ckpt_0040000.pt \
    --windows "$DATA/results/*/ckpt_0040000/*_windows.pt" --out $DATA/results/wm_error_0040000.json
```
