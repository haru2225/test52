# test52 — test50 + denser training data + periodic (torus) noise

test50 (SiO2 Si-only, `--replicate 2`, cutoff=8.0/8.2, irreps l≤4, sigma-max=1.5, updates=50000),
with two changes made after diagnosing test50's actual failure mode: generated atoms settling
near *some* locally-plausible Si arrangement without returning to their true crystallographic
sites (per-atom displacement from the true lattice ~1.3 Å — about 10x real thermal motion — and
not improving with more reverse steps within one run; see test50's own README for the full
analysis). **The NequIP_TimeEmbed architecture itself is unchanged** — only the training data
source and the noise model changed.

## 1. More training data, drawn directly from the raw MD trajectories

test50/51's bundled `simu_data/reference_frames.npz` is a sparse, already-decorrelated sample
(stride 200 *dumped* frames = 40,000 MD steps apart) of `md/nvt_traj_0..3.lammpstrj` (10,001
dumped frames each, dumped every 200 MD steps). All 184 frames are thermal snapshots of the
**same one crystal** — there is no structural diversity in this dataset, only vibrational noise
around one fixed registration.

`prepare --trajectories PATH [PATH ...] --stride N --offset M` (new) reads the raw `.lammpstrj`
dumps directly with a small bespoke parser (not ASE — these dumps carry no species/mass info,
just `id type xu yu zu`; type 1 → Si, type 2 → O, verified against `md/nvt_traj_0.lammpstrj`'s own
first-frame atom-type counts: 64 of type 1, 128 of type 2, matching SiO2's 1:2 ratio) and can pull
far more, denser frames — e.g. `--stride 20` on all four trajectories gives ~1800 frames instead
of 184. **This is still all one crystal's thermal ensemble, not new structural diversity** — just
a much larger and less redundant sample of it. The raw `.lammpstrj` files are **not bundled** in
this repo (66 MB each) — point `--trajectories` at your own copy of the main ScoreMD repo's `md/`
directory. `--reference-frames` (test50/51's original path) still works unchanged as the default
when `--trajectories` isn't given.

## 2. Periodic (torus) noise, replacing RattleParticles + raw displacement target

Ported from `test39.py`'s own real-space (Angstrom, not fractional) periodic Brownian motion and
`wrapped_score_target` — vendored into `test52.py` unchanged in spirit. Forward process:
`x_sigma = (x_0 + sigma * Normal(0,I)) mod cell`; the network is trained to predict
`-sigma * grad log p_sigma(noisy | clean)` (the exact periodic-Gaussian conditional score), not
test38/50's raw unwrapped displacement.

This fixes a real, previously-unaddressed correctness gap test50/51 both inherited from test38:
their reverse loop's math *assumes* the starting position already carries `sigma=start_sigma` of
noise, but a clean or crystal-noised start never actually received that much real corruption in a
periodicity-consistent way — and `--init random` was never verified to be in-distribution for a
non-periodic noise model's `sigma_max` in the first place. With periodic noise, `train()` now
enforces (test39's own criterion) that `--sigma-max` is large enough for the terminal distribution
to actually be near-uniform over the cell before training starts — **on the native 13.573 Å box
this requires `sigma-max ≳ 10.4 Å`**, dramatically larger than test50's `sigma-max=1.5` (which was
tuned for real-space *local* corruption, a completely different regime). `--sigma-max` now
defaults to the dataset's own box length when omitted (test39's convention) rather than a fixed
1.5. `--init random` is consequently now a well-defined "generate from nothing" test, not an open
question.

Side effect: since the graph is always rebuilt fresh from the actual post-noise (wrapped)
positions, `--large-cutoff`'s original role (a wider pre-noise candidate margin before
downselecting) no longer applies — it's kept as a CLI flag only for interface continuity with
test50 and the half-box safety check.

## Everything else is unchanged from test50.py

Coarse-graining (Si-only, index-preserving), `--replicate`, cutoff/large-cutoff/irreps/updates
defaults, the train-time half-box safety guard, `trajectory.extxyz` export, the
deterministic-steps warning. See test50.py's and test39.py's own module docstrings for what they
each vendor from test38.py.

## Usage

```bash
git clone git@github.com:haru2225/test52.git
cd test52

module load singularity
singularity build test52.sif Singularity.def

# 1. Dataset: dense sample straight from the raw NVT trajectories, 2x2x2-tiled
#    (point TRAJECTORIES at your own copy of the main repo's md/ directory)
qsub -P PROJECT_ID -v STAGE=prepare,OUTPUT=sio2-si-only/dataset-dense,REPLICATE=2,\
TRAJECTORIES="/path/to/md/nvt_traj_0.lammpstrj /path/to/md/nvt_traj_1.lammpstrj /path/to/md/nvt_traj_2.lammpstrj /path/to/md/nvt_traj_3.lammpstrj",\
OFFSET=1000,STRIDE=20 \
    run_test52.pbs

# 2. Train (cutoff=8.0/8.2, irreps l<=4, updates=50000 inherited from test50;
#    sigma-max auto-defaults to the box's own longest side, e.g. 27.146 A for --replicate 2)
qsub -P PROJECT_ID -v STAGE=train,DATASET=sio2-si-only/dataset-dense,OUTPUT=sio2-si-only/checkpoint1 \
    run_test52.pbs

# 3. Generate, now with a well-defined --init random
qsub -P PROJECT_ID -v STAGE=generate,CHECKPOINT=sio2-si-only/checkpoint1/checkpoint.pt,\
OUTPUT=sio2-si-only/checkpoint1/generated,INIT=random,\
REVERSE_STEPS=3000,DETERMINISTIC_STEPS=300,TRAJECTORY_STRIDE=10 \
    run_test52.pbs
```

Locally (no PBS/Singularity):

```bash
python test52.py prepare --trajectories ../../../md/nvt_traj_0.lammpstrj \
    ../../../md/nvt_traj_1.lammpstrj ../../../md/nvt_traj_2.lammpstrj ../../../md/nvt_traj_3.lammpstrj \
    --offset 1000 --stride 20 --replicate 2 --output sio2-si-only/dataset-dense
python test52.py train --dataset sio2-si-only/dataset-dense --output sio2-si-only/checkpoint1 --device cuda
python test52.py generate --checkpoint sio2-si-only/checkpoint1/checkpoint.pt \
    --output sio2-si-only/checkpoint1/generated --init random --device cuda
```

## Status

Smoke-tested locally (CPU): `prepare` from raw trajectories (single-file, multi-file, combined
with `--replicate`), `train` with the auto-computed sigma-max, and `generate` in all three `--init`
modes all run end to end without error. The sigma-max validity guard was verified to correctly
reject an under-sized `--sigma-max` (e.g. test50's old default of 1.5, which fails badly on this
box — Fourier residual 0.79 instead of the required ≤1e-5) and to accept the auto-computed default.
`generate --init random`'s periodic wrap was verified to keep all positions within the cell.
**Not yet trained for real or run on GPU** — whether periodic noise + denser data actually closes
the per-atom lattice-site gap test50 showed is unverified.
