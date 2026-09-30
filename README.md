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
far more, denser frames than the bundled 184-frame npz. The raw `.lammpstrj` files themselves are
**not bundled** in this repo (tens of MB each) — point `--trajectories` at your own copy if you
want to rebuild the dataset from scratch or with different files. `--reference-frames` (test50/51's
original path) still works unchanged as a `prepare` input when `--trajectories` isn't given.

**`train` doesn't need a `prepare` step at all** — `sio2-si-only/dataset-dense/` (bundled, ~26 MB)
is the *already-prepared* result of running `prepare` against **8** trajectories: the original
`md/nvt_traj_0..3.lammpstrj` (10,001 dumped frames each, 2 ns/replica) **plus 4 new, independent**
replicas run locally for this file, `md/nvt_traj_4..7.lammpstrj` (same input structure and Vashishta
potential, different seeds, 354,000 steps/replica, ~12 min each on 4 parallel CPU cores) — genuinely
new thermal samples, not just a denser re-read of the same 4 trajectories. Built with
`--offset 500 --stride 10 --replicate 2`: **4308 frames, 512 Si sites, 27.146 Å cell**. `train`'s
own `--dataset` now defaults to this bundled directory.

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

No `prepare` step needed — the dataset ships in the repo.

```bash
git clone git@github.com:haru2225/test52.git
cd test52

module load singularity
singularity build test52.sif Singularity.def

# 1. Train directly against the bundled sio2-si-only/dataset-dense/
#    (cutoff=8.0/8.2, irreps l<=4, updates=50000 inherited from test50;
#    sigma-max auto-defaults to the dataset's own box length, 27.146 A)
qsub -P PROJECT_ID -v STAGE=train,OUTPUT=sio2-si-only/checkpoint1 run_test52.pbs

# 2. Generate, now with a well-defined --init random
qsub -P PROJECT_ID -v STAGE=generate,CHECKPOINT=sio2-si-only/checkpoint1/checkpoint.pt,\
OUTPUT=sio2-si-only/checkpoint1/generated,INIT=random,\
REVERSE_STEPS=3000,DETERMINISTIC_STEPS=300,TRAJECTORY_STRIDE=10 \
    run_test52.pbs
```

Locally (no PBS/Singularity):

```bash
python test52.py train --output sio2-si-only/checkpoint1 --device cuda   # uses bundled dataset-dense/
python test52.py generate --checkpoint sio2-si-only/checkpoint1/checkpoint.pt \
    --output sio2-si-only/checkpoint1/generated --init random --device cuda
```

To rebuild the dataset yourself (e.g. from your own additional trajectories), `prepare` is still
there:

```bash
qsub -P PROJECT_ID -v STAGE=prepare,OUTPUT=sio2-si-only/dataset-custom,REPLICATE=2,\
TRAJECTORIES="/path/to/md/nvt_traj_0.lammpstrj /path/to/md/nvt_traj_1.lammpstrj ...",\
OFFSET=500,STRIDE=10 \
    run_test52.pbs
# then point STAGE=train at DATASET=sio2-si-only/dataset-custom
```

## Status

Smoke-tested locally (CPU): `prepare` from raw trajectories (single-file, multi-file, combined
with `--replicate`), `train` with the auto-computed sigma-max — including a full run against the
**actual bundled `dataset-dense/`** (4308 frames, cutoff=8, l≤4, 512 atoms/graph; 3 updates
completed correctly, checkpoint saved, ~14 MB) — and `generate` in all three `--init` modes all run
end to end without error. The sigma-max validity guard was verified to correctly reject an
under-sized `--sigma-max` (e.g. test50's old default of 1.5, which fails badly on this box —
Fourier residual 0.79 instead of the required ≤1e-5) and to accept the auto-computed default.
`generate --init random`'s periodic wrap was verified to keep all positions within the cell.
**Not yet trained for real (only a few smoke-test updates) or run on GPU** — whether periodic noise
+ denser, genuinely-independent 8-replica data actually closes the per-atom lattice-site gap
test50 showed is unverified.
