#!/usr/bin/env python3
"""test52: test50 (SiO2 Si-only, cutoff=8.0/8.2 + --replicate 2, irreps l<=4, sigma-max=1.5,
updates=50000), with two changes made after diagnosing test50's own failure mode -- generated
atoms settling near SOME locally-plausible Si arrangement without returning to their true
crystallographic sites (per-atom displacement from the true lattice ~1.3 A, ~10x real thermal
motion, and not improving with more reverse steps within one run):

(2) MORE TRAINING DATA, drawn directly from the raw NVT trajectories instead of the fixed
    184-frame pre-extracted snapshot: test50/51's `simu_data/reference_frames.npz` is a sparse,
    already-decorrelated sample (stride 200 dumped frames = 40000 MD steps apart) of
    `md/nvt_traj_0..3.lammpstrj` (10001 dumped frames each, dumped every 200 MD steps). All 184
    frames are thermal snapshots of the SAME one crystal -- there is no structural diversity in
    this dataset at all, only vibrational noise around one fixed registration. `prepare
    --trajectories PATH [PATH ...] --stride N` (new) reads the raw `.lammpstrj` dumps directly
    (a small bespoke parser below, not ASE -- these dumps have no embedded species/mass info,
    just `id type xu yu zu`; type 1 -> Si (64/frame), type 2 -> O (128/frame), verified against
    md/nvt_traj_0.lammpstrj's own first-frame atom-type counts) and can pull far more frames at a
    much finer stride (e.g. --stride 20 -> ~1800 frames instead of 184) -- still all one crystal's
    thermal ensemble, not new structural diversity, but a much larger and less redundant sample of
    it. `--reference-frames` (test50/51's original path, reading the bundled pre-extracted npz)
    still works unchanged as the default when `--trajectories` isn't given.

(3) PERIODIC (TORUS) NOISE replacing test38's RattleParticles + raw unwrapped-displacement
    target. Ported from test39.py's own real-space (Angstrom, not fractional) periodic Brownian
    motion and `wrapped_score_target` -- vendored into this file unchanged in spirit, adapted only
    to read plain `cell` arrays instead of test39's own `cell_lengths()` helper. Forward process:
    `x_sigma = (x_0 + sigma * Normal(0,I)) mod cell`; the network is trained to predict
    `-sigma * grad log p_sigma(noisy | clean)` (the exact periodic-Gaussian conditional score),
    not test38's raw dx. This fixes a real, previously-unaddressed correctness gap test50/51 both
    inherited from test38: their reverse loop's math ASSUMES the starting position already carries
    sigma=start_sigma of noise, but a clean or crystal-noised start never actually receives that
    much real periodic corruption -- and their `--init random` was never verified to be
    in-distribution for a non-periodic noise model's sigma_max in the first place. With periodic
    noise, `train()` now checks (test39's own criterion) that `--sigma-max` is large enough that
    the terminal distribution is actually uniform over the cell before allowing training to start,
    so `--init random` is now well-defined rather than an open question.

Everything else -- the coarse-graining (Si-only, index-preserving), NequIP_TimeEmbed itself,
cutoff/large-cutoff/irreps/updates defaults, `--replicate`, the train-time half-box safety guard,
`--init crystal`/`crystal-noised` (now periodically-consistent), the deterministic-steps warning,
trajectory.extxyz export -- is unchanged from test50.py. See test50.py's and test39.py's own module
docstrings for what they each vendor from test38.py.

    python test52.py prepare --trajectories ../../../md/nvt_traj_0.lammpstrj \
        ../../../md/nvt_traj_1.lammpstrj ../../../md/nvt_traj_2.lammpstrj \
        ../../../md/nvt_traj_3.lammpstrj --stride 20 --output sio2-si-only/dataset-dense
    python test52.py train --dataset sio2-si-only/dataset-dense \
        --output sio2-si-only/checkpoint1 --device cuda
    python test52.py generate --checkpoint sio2-si-only/checkpoint1/checkpoint.pt \
        --output sio2-si-only/checkpoint1/generated --init crystal-noised --device cuda

Everything test38.py's own CAVEAT says still applies: no scalar energy, no equilibrium claim, no
physical clock, one model per condition. Additionally here: no O positions are modeled at all, so
this is not a full SiO2 structure generator -- only the Si sublattice.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from functools import partial
from pathlib import Path

import ase.io
import numpy as np
import torch

torch.serialization.add_safe_globals([slice])  # e3nn loads its own constants.pt with torch.load

from ase import Atoms
from ase.neighborlist import primitive_neighbor_list
from e3nn import o3
from e3nn.nn import FullyConnectedNet, Gate
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import scatter

# ===== 基本設定・定数 =====
ROOT = Path(__file__).resolve().parent
FORMAT = "test52-sio2-si-cg-periodic-denoiser-v1"  # このtest52用チェックポイントの識別子(test50のものとは別)
DATASET_FORMAT = "test52-sio2-si-only-cg-v1"  # prepare()が書き出すデータセット形式の識別子
DATASET_FORMATS = {DATASET_FORMAT}  # 読み込めるデータセット形式(load_dataset()はtest38と同じ関数、中身の集合だけ差し替え)
CAVEAT = (
    "test52 sigma-conditioned periodic (torus) score denoiser with a reverse variance-exploding "
    "SDE sampler -- test38's NequIP_TimeEmbed architecture, unchanged, but trained against "
    "test39's periodic Brownian-motion forward process and wrapped conditional score (not "
    "test38/50's raw unwrapped displacement target), on an SiO2 (beta-cristobalite) Si-only "
    "coarse-graining (one CG site per original Si atom; every O atom of every SiO4 tetrahedron "
    "is dropped, not averaged), optionally trained on a much denser frame sample drawn directly "
    "from the raw NVT trajectories. Not a scalar energy or a temperature-conditioned equilibrium "
    "score. No O positions, no energy/virial, no explicit electrostatics, pressure, shear "
    "response or physical kinetics are provided. One model per condition; generation frames are "
    "not equilibrium MD data."
)
STOP = False  # SIGINT/SIGTERMを受け取ったらTrueにして、train/generateループを安全に中断させるフラグ


def request_stop(signum, frame):
    # シグナルハンドラ: Ctrl-CやHPCのジョブ時間切れ通知を受けたときに呼ばれる。
    # ここで即座に終了せず、ループ側にSTOPを見てもらってから
    # チェックポイントを保存してから終了する(生成/学習の再開性を保つため)。
    global STOP
    STOP = True


def positive(value):
    # argparseの型変換関数: 正の有限値であることを保証する(sigmaや学習率など)。
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return number


def count(value):
    # argparseの型変換関数: 1以上の整数であることを保証する(ステップ数など)。
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_count(value):
    # argparseの型変換関数: 0以上の整数であることを保証する(deterministic-stepsなど、0も許す値)。
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def digest(path):
    # ファイルのSHA-256ハッシュを計算する。データセットやチェックポイントが
    # 途中で書き換わっていないか(再現性)を確認するために使う。
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, obj):
    # JSONを一時ファイルに書いてからatomicにrename。書き込み途中でプロセスが
    # 落ちても、既存の正しいファイルが壊れた中途半端な内容で上書きされない。
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_checkpoint(path, obj):
    # チェックポイント(torch.save)も同様にtmp書き込み→rename でatomicに保存する。
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(obj, temporary)
    temporary.replace(path)


def new_output(path):
    # 出力先ディレクトリを新規作成する。既に中身がある場合はエラーにして、
    # 別の実行結果を誤って上書き・混在させないようにする。
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise ValueError(f"Use a new or empty output directory: {path}")
    return path


def device_for(name):
    # "auto"ならCUDA→MPS→CPUの優先順で自動選択。明示指定されたデバイスが
    # 実際には使えない場合はここでエラーにする(黙って別デバイスに落とさない)。
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(name)


def rng_state():
    # numpy/torch(+CUDA/MPSがあれば)の乱数状態をまとめて保存用に取得する。
    # --resumeで学習・生成を再開したときに、乱数列を継続させるために使う。
    state = {"numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state):
    # rng_state()で保存した乱数状態を復元する。
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


# ===== test37から移植したデータ・グラフ・種(species)まわりのヘルパー =====

def bessel(x, start=0.0, end=1.0, num_basis=8, eps=1e-5):
    """Vendored from DM2/src/graphite/nn/basis.py (function bessel)."""
    # スカラー値(ここではエッジの距離)を、複数のBessel基底関数の値に展開する。
    # NequIPが距離をそのまま数値として使うのではなく、周波数の異なる
    # sin波の重ね合わせとして表現することで、距離依存性を学習しやすくする。
    x = x[..., None] - start + eps
    c = end - start
    n = torch.arange(1, num_basis + 1, dtype=x.dtype, device=x.device)
    return ((2 / c) ** 0.5) * torch.sin(n * torch.pi * x / c) / x


class InitialEmbedding(nn.Module):
    """Same embedding as test32.py/test37.py: two species embeddings and Bessel edges."""
    # ノード(原子)の「種」を2種類の埋め込みベクトルに変換し、エッジ(ボンド)の
    # 距離をBessel基底に変換する、NequIPモデルの最初の入力層。
    def __init__(self, num_species, cutoff):
        super().__init__()
        self.embed_node_x = nn.Embedding(num_species, 8)  # 更新されていくノード特徴量の初期値
        self.embed_node_z = nn.Embedding(num_species, 8)  # モデル全体を通して固定される補助的なノード特徴量
        self.embed_edge = partial(bessel, start=0.0, end=cutoff, num_basis=16)

    def forward(self, data):
        data.h_node_x = self.embed_node_x(data.x)
        data.h_node_z = self.embed_node_z(data.x)
        data.h_edge = self.embed_edge(data.edge_attr.norm(dim=-1))
        return data


def architecture(num_species, cutoff, irreps_hidden="64x0e + 32x1e + 16x2e + 8x3e + 4x4e",
                  irreps_edge="4x0e + 4x1e + 2x2e + 2x3e + 1x4e"):
    # モデルの構造(irreps=e3nnの回転等変な特徴量の型、畳み込み層の数など)を
    # 1つの辞書にまとめたもの。チェックポイントに保存しておき、生成時に
    # 同じ構造のモデルを再構築するために使う。test37と同じ構造。
    # irreps_hidden/irreps_edge引数はtest50独自の追加(test38は引数なしのハードコード、
    # l<=1隠れ層/l<=2エッジ固定)。ここでのデフォルトはl<=4構成(下のtrainの--irreps-hidden/
    # --irreps-edgeのデフォルトと合わせてある)。
    return dict(num_species=num_species, cutoff_angstrom=cutoff,
                irreps_node_x="8x0e", irreps_node_z="8x0e",
                irreps_hidden=irreps_hidden, irreps_edge=irreps_edge,
                irreps_out="1x1e", num_convs=3, radial_neurons=[16, 64], num_neighbors=12)


def graph(positions, cell, type_ids, cutoff, device):
    # Same periodic neighbor construction as test32/37; graph vectors are not
    # differentiable w.r.t. positions. This is deliberate for a dx model.
    # 周期境界条件(PBC)付きで、cutoff半径以内の原子対(i, j)とその変位ベクトルvecを列挙し、
    # PyTorch Geometricの`Data`グラフオブジェクトを組み立てる。
    i, j, vec = primitive_neighbor_list("ijD", [True] * 3, cell, positions, cutoff=cutoff)
    if not len(i):
        raise ValueError("No graph edges: check box, units and cutoff")
    return Data(x=torch.as_tensor(type_ids, dtype=torch.long, device=device),
                pos=torch.as_tensor(np.asarray(positions).copy(), dtype=torch.float32, device=device),
                edge_index=torch.as_tensor(np.stack((i, j)), dtype=torch.long, device=device),
                edge_attr=torch.as_tensor(vec, dtype=torch.float32, device=device))


def load_dataset(folder):
    # test50形式のデータセット(positions.npy, cells.npy, metadata.json; prepare()が書き出す)を読み込む。
    # メタデータのフォーマット・単位・配列の形状・保存後の改ざん有無(sha256)を検証してから返す。
    # (関数自体はtest38.pyのload_dataset()と同一ロジック。DATASET_FORMATSの中身だけがtest50用に差し替わっている。)
    folder = Path(folder).resolve()
    meta = json.loads((folder / "metadata.json").read_text())
    if meta.get("format") not in DATASET_FORMATS or meta.get("length_unit") != "angstrom":
        raise ValueError("Expected a test50 SiO2 Si-only CG dataset with explicit angstrom units")
    pos = np.load(folder / "positions.npy", mmap_mode="r", allow_pickle=False)
    cells = np.load(folder / "cells.npy", mmap_mode="r", allow_pickle=False)
    if pos.shape != (meta["frames"], len(meta["type_ids"]), 3) or cells.shape != (len(pos), 3, 3):
        raise ValueError("Dataset shapes disagree with metadata")
    for name in ("positions.npy", "cells.npy"):
        if digest(folder / name) != meta["sha256"][name]:
            raise ValueError(f"Dataset modified after preparation: {name}")
    return pos, cells, meta


def atoms_from_meta(positions, cell, meta):
    # モデル用のtype_id配列を、可視化・エクスポート用にase.Atomsオブジェクト
    # (実際の原子番号・質量を持つ)へ変換する。test50ではspeciesはSi一種類のみ
    # (元のtest38/test37の粘土CGでは酸素の役割ごとに4種を別種として扱っていたが、
    # ここでは単にSi原子のインデックスをそのまま残す)。
    ids = np.asarray(meta["type_ids"])
    atoms = Atoms(numbers=[meta["species"][i]["atomic_number"] for i in ids],
                  positions=positions, cell=cell, pbc=True,
                  masses=[meta["species"][i]["mass_amu"] for i in ids])
    atoms.set_array("cg_type", ids.copy())
    return atoms


# --- Vendored from DM2/src/graphite (nn/conv/e3nn_nequip.py, nn/models/e3nn_nequip.py,
# transforms/downselect_edges.py, transforms/rattle_particles.py) so that this file does
# not import DM2 or require a DM2 checkout / DM2_ROOT to be present. -----------------

def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    # 2つのirreps(既約表現)のテンソル積が、指定した出力既約表現ir_outを
    # 生成しうるかどうかを判定する。e3nnのGate/TensorProductを組み立てる際に、
    # 「この組み合わせは数学的に意味があるか」を事前にチェックするために使う。
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)
    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


class Compose(nn.Module):
    # 2つのモジュール(例: Interaction畳み込み層とGate活性化層)を直列に繋げるだけの
    # 薄いラッパー。NequIP_TimeEmbedの各層は Compose(Interaction, Gate) として作られる。
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second
        self.irreps_in = self.first.irreps_in
        self.irreps_out = self.second.irreps_out

    def forward(self, *input):
        x = self.first(*input)
        return self.second(x)


class GaussianBasisEmbedding(nn.Module):
    """Embeds a scalar value in [0,1] using a Gaussian basis set followed by a dense layer."""
    # sigma/time条件付けの核心部分: スカラー値t(=sigma/sigma_max_train, 0〜1の値)を
    # 複数のガウス基底関数に展開し(bessel関数と同じ発想)、2層のMLPで
    # ベクトル特徴量に変換する。NequIP_TimeEmbedがこの出力(h_node_t)を
    # 各畳み込み層のノード特徴量に足し込むことで、モデルが「今どれくらいの
    # ノイズレベルを相手にしているか」を認識できるようになる。
    def __init__(self, num_basis=12, embedding_dim=32, min_sigma=0.1,
                 learn_means=False, learn_sigmas=False, min_value=0, max_value=1):
        super().__init__()
        means = torch.linspace(min_value, max_value, num_basis)  # 各ガウス基底の中心位置
        if learn_means:
            self.means = nn.Parameter(means)
        else:
            self.register_buffer('means', means)
        sigmas = torch.ones_like(means) * max(min_sigma, 1.0 / (num_basis - 1))  # 各ガウス基底の幅
        if learn_sigmas:
            self.sigmas = nn.Parameter(sigmas)
        else:
            self.register_buffer('sigmas', sigmas)
        hidden_dim = max(embedding_dim * 2, num_basis)
        self.layer1 = nn.Linear(num_basis, hidden_dim)
        self.activation = nn.Softplus()
        self.layer2 = nn.Linear(hidden_dim, embedding_dim)

    def gaussian_basis(self, x):
        # tの値を、各ガウス基底中心からの距離に応じた「近さ」のベクトルに変換する。
        if x.dim() == 1:
            x = x.unsqueeze(1)
        x_expanded = x.expand(-1, self.means.shape[0])
        return torch.exp(-0.5 * ((x_expanded - self.means) / self.sigmas) ** 2)

    def forward(self, x):
        basis_activation = self.gaussian_basis(x)
        hidden = self.activation(self.layer1(basis_activation))
        return self.layer2(hidden)


class Interaction(nn.Module):
    """Equivariant `Interaction` layer from NequIP (https://arxiv.org/pdf/2101.03164.pdf)."""
    # NequIPの中核となる1つの畳み込み(メッセージパッシング)層。
    # 回転・並進に対して等変(equivariant)なテンソル積を使って、
    # 各原子の近傍からの情報を集約し、ノード特徴量を更新する。
    def __init__(self, irreps_in, irreps_node, irreps_edge, irreps_out,
                 radial_neurons=[16, 64], num_neighbors=1):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_node = o3.Irreps(irreps_node)
        self.irreps_edge = o3.Irreps(irreps_edge)
        self.irreps_out = o3.Irreps(irreps_out)
        self.num_neighbors = num_neighbors

        # irreps_in(ノード特徴量)とirreps_edge(球面調和関数)のテンソル積のうち、
        # 最終的にirreps_outとして使えるものだけを集めて、中間表現irreps_midを組み立てる。
        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_in):
            for j, (_, ir_edge) in enumerate(self.irreps_edge):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, 'uvu', True))
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()
        assert irreps_mid.dim > 0, (
            f"irreps_in={self.irreps_in} times irreps_edge={self.irreps_edge} "
            f"produces nothing in irreps_out={self.irreps_out}."
        )
        instructions = [
            (i_1, i_2, p[i_out], mode, train)
            for i_1, i_2, i_out, mode, train in instructions
        ]

        self.sc = o3.FullyConnectedTensorProduct(self.irreps_in, self.irreps_node, self.irreps_out)  # 自己結合(残差)経路
        self.lin1 = o3.FullyConnectedTensorProduct(self.irreps_in, self.irreps_node, self.irreps_in)  # メッセージパッシング前の線形変換
        self.conv = o3.TensorProduct(
            self.irreps_in, self.irreps_edge, irreps_mid, instructions,
            internal_weights=False, shared_weights=False,  # 重みは下のmlp(ボンド長依存)から供給される
        )
        self.lin2 = o3.FullyConnectedTensorProduct(irreps_mid, self.irreps_node, self.irreps_out)  # メッセージパッシング後の線形変換
        self.mlp = FullyConnectedNet(radial_neurons + [self.conv.weight_numel], torch.nn.functional.silu)  # ボンド長(Bessel基底)からconvの重みを生成するMLP

        # SkipInit mechanism inspired by https://arxiv.org/pdf/2002.10444.pdf
        # 学習開始時点ではconvパスの寄与をゼロにしておき(alpha=0スタート)、
        # 学習が進むにつれて徐々にconvパスの影響を強めていく安定化トリック。
        self.alpha = o3.FullyConnectedTensorProduct(irreps_mid, self.irreps_node, "0e")
        with torch.no_grad():
            self.alpha.weight.zero_()
        assert self.alpha.output_mask[0] == 1.0, (
            f"irreps_mid={irreps_mid} and irreps_node={self.irreps_node} are not able to generate scalars."
        )

    def forward(self, x, node_attr, edge_index, edge_attr, edge_len_emb):
        i, j = edge_index
        num_nodes = x.size(0)
        node_self_connection = self.sc(x, node_attr)  # 残差(自己)経路の出力
        node_features = self.lin1(x, node_attr)
        # 各エッジについて、送り手ノードiの特徴量とエッジの球面調和関数edge_attrとの
        # テンソル積を、ボンド長依存の重み(self.mlp(edge_len_emb))で計算する。
        edge_features = self.conv(node_features[i], edge_attr, weight=self.mlp(edge_len_emb))
        # 受け手ノードjごとにエッジメッセージを合計(scatter)し、近傍数で正規化する。
        node_features = scatter(edge_features, j, dim=0, dim_size=num_nodes).div(self.num_neighbors ** 0.5)
        node_conv_out = self.lin2(node_features, node_attr)
        alpha = self.alpha(node_features, node_attr)
        m = self.sc.output_mask
        alpha = (1 - m) + alpha * m
        # 残差経路 + alphaでスケールした畳み込み経路、を足し合わせて出力する。
        return node_self_connection + alpha * node_conv_out


class NequIP_TimeEmbed(nn.Module):
    """Sigma/time-conditioned NequIP (https://arxiv.org/pdf/2101.03164.pdf), vendored from
    DM2/src/graphite/nn/models/e3nn_nequip.py.

    Args:
        init_embed (function): Initial embedding function/class for nodes and edges.
        irreps_node_x (Irreps or str): Irreps of input node features.
        irreps_node_z (Irreps or str): Irreps of auxiliary node features (not updated throughout model).
        irreps_hidden (Irreps or str): Irreps of node features at hidden layers.
        irreps_edge (Irreps or str): Irreps of spherical_harmonics.
        irreps_out (Irreps or str): Irreps of output node features.
        num_convs (int): Number of interaction/conv layers. Must be more than 1.
        radial_neurons (list of ints): Number of neurons per layers in the MLP that learns from bond distances.
        num_neighbors (float): Typical or average node degree (used for normalization).
    """
    def __init__(self, init_embed, irreps_node_x='8x0e', irreps_node_z='8x0e',
                 irreps_hidden='64x0e + 32x1e + 32x2e', irreps_edge='1x0e + 1x1e + 1x2e',
                 irreps_out='1x1e', num_convs=3, radial_neurons=[16, 64], num_neighbors=12):
        super().__init__()
        self.init_embed = init_embed
        self.irreps_node_x = o3.Irreps(irreps_node_x)
        self.irreps_node_z = o3.Irreps(irreps_node_z)
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_edge = o3.Irreps(irreps_edge)
        self.num_convs = num_convs

        act_scalars = {1: nn.functional.silu, -1: torch.tanh}
        act_gates = {1: torch.sigmoid, -1: torch.tanh}

        # num_convs層分のInteraction+Gateを積み重ねる。各層で、スカラー成分と
        # ベクトル/テンソル成分をGateで非線形活性化しながら特徴量を更新していく。
        irreps = self.irreps_node_x
        self.interactions = nn.ModuleList()
        for _ in range(num_convs):
            irreps_scalars = o3.Irreps([(m, ir) for m, ir in self.irreps_hidden
                                         if ir.l == 0 and tp_path_exists(irreps, self.irreps_edge, ir)])
            irreps_gated = o3.Irreps([(m, ir) for m, ir in self.irreps_hidden
                                       if ir.l > 0 and tp_path_exists(irreps, self.irreps_edge, ir)])

            if irreps_gated.dim > 0:
                if tp_path_exists(irreps_node_z, self.irreps_edge, "0e"):
                    ir = "0e"
                elif tp_path_exists(irreps_node_z, self.irreps_edge, "0o"):
                    ir = "0o"
                else:
                    raise ValueError(f"irreps={irreps} times irreps_edge={self.irreps_edge} is unable "
                                      f"to produce gates needed for irreps_gated={irreps_gated}.")
            else:
                ir = None
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()

            gate = Gate(
                irreps_scalars, [act_scalars[ir.p] for _, ir in irreps_scalars],
                irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],
                irreps_gated,
            )
            conv = Interaction(
                irreps_in=irreps, irreps_node=self.irreps_node_z, irreps_edge=self.irreps_edge,
                irreps_out=gate.irreps_in, radial_neurons=radial_neurons, num_neighbors=num_neighbors,
            )
            irreps = gate.irreps_out
            self.interactions.append(Compose(conv, gate))

        self.out = o3.FullyConnectedTensorProduct(
            irreps_in1=irreps, irreps_in2=self.irreps_node_z, irreps_out=self.irreps_out,
        )

        # ここがtest38の要: sigma/time条件付けのための層を追加する。
        # 各畳み込み層のスカラー隠れ次元数(size_embed)に合わせたGaussianBasisEmbeddingで
        # t(=sigma/sigma_max_train)を埋め込み、t_projectionで各ノード特徴量の次元数に線形変換する。
        size_embed = int(str(irreps).split("x")[0])
        self.t_embed = GaussianBasisEmbedding(embedding_dim=size_embed)
        t_embed_dim = self.t_embed.layer2.out_features
        self.t_projection = nn.Linear(t_embed_dim, irreps.dim)

    def forward(self, data, t):
        data = self.init_embed(data)
        edge_index, edge_attr = data.edge_index, data.edge_attr
        h_node_x, h_node_z, h_edge = data.h_node_x, data.h_node_z, data.h_edge

        # スカラー値t(このバッチ全体で1つの値)を埋め込み、全ノードに同じベクトルとして
        # ブロードキャストする。1バッチ=1つのグラフしか正しく条件付けできない点に注意
        # (下のtrain()内のrattle_atのコメント参照)。
        h_node_t = self.t_embed(t)
        h_node_t = h_node_t.expand(h_node_x.shape[0], -1)
        h_node_t = self.t_projection(h_node_t)

        # エッジベクトルを球面調和関数に変換してから、各Interaction+Gate層を通し、
        # 毎層h_node_tを足し込むことでsigma情報を伝え続ける。
        edge_sh = o3.spherical_harmonics(self.irreps_edge, edge_attr, normalize=True, normalization='component')
        for layer in self.interactions:
            h_node_x = layer(h_node_x, h_node_z, edge_index, edge_sh, h_edge)
            h_node_x = h_node_x + h_node_t

        # 最終的に3次元ベクトル(irreps_out='1x1e')、つまり各原子の変位予測dxを出力する。
        return self.out(h_node_x, h_node_z)


class DownselectEdges(BaseTransform):
    """Vendored from DM2/src/graphite/transforms/downselect_edges.py."""
    # graph()はtraining用に少し大きめのlarge_cutoffで候補エッジを作っておき、
    # このDownselectEdgesで実際のモデルcutoff以内のエッジだけに絞り込む。
    # (RattleParticlesでノイズを加えた後、距離が変化してからこの絞り込みを行うことで、
    # ノイズ後もcutoff以内に収まっているエッジだけを使う。)
    def __init__(self, cutoff, cell=None):
        super().__init__()
        self.cutoff = cutoff
        self.cell = cell

    def __call__(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        mask = (edge_attr[:, :3].norm(dim=1) <= self.cutoff)
        data.edge_index = edge_index[:, mask]
        data.edge_attr = edge_attr[mask]
        return data

    def forward(self, data):
        return self.__call__(data)

    def __repr__(self):
        return f'{self.__class__.__name__}(cutoff={self.cutoff})'


class RattleParticles(BaseTransform):
    """Vendored from DM2/src/graphite/transforms/rattle_particles.py. Applies a random
    Gaussian noise to particle positions, with standard deviation drawn uniformly from
    [sigma_min, sigma_max]."""
    # 学習時に「正解の構造」にガウスノイズを加えて壊し(corrupt)、モデルには
    # 「加えられたノイズdxを予測して元に戻す」タスクを学習させる。これが
    # denoising score matching(スコアベース生成モデル)の学習の基本形。
    def __init__(self, sigma_max, sigma_min=0.001):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, data):
        if data.batch is not None:
            # バッチ内のグラフごとに別々のsigmaを[sigma_min, sigma_max]から一様サンプルする
            # (test38ではtrain()側でsigma_min=sigma_maxに固定して呼ぶため、実質バッチ全体で1つのsigmaになる)。
            sigma = torch.empty(data.num_graphs, device=data.pos.device).uniform_(
                self.sigma_min, self.sigma_max)
            sigma = sigma[data.batch, None]
        else:
            sigma = torch.empty(1, device=data.pos.device).uniform_(self.sigma_min, self.sigma_max)

        eps = torch.randn_like(data.pos)  # 標準正規分布ノイズ
        data.dx = sigma * eps  # モデルが予測すべき正解の変位(ノイズそのもの)
        data.pos = data.pos + data.dx  # 座標を実際に壊す

        if data.edge_attr is not None:
            # 座標を動かしたので、既存のエッジベクトル(相対変位)も整合するように更新する。
            i, j = data.edge_index
            data.edge_attr = data.edge_attr + data.dx[j] - data.dx[i]

        data.sigma = sigma  # 後で参照できるように保存(このtest38ではdata.sigmaは直接は使わずtを別途渡す)
        data.eps = eps
        return data

    def forward(self, data):
        return self.__call__(data)


# --- test38-specific code -------------------------------------------------------------

def build_time_model(config: dict, device: torch.device) -> nn.Module:
    # architecture()で作った構成辞書からNequIP_TimeEmbedモデルを組み立てる。
    values = {k: v for k, v in config.items() if k not in ("num_species", "cutoff_angstrom")}
    model = NequIP_TimeEmbed(
        init_embed=InitialEmbedding(config["num_species"], config["cutoff_angstrom"]),
        **values,
    )
    return model.to(device)


def warm_start_from_plain_checkpoint(model: NequIP_TimeEmbed, plain_state_dict: dict) -> None:
    """Copy every weight NequIP_TimeEmbed shares with plain NequIP, then zero the new
    time-projection layer so the warm-started model equals the source checkpoint at
    t=anything until training moves it."""
    # test37(sigma条件付けなしのplain NequIP)で既に学習済みのチェックポイントから
    # 重みを引き継ぐための関数。NequIP_TimeEmbedはplain NequIPと全く同じ
    # Interaction/Gate/出力層を持つので、それらの重みはそのままコピーできる。
    # 新規に追加されたt_embed/t_projection(sigma条件付け用の層)だけは
    # plain側チェックポイントに存在しないので、ここではまだ扱わない。
    own_state = model.state_dict()
    missing = [k for k in own_state if k not in plain_state_dict]
    unexpected = [k for k in plain_state_dict if k not in own_state]
    if unexpected:
        raise ValueError(f"--warm-start checkpoint has unexpected keys for this architecture: {unexpected}")
    if any(not k.startswith(("t_embed.", "t_projection.", "time_scalar_mask")) for k in missing):
        raise ValueError(f"--warm-start checkpoint is missing non-time-conditioning keys: {missing}")
    own_state.update(plain_state_dict)
    model.load_state_dict(own_state)
    # t_projectionの重み・バイアスをゼロにすることで、h_node_t(sigma由来の特徴量)が
    # 各層の出力に何も足さない状態にする。つまりウォームスタート直後は
    # 「sigmaを完全に無視するモデル」= 元のplainチェックポイントと数値的に全く同じ
    # 挙動になり、そこから学習を進めるにつれて徐々にsigma依存性を獲得していく。
    nn.init.zeros_(model.t_projection.weight)
    nn.init.zeros_(model.t_projection.bias)


def checkpoint_time_model(path, device):
    # test50形式のチェックポイントを読み込み、保存されていたarchitectureから
    # モデルを再構築して重みを復元する(生成時に使う)。
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("format") != FORMAT:
        raise ValueError("Expected a test52 sigma-conditioned checkpoint")
    model = build_time_model(ck["architecture"], device)
    model.load_state_dict(ck["model_state_dict"])
    return model.eval(), ck


# --- test50-specific code: build the Si-only SiO2 CG dataset test38's load_dataset() expects ---

def parse_lammpstrj_si_only(path, stride, offset):
    # test52固有(新規): LAMMPSカスタムダンプ(`id type xu yu zu`、非wrap座標)を直接読む
    # 軽量パーサー。ASEを使わない理由: このダンプ形式には原子種・質量情報が一切埋め込まれて
    # おらず、md/nvt_traj_0.lammpstrj先頭フレームの実測(type1が64個、type2が128個/フレーム、
    # SiO2のSi:O=1:2比と一致)から type1=Si(14), type2=O(8) と判明している対応をそのまま
    # 使う(test47/48/49のgenerate.pyのZ_of_type={1:14, 2:8}と同じ規約)。
    # フレーム内の原子順序(id)でソートしてから種でフィルタするので、複数ファイル・複数
    # ストライドをまたいでも常に同じ64個のSiが同じ順序で並ぶ(index-preserving)。
    frames, lengths = [], []
    frame_index = 0
    with Path(path).open() as stream:
        while True:
            line = stream.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            stream.readline()  # timestep値(未使用)
            if not stream.readline().startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError(f"{path}: unexpected dump format (NUMBER OF ATOMS)")
            n_atoms = int(stream.readline())
            if not stream.readline().startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"{path}: unexpected dump format (BOX BOUNDS)")
            box = [hi - lo for lo, hi in (map(float, stream.readline().split()) for _ in range(3))]
            if not stream.readline().startswith("ITEM: ATOMS"):
                raise ValueError(f"{path}: unexpected dump format (ATOMS)")
            keep = frame_index >= offset and (frame_index - offset) % stride == 0
            rows = []
            for _ in range(n_atoms):
                parts = stream.readline().split()
                rows.append((int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
            if keep:
                rows.sort(key=lambda row: row[0])
                si = np.array([(x, y, z) for _, t, x, y, z in rows if t == 1], dtype=np.float32)
                if len(si) == 0:
                    raise ValueError(f"{path}: no type=1 (Si) atoms in frame {frame_index}")
                frames.append(si)
                lengths.append(box)
            frame_index += 1
    if not frames:
        raise ValueError(f"{path}: no frames matched --offset/--stride")
    counts = {len(f) for f in frames}
    if len(counts) != 1:
        raise ValueError(f"{path}: inconsistent Si count across frames: {sorted(counts)}")
    return np.stack(frames), np.asarray(lengths, dtype=np.float64), frame_index


def prepare(args):
    # SiO2 (beta-cristobalite) Si-only CG mapping: keep exactly the central Si of every SiO4
    # tetrahedron (one CG site per original Si atom, index-preserving subset selection), drop
    # every O atom outright. Not a weighted/averaged mapping like test37.py's general prepare()
    # (there is nothing to average -- each retained site already coincides with a real atom).
    if args.trajectories:
        # test52固有(新規): 事前抽出済みの184フレームnpzではなく、生のNVTトラジェクトリから
        # 直接、より密な(=より多い)フレームを取り出す。184フレームは全部「同じ1つの結晶の
        # 熱振動」であり構造的な多様性はないが、生ダンプ(各10001フレーム)から細かい
        # strideで取り出せば、同じ熱アンサンブルのより大きく相関の少ないサンプルが作れる。
        si_frames, lengths, dumped = [], [], 0
        for path in args.trajectories:
            frames_i, lengths_i, dumped_i = parse_lammpstrj_si_only(path, args.stride, args.offset)
            si_frames.append(frames_i)
            lengths.append(lengths_i)
            dumped += dumped_i
            print(f"  {path}: {len(frames_i)} frames kept (of {dumped_i} dumped)", flush=True)
        si_positions = np.concatenate(si_frames, axis=0)
        lengths = np.concatenate(lengths, axis=0)
        si_count = si_positions.shape[1]
        total_atoms = None  # 生ダンプにはO原子数の直接の記録がないので不明(参考値として省略)
        source_desc = [str(p.resolve()) for p in args.trajectories]
        source_meta = dict(offset=args.offset)
    else:
        archive = np.load(args.reference_frames)
        for key in ("positions", "cell_lengths", "numbers"):
            if key not in archive.files:
                raise ValueError(f"--reference-frames is missing array '{key}'")
        positions, cell_lengths, numbers = archive["positions"], archive["cell_lengths"], archive["numbers"]
        if positions.ndim != 3 or positions.shape[2] != 3 or positions.shape[0] != cell_lengths.shape[0] \
                or cell_lengths.shape[1] != 3 or positions.shape[1] != numbers.shape[0]:
            raise ValueError("Expected positions (frames,atoms,3), cell_lengths (frames,3), "
                              "numbers (atoms,) arrays of agreeing shape")
        si_indices = np.flatnonzero(numbers == 14)
        if len(si_indices) == 0:
            raise ValueError("No Si (atomic number 14) atoms found in --reference-frames")
        frames = positions[::args.stride]
        lengths = cell_lengths[::args.stride]
        si_positions = np.ascontiguousarray(frames[:, si_indices, :]).astype(np.float32)
        si_count = len(si_indices)
        total_atoms = int(numbers.shape[0])
        source_desc = str(args.reference_frames.resolve())
        source_meta_path = args.reference_frames.with_name(args.reference_frames.stem + "_metadata.json")
        source_meta = json.loads(source_meta_path.read_text()) if source_meta_path.is_file() else None

    if len(si_positions) < 2:
        raise ValueError("Need at least two frames after striding")
    if not np.allclose(lengths, lengths[:, :1]):
        raise ValueError("test52 assumes a cubic cell (Lx=Ly=Lz) at every frame")

    replicate = args.replicate
    if replicate > 1:
        # 箱をreplicate^3倍にタイル化する: cutoffを安全に広げたい(半箱を大きくしたい)が
        # 新規にその箱サイズでMDを回していない場合の、原子配置の人工的な拡大手段。
        # 各セルの周期像を実際の原子として複製するだけなので、半箱は必ずreplicate倍になり、
        # cutoff/large_cutoffの安全上限(半箱未満、下のtrain()のガードを参照)もreplicate倍
        # まで安全に引き上げられる。
        # 注意(重要な限界): 複製された像は元の配置を厳密にコピーしただけで、独立した
        # 熱ゆらぎを持つ別配置ではない。本物により大きな箱でMDを回した場合と違い、
        # replicate^3個の像は互いに完全に相関している(人工的な並進対称性を持つ)。
        # cutoff拡大の効果を見るための近似的な手段であり、本物の大箱MDの代用にはならない。
        n = replicate
        shifts = np.array([(i, j, k) for i in range(n) for j in range(n) for k in range(n)],
                           dtype=np.float32)  # (n^3, 3)
        old_cell_length = lengths[:, 0].astype(np.float32)  # (frames,)
        # (frames, n_si, 3) + (n^3, 1, 3)*(frames,1,1,1) -> (frames, n^3, n_si, 3)
        tiled = si_positions[:, None, :, :] + shifts[None, :, None, :] * old_cell_length[:, None, None, None]
        si_positions = np.ascontiguousarray(tiled.reshape(len(si_positions), -1, 3)).astype(np.float32)
        lengths = lengths * n

    n_frames = si_positions.shape[0]
    n_sites = si_positions.shape[1]
    cells = np.zeros((n_frames, 3, 3), dtype=np.float64)
    for axis in range(3):
        cells[:, axis, axis] = lengths[:, 0].astype(np.float64)

    output = new_output(args.output)
    np.save(output / "positions.npy", si_positions)
    np.save(output / "cells.npy", cells)
    species = [{"name": "Si", "atomic_number": 14, "mass_amu": 28.0855}]
    type_ids = [0] * n_sites

    meta = dict(
        format=DATASET_FORMAT, length_unit="angstrom", frames=n_frames, species=species,
        type_ids=type_ids, condition={"temperature_k": args.temperature_k},
        cg_mapping=("test52 Si-only: one CG site per original Si atom (index-preserving "
                    "subset), every O atom of every SiO4 tetrahedron dropped outright "
                    "(not averaged/merged)"),
        original_atom_count=total_atoms, si_atom_count=si_count,
        replicate=replicate, replicated_site_count=n_sites,
        replicate_caveat=(None if replicate == 1 else
            f"positions are tiled {replicate}x{replicate}x{replicate} ({replicate**3} exact "
            "periodic copies of the same configuration per frame) to safely enlarge the box for "
            "a bigger --cutoff, NOT an independent larger-box MD run -- the copies are perfectly "
            "correlated with each other, unlike real thermal disorder at that box size."),
        source=source_desc, source_metadata=source_meta,
        source_stride=args.stride, source_offset=(args.offset if args.trajectories else None),
        scientific_caveat=CAVEAT,
        sha256={name: digest(output / name) for name in ("positions.npy", "cells.npy")},
    )
    save_json(output / "metadata.json", meta)
    print(f"Prepared {n_frames} frames, {n_sites} Si CG sites"
          f"{f', replicated {replicate}x{replicate}x{replicate}' if replicate > 1 else ''}: {output}")


# --- test52-specific code: periodic (torus) noise, vendored from test39.py's own real-space
# (Angstrom) wrapped_score_target, unchanged in spirit -------------------------------------------

def wrapped_score_target(noisy, clean, lengths, sigma):
    """Return -sigma * grad log p_sigma(noisy | clean) for Brownian motion on a torus.

    Vendored from test39.py (same function, same name) -- see that file's own comment for why
    image sums are used at small sigma and a Fourier heat-kernel series at large sigma (the two
    overlap near 0.2 L). Each Cartesian dimension factorizes for an orthorhombic cell.
    """
    length = torch.as_tensor(lengths, dtype=noisy.dtype, device=noisy.device)
    delta = torch.remainder(noisy - clean + length / 2, length) - length / 2
    result = torch.empty_like(delta)
    for axis in range(3):
        side = length[axis]
        d = delta[:, axis:axis + 1]
        if sigma / float(side) < 0.2:
            n = torch.arange(-1, 2, dtype=noisy.dtype, device=noisy.device)
            images = d + n * side
            weights = torch.softmax(-0.5 * (images / sigma).square(), dim=-1)
            result[:, axis] = (weights * images).sum(dim=-1) / sigma
        else:
            k = torch.arange(1, 13, dtype=noisy.dtype, device=noisy.device)
            amplitude = torch.exp(-2 * math.pi**2 * k.square() * (sigma / side) ** 2)
            angle = 2 * math.pi * d * k / side
            density = 1 + 2 * (amplitude * torch.cos(angle)).sum(dim=-1)
            derivative = -(4 * math.pi / side) * (
                k * amplitude * torch.sin(angle)).sum(dim=-1)
            result[:, axis] = -sigma * derivative / density.clamp_min(1e-8)
    return result


def sigma_max_for(cells, requested):
    # 要求されたsigma-max(未指定ならNone)を検証、またはセルの最長辺から自動決定する。
    # test39と同じ基準: 終端(sigma=sigma_max)でのフーリエ残差が十分小さくないと、
    # 「セル内一様分布」という前提(--init randomや理論上の終端分布)が成り立たない。
    box_length = float(np.asarray(cells)[:, [0, 1, 2], [0, 1, 2]].max())
    sigma_max = float(requested) if requested else box_length
    residual = math.exp(-2 * math.pi**2 * (sigma_max / box_length) ** 2)
    if residual > 1e-5:
        minimum = math.sqrt(-math.log(1e-5) / (2 * math.pi**2)) * box_length
        raise ValueError(
            f"--sigma-max ({sigma_max:.4g}) is too small relative to the box ({box_length:.4g} A) "
            f"for the terminal distribution to be near-uniform (Fourier residual={residual:.3g}); "
            f"use --sigma-max >= {minimum:.4g}, or omit --sigma-max to default to the box's own "
            f"longest side ({box_length:.4g} A), matching test39's own convention.")
    return sigma_max


def train(args):
    # --- 入力チェックとセットアップ ---
    positions, cells, meta = load_dataset(args.dataset)
    if len(positions) < 3:
        raise ValueError("Training requires at least three frames")
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("validation-fraction must be between zero and 0.5")
    if args.large_cutoff < args.cutoff:
        raise ValueError("Require large-cutoff >= cutoff")
    # test52固有(test39から): sigma-maxがセルの大きさに対して十分大きくないと、終端分布が
    # セル内一様分布に近づかない(周期ノイズなので、test50のsigma-max=1.5のような「局所的な
    # 揺らぎ」のスケールでは全く足りない -- 未指定ならセルの最長辺そのものをデフォルトにする、
    # test39と同じ規約)。
    sigma_max = sigma_max_for(cells, args.sigma_max)
    # test50固有のガード(test38にはない): cutoff/large_cutoffが半箱以上だと、同じ原子対が
    # 2つ以上の周期像を通じて二重に繋がってしまう(test33/48で文書化された周期像重複バグ)。
    # このバグは黙って学習データを壊すだけで例外を出さないので、ここで明示的に弾く。
    # (--replicateでタイル化した大きい箱のデータセットなら、より大きいcutoffも安全に通る。)
    half_box = float(np.asarray(cells)[:, [0, 1, 2], [0, 1, 2]].min()) / 2
    if args.large_cutoff >= half_box:
        raise ValueError(
            f"--large-cutoff ({args.large_cutoff}) must be strictly less than half the box "
            f"({half_box:.4f} A) -- otherwise the same atom pair gets connected through 2+ "
            f"periodic images at once (the test33/48 duplicate-periodic-image bug). Use a smaller "
            f"cutoff, or rebuild the dataset with `prepare --replicate N` to enlarge the box.")
    device = device_for(args.device)
    output = args.output.resolve() if args.resume else new_output(args.output)
    checkpoint = output / "checkpoint.pt"
    if args.resume and not checkpoint.is_file():
        raise ValueError("--resume requires output/checkpoint.pt")

    # フレームを学習用/検証用に分割(先頭split個が学習、残りが検証)。
    split = max(1, int(len(positions) * (1 - args.validation_fraction)))
    # この実行の設定をすべて記録しておく。--resumeで再開する際、設定が
    # 完全一致するかどうかの検証にも使う(途中でハイパラを変えて再開させない)。
    settings = dict(
        dataset_sha256=meta["sha256"], metadata_sha256=digest(args.dataset / "metadata.json"),
        cutoff=args.cutoff, large_cutoff=args.large_cutoff,
        irreps_hidden=args.irreps_hidden, irreps_edge=args.irreps_edge,
        sigma_min=args.sigma_min, sigma_max=sigma_max,  # 解決済みの値(--sigma-max未指定ならセル最長辺)
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        seed=args.seed, split_frame=split, device=str(device), log_every=args.log_every,
        warm_start_sha256=(digest(args.warm_start) if args.warm_start else None),
    )
    config = architecture(len(meta["species"]), args.cutoff, args.irreps_hidden, args.irreps_edge)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = build_time_model(config, device)
    if not args.resume and args.warm_start is not None:
        # test37/36で学習済みのplainチェックポイントからウォームスタートする場合。
        # アーキテクチャ(species数・cutoffなど)が一致していることを確認してから、
        # 共有できる重みだけをコピーする。
        source_ck = torch.load(args.warm_start, map_location="cpu", weights_only=False)
        if source_ck.get("architecture") != config:
            raise ValueError("--warm-start checkpoint architecture does not match this dataset")
        warm_start_from_plain_checkpoint(model, source_ck["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history, completed = [], 0
    if args.resume:
        # 既存のtest52チェックポイントから学習状態(重み・optimizer・乱数状態・履歴)を復元する。
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if ck.get("format") != FORMAT or ck["settings"] != settings:
            raise ValueError("Resume settings/data/device differ from checkpoint")
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer"])
        restore_rng(ck["rng"])
        history, completed = ck["history"], ck["completed_updates"]
    # test52固有: RattleParticles/DownselectEdgesの代わりに、周期(トーラス)ノイズで
    # ノイズを加えた「後の」座標から毎回グラフを新規に組み立てる(large_cutoffで広めに
    # 候補エッジを作っておいて後で絞り込む、というtest50のトリックは、周期ラップ後は
    # 境界をまたいで近傍関係が不連続に変わりうるため使えない)。NequIP_TimeEmbedがバッチ
    # 全体で1つのtしか受け付けない事情はtest50と同じなので、バッチ内の全構造に同じ
    # sigma_valueを使う。
    def noisy_batch(indices, sigma_value):
        examples, targets = [], []
        for i in indices:
            cell_diag = torch.tensor(np.diag(cells[i]), dtype=torch.float32, device=device)
            clean = torch.tensor(np.asarray(positions[i]).copy(), dtype=torch.float32, device=device)
            noisy = torch.remainder(clean + sigma_value * torch.randn_like(clean), cell_diag)
            examples.append(graph(noisy.detach().cpu().numpy(), cells[i], meta["type_ids"],
                                   args.cutoff, device))
            targets.append(wrapped_score_target(noisy, clean, cell_diag, sigma_value))
        return Batch.from_data_list(examples), torch.cat(targets, dim=0)
    deadline = time.monotonic() + args.time_budget_hours * 3600  # HPCジョブの時間切れ前に安全に停止するための締切

    def save():
        # 学習の途中経過(重み・optimizer状態・乱数状態・履歴)をチェックポイントに、
        # 進捗の要約をtraining.jsonに、それぞれ書き出す。
        save_checkpoint(checkpoint, dict(
            format=FORMAT, architecture=config, settings=settings,
            model_state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
            optimizer=optimizer.state_dict(), rng=rng_state(), history=history,
            completed_updates=completed, requested_updates=args.updates,
            dataset_metadata=meta, large_cutoff=args.large_cutoff,
            start_positions_angstrom=np.asarray(positions[0]).copy(),
            cell_angstrom=np.asarray(cells[0]).copy(), scientific_caveat=CAVEAT,
        ))
        save_json(output / "training.json", dict(
            completed_updates=completed, requested_updates=args.updates,
            history=history, scientific_caveat=CAVEAT,
        ))

    print(f"train (test52, sigma-conditioned, periodic score): device={device}, "
          f"frames={len(positions)}, train/validation={split}/{len(positions) - split}, "
          f"sigma_max={sigma_max:.4g} A, warm_start={'yes' if args.warm_start else 'no'}", flush=True)
    # --- 学習ループ本体 ---
    for step in range(completed + 1, args.updates + 1):
        if STOP or time.monotonic() >= deadline:
            # 中断シグナルか時間切れなら、その場でチェックポイントを保存して終了コード75を返す
            # (HPCジョブスケジューラに「再投入すれば続きから再開できる」ことを伝える慣習的な値)。
            save()
            print("Training paused; resume with --resume", flush=True)
            return 75
        model.train()
        # 学習フレームからランダムにbatch_size個選び、
        indices = np.random.randint(split, size=args.batch_size)
        # このステップで使うsigmaを[sigma_min, sigma_max]から1つだけサンプルする
        # (バッチ内の全グラフに同じsigmaを使う。理由は上のnoisy_batch()のコメント参照)。
        sigma_value = float(np.random.uniform(args.sigma_min, sigma_max))
        batch, target = noisy_batch(indices, sigma_value)
        # モデルに渡すt(正規化されたsigma)を計算し、順伝播・損失計算・逆伝播。
        t = torch.tensor([sigma_value / sigma_max], device=device, dtype=batch.pos.dtype)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch, t)
        # 目的関数: 周期的な条件付きスコア(wrapped_score_target)をどれだけ正確に
        # 予測できたかのMSE(denoising score matching、test38/50の生dxターゲットとは異なる)。
        loss = torch.nn.functional.mse_loss(prediction, target)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        completed = step
        if step == 1 or step % args.log_every == 0 or step == args.updates:
            # 定期的に検証データでの損失もログに出す。乱数状態を退避・復元することで、
            # 検証評価が学習側の乱数列(データ選択やノイズ)に影響を与えないようにしている。
            model.eval()
            saved_rng = rng_state()
            losses = []
            valid_rng = np.random.default_rng(args.seed + 1)
            for i in range(split, min(split + 4, len(positions))):
                valid_sigma = float(valid_rng.uniform(args.sigma_min, sigma_max))
                valid_batch, valid_target = noisy_batch([i], valid_sigma)
                valid_t = torch.tensor([valid_sigma / sigma_max], device=device, dtype=valid_batch.pos.dtype)
                with torch.no_grad():
                    losses.append(torch.nn.functional.mse_loss(
                        model(valid_batch, valid_t), valid_target
                    ).item())
            restore_rng(saved_rng)
            val_loss = float(np.mean(losses))
            row = dict(step=step, train_mse_A2=float(loss.detach().cpu()), valid_mse_A2=val_loss)
            history.append(row)
            print(json.dumps(row), flush=True)
        if step % args.checkpoint_every == 0:
            save()
    save()
    print(f"Checkpoint: {checkpoint}")


@torch.no_grad()
def generate(args):
    # --- セットアップ: チェックポイント読み込みと出力先準備 ---
    device = device_for(args.device)
    model, ck = checkpoint_time_model(args.checkpoint, device)
    meta, cell = ck["dataset_metadata"], ck["cell_angstrom"]
    cutoff = ck["architecture"]["cutoff_angstrom"]
    sigma_max_train = ck["settings"]["sigma_max"]  # tの正規化に使う、学習時のsigma_max(解決済みの値)
    cell_diag = torch.tensor(np.diag(np.asarray(cell)), dtype=torch.float32, device=device)
    # --start-sigmaは未指定なら学習時のsigma_max(=このチェックポイントが実際にカバーした
    # 最大ノイズ)をそのまま使う(test39は分離したstart_sigma自体を持たないが、test50由来の
    # 「途中のノイズレベルから開始できる」柔軟性は残す)。
    start_sigma = float(args.start_sigma) if args.start_sigma is not None else sigma_max_train
    if start_sigma < args.sigma_min:
        raise ValueError("start-sigma must be at least sigma-min")
    output = args.output.resolve() if args.resume else new_output(args.output)
    settings = dict(
        checkpoint_sha256=digest(args.checkpoint), reverse_steps=args.reverse_steps,
        deterministic_steps=args.deterministic_steps, start_sigma=start_sigma,
        sigma_min=args.sigma_min, thermal_scale=args.thermal_scale, init=args.init,
        seed=args.seed, device=str(device), cutoff_angstrom=cutoff,
    )
    total = args.reverse_steps
    state_path = output / "generation_restart.pt"
    if args.resume:
        # 中断していた生成を再開する場合: 直前の座標・乱数状態を復元し、
        # 既存のpositions.npyメモリマップを追記モードで開く。
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["settings"] != settings:
            raise ValueError("Generation resume settings differ; use a new output directory")
        pos, completed = state["positions"].to(device), state["step"]
        restore_rng(state["rng"])
        trajectory = np.lib.format.open_memmap(output / "positions.npy", mode="r+")
    else:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if args.init == "random":
            # 周期ノイズなら「セル内一様分布」は理論的にもtrain()のsigma-max検証済みの
            # 終端分布そのものなので、そこから直接始めるのは正しく定義された操作になる
            # (test50/51の非周期ノイズでは、これは検証されていない仮定だった)。
            n_atoms = len(ck["start_positions_angstrom"])
            pos = torch.rand((n_atoms, 3), dtype=torch.float32, device=device) * cell_diag
        else:
            # 学習データセットの最初のフレーム(訓練時に保存しておいたstart_positions_angstrom)
            # を出発点にする。
            pos = torch.tensor(ck["start_positions_angstrom"], dtype=torch.float32, device=device)
            if args.init == "crystal-noised":
                # 学習時と全く同じ周期ノイズモデル(pos = (pos + start_sigma*eps) mod cell)を
                # 明示的に一度適用してから焼きなましを始める。これで「拡散過程で実際に
                # sigma=start_sigmaまで拡散させたノイズ構造」から出発する、本来の意味での
                # reverse diffusionになる(test50由来の--init crystal-noisedと同じ動機)。
                pos = torch.remainder(pos + start_sigma * torch.randn_like(pos), cell_diag)
        completed = 0
        trajectory = np.lib.format.open_memmap(
            output / "positions.npy", mode="w+", dtype=np.float32, shape=(total + 1, len(pos), 3)
        )
        trajectory[0] = pos.cpu().numpy()

    # --- sigmaスケジュールの構築(test39と同じ: start_sigmaからsigma_minまで対数等分割) ---
    first_sigma = max(start_sigma, args.sigma_min)
    if args.reverse_steps == 1:
        sigma_schedule = torch.tensor([first_sigma, 0.0], device=device, dtype=torch.float64)
    else:
        positive_levels = torch.exp(torch.linspace(
            math.log(first_sigma), math.log(args.sigma_min), args.reverse_steps,
            device=device, dtype=torch.float64,
        ))
        sigma_schedule = torch.cat((positive_levels, positive_levels.new_zeros(1)))
    # 全reverse_stepsのうち、後半deterministic_steps回はノイズ注入を止め、ドリフトの
    # 強さも半分にする(test39の設計そのまま。test38/50のDDIM風更新(1 - next_sigma/sigma)
    # とは式が異なる -- 予測がdx自体ではなく-sigma*scoreになったことに整合させるため)。
    stochastic_steps = args.reverse_steps - args.deterministic_steps
    if stochastic_steps <= 0:
        print(f"NOTE: --deterministic-steps ({args.deterministic_steps}) >= --reverse-steps "
              f"({args.reverse_steps}): every one of the {args.reverse_steps} steps will run "
              f"the deterministic (half-strength, no-noise) branch. This does NOT make the "
              f"schedule finer or longer -- to do that, increase --reverse-steps itself.",
              flush=True)
    deadline = time.monotonic() + args.time_budget_hours * 3600

    def save():
        # 生成中の座標(positions.npy)と再開用チェックポイント、進捗JSONを保存する。
        trajectory.flush()
        save_checkpoint(state_path, dict(
            settings=settings, step=completed, positions=pos.detach().cpu(), rng=rng_state(),
        ))
        save_json(output / "generation.json", dict(
            completed_steps=completed, requested_steps=total, valid_frames=completed + 1,
            complete=completed == total, length_unit="angstrom", settings=settings,
            cell_angstrom=np.asarray(cell).tolist(), dataset_metadata=meta,
            is_equilibrium_trajectory=False, scientific_caveat=CAVEAT,
        ))

    # --- 生成(逆拡散)ループ本体 ---
    for step in range(completed, total):
        if STOP or time.monotonic() >= deadline:
            save()
            print("Generation paused; resume with --resume", flush=True)
            return 75
        sigma, next_sigma = float(sigma_schedule[step]), float(sigma_schedule[step + 1])
        # 現在の座標からグラフを再構築し、モデルにt=sigma/sigma_max_trainを渡して
        # 「-sigma*スコア」を予測させる(test38/50の生dx予測とは異なる目的関数)。
        data = graph(pos.cpu().numpy(), cell, meta["type_ids"], cutoff, device)
        t = torch.full((1,), sigma / sigma_max_train, device=device, dtype=pos.dtype)
        scaled_negative_score = model(data, t)
        # Reverse SDE(test39と同じ式): drift = variance_drop * score = -variance_drop/sigma *
        # scaled_negative_score。決定論的フェーズ(仕上げ)はドリフトを半分にし、ノイズ注入を
        # 止める(test39のmultiplier=0.5の設計をそのまま踏襲)。
        variance_drop = max(sigma**2 - next_sigma**2, 0.0)
        multiplier = 0.5 if step >= stochastic_steps else 1.0
        pos = pos - multiplier * (variance_drop / sigma) * scaled_negative_score
        if multiplier == 1.0:
            pos = pos + math.sqrt(variance_drop) * torch.randn_like(pos) * args.thermal_scale
        pos = torch.remainder(pos, cell_diag)  # 周期ノイズなので毎ステップ必ずセル内に折り返す
        if not torch.isfinite(pos).all():
            raise RuntimeError("Non-finite generated positions")
        completed = step + 1
        trajectory[completed] = pos.cpu().numpy()
        if completed % args.checkpoint_every == 0:
            save()
            print(f"generation {completed}/{total} (sigma={sigma:.4f})", flush=True)
    save()
    # 生成が完了したら、最終フレームをase.Atomsに変換してextxyzファイルとしても書き出す。
    atoms = atoms_from_meta(pos.cpu().numpy(), cell, meta)
    atoms.wrap()
    ase.io.write(output / "final.extxyz", atoms)
    if args.trajectory_stride > 0:
        # --trajectory-stride>0なら、結晶構造がステップを追って組み上がっていく様子を
        # 見られるよう、positions.npy(全ステップの座標)をマルチフレームのextxyzに
        # 書き出す(可視化ソフト(OVITO/VMD等)でそのままアニメーション再生できる)。
        write_trajectory_extxyz(output, meta, cell, args.trajectory_stride)
    print(f"Generated {total + 1} frames: {output}")


def write_trajectory_extxyz(output, meta, cell, stride):
    # positions.npy(generate()が保存した全ステップの座標)を読み直し、strideおきの
    # フレーム(+最終フレームは必ず含める)を1つのマルチフレームextxyzに書き出す。
    # 各フレームのase.Atoms.infoに"step"を残しておくので、可視化時にステップ番号が分かる。
    trajectory = np.load(output / "positions.npy", mmap_mode="r")
    path = output / "trajectory.extxyz"
    if path.exists():
        path.unlink()
    steps = list(range(0, len(trajectory), stride))
    if steps[-1] != len(trajectory) - 1:
        steps.append(len(trajectory) - 1)
    for step in steps:
        atoms = atoms_from_meta(np.asarray(trajectory[step]), cell, meta)
        atoms.wrap()
        atoms.info["step"] = step
        ase.io.write(path, atoms, append=True)
    print(f"Trajectory ({len(steps)} frames, stride={stride}): {path}")


def parser():
    # コマンドラインインターフェース定義: "prepare"・"train"・"generate"の3つのサブコマンドを持つ。
    # ("prepare"はtest50独自の追加。train/generateはtest38.pyと同一。)
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("prepare", help="SiO2 Si-only CG dataset: keep one CG site per original Si atom, drop every O")
    p.add_argument("--reference-frames", type=Path, default=ROOT / "simu_data" / "reference_frames.npz",
                   help="npz with 'positions' (frames,atoms,3), 'cell_lengths' (frames,3), "
                        "'numbers' (atoms,) arrays in Angstrom; defaults to the bundled "
                        "simu_data/reference_frames.npz (test47/48/49's own beta-cristobalite "
                        "NVT reference, 184 frames). Ignored if --trajectories is given.")
    p.add_argument("--trajectories", type=Path, nargs="+", default=None,
                   help="test52 addition: one or more raw md/nvt_traj_*.lammpstrj dumps (NOT "
                        "bundled in this repo -- point this at your own checkout/copy of the "
                        "main ScoreMD repo's md/ directory) to draw far more, denser frames "
                        "directly from the MD trajectory than the bundled 184-frame npz has. "
                        "Still all thermal snapshots of the SAME one crystal, not new "
                        "structural diversity -- see test52.py's own module docstring.")
    p.add_argument("--offset", type=nonnegative_count, default=1000,
                   help="With --trajectories: skip this many dumped frames as burn-in before "
                        "applying --stride (matches the original 184-frame extraction's own "
                        "index 1000::200 convention).")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--temperature-k", type=positive, default=300.0)
    p.add_argument("--stride", type=count, default=1,
                   help="Without --trajectories: keep every Nth frame of the reference npz "
                        "(unchanged test50/51 behavior). With --trajectories: keep every Nth "
                        "DUMPED frame after --offset (e.g. --stride 20 on nvt_traj_0..3, each "
                        "10001 dumped frames, offset 1000 -> ~450 kept frames/file, ~1800 total "
                        "vs. the bundled npz's 184).")
    p.add_argument("--replicate", type=count, default=1,
                   help="Tile the box N x N x N (exact periodic copies, not independent thermal "
                        "samples) to safely enlarge it for a bigger --cutoff at train time -- e.g. "
                        "--replicate 2 turns the 13.573 A box into 27.146 A (half-box 13.573 A), "
                        "safely covering --cutoff 8. Default 1 (no tiling, unchanged behavior).")
    p.set_defaults(handler=prepare)

    p = sub.add_parser("train", help="sigma-conditioned NequIP_TimeEmbed + periodic (torus) noise + wrapped-score MSE")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--warm-start", type=Path, default=None,
                   help="Plain (non-time-conditioned) NequIP checkpoint, same architecture as "
                        "this dataset, to initialize shared weights from")
    p.add_argument("--updates", type=count, default=50000)  # 勾配更新の総回数(元は6000→20000→50000と要望に合わせて引き上げ)
    p.add_argument("--batch-size", type=count, default=16)
    p.add_argument("--learning-rate", type=positive, default=2.e-4)
    # test38の元のデフォルトは10.0(粘土系の大きな箱用)。このSiO2結晶の素の箱は13.573A
    # (半箱~6.7865A)しかなく、それを超えるcutoffは周期像重複バグ(test33/48)を踏むため、
    # 以前は安全な範囲内で6.5/6.7にとどめていた。今回cutoff=8の要望に対応するため、
    # `prepare --replicate 2`で箱を2x2x2タイル化する前提に変更: 箱が27.146A(半箱13.573A)
    # になるので、cutoff=8/large-cutoff=8.2は余裕を持って安全(半箱まで5.57A以上の余裕)。
    # train()側にもガードを追加済みで、--replicateしていない小さい箱のデータセットに
    # このデフォルトをうっかり使うと、黙って壊れる代わりに明示的なエラーで弾かれる。
    p.add_argument("--cutoff", type=positive, default=8.0)  # モデルが実際に使うグラフcutoff(--replicate 2の箱が前提)
    # test52では周期ノイズ後の座標から毎回グラフを新規に組み立てる(noisy_batch()参照)ので、
    # large_cutoffはもう「ノイズ前に広めの候補エッジを作っておく」役割を持たない。cutoffとの
    # 大小関係チェック(下のtrain()内)とチェックポイントのsettings欄との互換性のためだけに
    # CLI引数として残してある(test50からのインターフェース継続性)。
    p.add_argument("--large-cutoff", type=positive, default=8.2)
    p.add_argument("--sigma-min", type=positive, default=0.001)
    # test52固有: 周期ノイズでは「セル内一様分布に十分近い終端分布」を得るのに必要な
    # sigma-maxが箱のサイズそのもので決まる(train()のsigma_max_for()参照)。test50の
    # sigma-max=1.5(局所的な揺らぎのスケール)は、周期ノイズの下ではこの基準を満たさない
    # (例: 箱13.573Aに対し必要な最小値は~10.4A)。そのためデフォルトをNone(未指定)に変更し、
    # 未指定時はセルの最長辺をそのまま使う(test39と同じ規約)。
    p.add_argument("--sigma-max", type=positive, default=None)
    # test50独自の追加(test38にはこの2つのCLI引数はなく、architecture()内にl<=1隠れ層/
    # l<=2エッジでハードコードされている)。要望により、l=4まで広げた構成
    # (test47/48のl<=5構成を1段階切り詰めたもの)をデフォルトに変更 -- 追加のフラグなしで
    # l=4になる。l_maxを上げるほどe3nnのテンソル積のパス数・中間テンソルが急増しGPU
    # メモリを多く使う(test49のCUDA OOMと同じ理由)ので、OOMが出たら--batch-sizeを
    # 下げること(元のtest38相当のl<=1/l<=2に戻したい場合は明示的に
    # --irreps-hidden "64x0e + 32x1e" --irreps-edge "4x0e + 4x1e + 2x2e" を渡す)。
    p.add_argument("--irreps-hidden", type=str, default="64x0e + 32x1e + 16x2e + 8x3e + 4x4e")
    p.add_argument("--irreps-edge", type=str, default="4x0e + 4x1e + 2x2e + 2x3e + 1x4e")
    p.add_argument("--validation-fraction", type=positive, default=0.1)
    p.add_argument("--log-every", type=count, default=100)
    p.set_defaults(handler=train)

    p = sub.add_parser("generate", help="periodic (torus) reverse variance-exploding SDE sampler with a half-strength, no-noise polish tail")
    p.add_argument("--init", choices=("crystal", "crystal-noised", "random"), default="crystal",
                   help="'crystal' (default): start from the clean training frame and let the "
                        "loop's own noise injection be the only corruption applied. "
                        "'crystal-noised': explicitly corrupt the clean frame with the SAME "
                        "periodic noise model training uses (pos = (pos + start_sigma*randn) mod "
                        "cell) before the reverse loop starts. 'random': atoms placed "
                        "independently uniformly at random in the cell -- with periodic noise "
                        "(unlike test50/51's real-space noise) this is now provably the correct "
                        "terminal distribution at sigma=sigma_max (train()'s own Fourier-residual "
                        "check enforces this at training time), so this is a well-defined "
                        "'generate from nothing' test, not an open question.")
    p.add_argument("--reverse-steps", type=count, default=300)  # sigmaスケジュールの段数(細かいほど1歩あたりの補正が小さくなる)
    p.add_argument("--deterministic-steps", type=nonnegative_count, default=30)  # 末尾何ステップをノイズなし半強度更新にするか
    # test50は0.75(局所的な揺らぎのスケール)がデフォルトだったが、周期ノイズではそれでは
    # 終端分布がセル内一様に程遠い(train()と同じ理由)。未指定時はチェックポイントの
    # 学習時sigma_max(=そのチェックポイントが実際に検証済みの最大ノイズ)をそのまま使う。
    p.add_argument("--start-sigma", type=positive, default=None)
    p.add_argument("--sigma-min", type=positive, default=0.001)  # 確率的ステップを終える下限ノイズレベル
    p.add_argument("--thermal-scale", type=positive, default=1.0)  # 注入ノイズの強さを調整する倍率
    p.add_argument("--trajectory-stride", type=nonnegative_count, default=1,
                   help="Write every Nth generation step (plus the final step) to "
                        "output/trajectory.extxyz as one multi-frame trajectory, so the "
                        "structure's step-by-step formation can be viewed (e.g. in OVITO/VMD). "
                        "0 disables this and only writes final.extxyz.")
    p.set_defaults(handler=generate)

    # train/generate共通の引数(出力先・デバイス・乱数シード・時間予算・再開オプションなど)。
    # ("prepare"はここでは扱わない。--outputは既にprepare自身のサブパーサーで定義済みで、
    #  device/seed/time-budget/resumeのような概念はデータセット準備には存在しないため。)
    for name, p in sub.choices.items():
        if name == "prepare":
            continue
        p.add_argument("--output", type=Path, required=True)
        if name == "generate":
            p.add_argument("--checkpoint", type=Path, required=True)
        p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
        p.add_argument("--seed", type=int, default=1337)
        p.add_argument("--time-budget-hours", type=positive, default=11.5)  # この時間を超えたら自動で中断・保存する(HPCジョブの壁時計制限対策)
        p.add_argument("--resume", action="store_true")
        p.add_argument("--checkpoint-every", type=count, default=25)
    return root


def main():
    # SIGTERM/SIGINTを受けたらrequest_stop()でSTOPフラグを立てるようにしてから、
    # 指定されたサブコマンド(train/generate)のハンドラを実行する。
    import signal
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    args = parser().parse_args()
    print(CAVEAT, flush=True)
    return args.handler(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
