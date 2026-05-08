"""
Phase 3: PPO RL — MaskablePPO (stable-baselines3 + sb3-contrib)
================================================================
Starts from the SL checkpoint and improves via PPO with action masking.

Install:
    pip install stable-baselines3 sb3-contrib gymnasium

Usage:
    python train_rl.py --sl model_sl.pt --out model_rl.zip
    python train_rl.py --sl model_sl.pt --out model_rl.zip --iters 200 --n-envs 8

After training, export value-head to ONNX (Go-compatible):
    python train_rl.py --sl model_sl.pt --out model_rl.zip --export model_rl.onnx
"""

import argparse
import os
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# ── Constants ──────────────────────────────────────────────────────────────────
ROWS, COLS = 10, 17
NCELLS     = ROWS * COLS      # 170
MAX_SCORE  = float(NCELLS)    # 170.0
MAX_SUM    = NCELLS * 9.0     # 1530.0
N_RECTS    = 8415             # all valid (r1,c1,r2,c2) combos

# ── Pre-build rectangle index arrays (module-level, built once) ────────────────

def _build_rect_arrays():
    r1s, c1s, r2s, c2s = [], [], [], []
    for r1 in range(ROWS):
        for r2 in range(r1, ROWS):
            for c1 in range(COLS):
                for c2 in range(c1, COLS):
                    r1s.append(r1); c1s.append(c1)
                    r2s.append(r2); c2s.append(c2)
    return (np.array(r1s, dtype=np.int32), np.array(c1s, dtype=np.int32),
            np.array(r2s, dtype=np.int32), np.array(c2s, dtype=np.int32))

_R1, _C1, _R2, _C2 = _build_rect_arrays()   # each shape (8415,)


# ── Vectorised action mask ─────────────────────────────────────────────────────

def compute_mask(board: np.ndarray) -> np.ndarray:
    """
    board: (10, 17) int / uint8
    Returns bool array (8415,) where True = valid (sum==10, ≥1 non-zero).
    Uses 2-D prefix sums → O(R·C + N_RECTS).
    """
    ps = np.zeros((ROWS + 1, COLS + 1), dtype=np.int32)
    ps[1:, 1:] = np.cumsum(np.cumsum(board.astype(np.int32), axis=0), axis=1)

    # Rectangle sums for all 8415 rects at once
    sums = (ps[_R2 + 1, _C2 + 1]
            - ps[_R1,     _C2 + 1]
            - ps[_R2 + 1, _C1    ]
            + ps[_R1,     _C1    ])

    # Non-zero count (≥1 cell must be non-zero so removal is meaningful)
    nz_ps = np.zeros((ROWS + 1, COLS + 1), dtype=np.int32)
    nz_ps[1:, 1:] = np.cumsum(np.cumsum((board > 0).astype(np.int32), axis=0), axis=1)
    nz = (nz_ps[_R2 + 1, _C2 + 1]
          - nz_ps[_R1,     _C2 + 1]
          - nz_ps[_R2 + 1, _C1    ]
          + nz_ps[_R1,     _C1    ])

    return (sums == 10) & (nz > 0)


# ── Random board generator ─────────────────────────────────────────────────────

def random_board() -> np.ndarray:
    """Generate a valid random board (total sum divisible by 10)."""
    while True:
        cells = np.random.randint(1, 10, size=NCELLS - 1, dtype=np.uint8)
        total = int(cells.sum())
        r = (10 - total % 10) % 10
        if 1 <= r <= 9:
            board = np.append(cells, np.uint8(r)).reshape(ROWS, COLS)
            return board.astype(np.uint8)


# ── Gymnasium Environment ──────────────────────────────────────────────────────

class AppleGameEnv(gym.Env):
    """
    Apple game as a Gymnasium environment for MaskablePPO.
    observation: (10, 17) uint8 board
    action:      Discrete(8415) = index into ALL_RECTS
    reward:      number of cells removed by the chosen rectangle
    """
    metadata = {}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0, high=9, shape=(ROWS, COLS), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(N_RECTS)
        self._board: np.ndarray = np.zeros((ROWS, COLS), dtype=np.uint8)

    # SB3 ActionMasker callback
    def action_masks(self) -> np.ndarray:
        return compute_mask(self._board)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self._board = random_board()
        return self._board.copy(), {}

    def step(self, action: int):
        r1, c1, r2, c2 = _R1[action], _C1[action], _R2[action], _C2[action]
        mask = self._board[r1:r2+1, c1:c2+1] > 0
        removed = int(mask.sum())
        self._board[r1:r2+1, c1:c2+1] *= (~mask).astype(np.uint8)

        # Terminal when no valid moves remain
        done = not compute_mask(self._board).any()
        return self._board.copy(), float(removed), done, False, {}

    def render(self):
        pass


def make_env(rank: int = 0):
    def _init():
        env = AppleGameEnv()
        env = ActionMasker(env, lambda e: e.action_masks())
        return env
    return _init


# ── Custom Feature Extractor (loads SL backbone) ──────────────────────────────

class AppleNetExtractor(BaseFeaturesExtractor):
    """
    ResNet feature extractor for MaskablePPO.
    Loads stem+blocks from the SL checkpoint when sl_ckpt is provided.
    outputs features_dim = channels + 3  (backbone GAP + aux)
    """

    def __init__(self, observation_space: spaces.Box,
                 channels: int = 128, n_blocks: int = 6,
                 sl_ckpt: str = ""):
        features_dim = channels + 3
        super().__init__(observation_space, features_dim=features_dim)
        self._ch = channels

        from train_sl import ResBlock
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(n_blocks)])
        self.gap    = nn.AdaptiveAvgPool2d(1)

        if sl_ckpt and os.path.exists(sl_ckpt):
            ckpt = torch.load(sl_ckpt, map_location="cpu")
            sd   = ckpt["model"]
            stem_sd  = {k[len("stem."):]:  v for k, v in sd.items() if k.startswith("stem.")}
            block_sd = {k[len("blocks."): ]: v for k, v in sd.items() if k.startswith("blocks.")}
            self.stem.load_state_dict(stem_sd)
            self.blocks.load_state_dict(block_sd)
            print(f"[AppleNetExtractor] Loaded backbone from {sl_ckpt}")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, 10, 17) uint8 (SB3 converts to float32 automatically)
        B = obs.shape[0]
        x = obs / 9.0                          # (B, 10, 17)  float32
        board_in = x.unsqueeze(1)              # (B, 1, 10, 17)

        feat = self.gap(self.blocks(self.stem(board_in))).flatten(1)  # (B, ch)

        # Aux features — match train_sl.py exactly
        nz_count = (obs > 0).float().view(B, -1).sum(1)               # (B,)
        nz = nz_count / NCELLS
        s  = obs.float().view(B, -1).sum(1) / MAX_SUM                  # (B,)
        aux = torch.stack([nz, nz * (nz - 1.0 / NCELLS), s], dim=1)   # (B, 3)

        return torch.cat([feat, aux], dim=1)   # (B, ch+3)


# ── Callback: track episode scores ────────────────────────────────────────────

class ScoreLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._ep_scores = []
        self._best      = 0.0
        self._iter      = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self._ep_scores.append(info["episode"]["r"])
        return True

    def _on_rollout_end(self) -> None:
        self._iter += 1
        if self._ep_scores:
            mean  = float(np.mean(self._ep_scores))
            med   = float(np.median(self._ep_scores))
            best  = float(np.max(self._ep_scores))
            n     = len(self._ep_scores)
            if mean > self._best:
                self._best = mean
            print(f"  iter {self._iter:4d}  n={n:4d}  "
                  f"mean={mean:.2f}  med={med:.2f}  max={best:.0f}  "
                  f"best_mean={self._best:.2f}")
            self._ep_scores.clear()


# ── ONNX export ────────────────────────────────────────────────────────────────

class ValueWrapper(nn.Module):
    """
    Wraps the MaskablePPO critic for ONNX export.
    Inputs : board (1, 1, 10, 17) float32,  aux (1, 3) float32
    Output : value scalar (1,)  float32

    Compatible with Go's nnCtxV2 which reads a single output tensor.
    """

    def __init__(self, policy):
        super().__init__()
        # policy.features_extractor has stem/blocks/gap
        fe = policy.features_extractor
        self.stem   = fe.stem
        self.blocks = fe.blocks
        self.gap    = fe.gap
        self._ch    = fe._ch
        # Value network (mlp_extractor value_net + value_net head)
        self.mlp_value = policy.mlp_extractor.value_net
        self.value_net = policy.value_net

    def forward(self, board: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """
        board : (B, 1, 10, 17)  already divided by 9 (or raw; we normalise here)
        aux   : (B, 3)
        """
        feat   = self.gap(self.blocks(self.stem(board))).flatten(1)   # (B, ch)
        feats  = torch.cat([feat, aux], dim=1)                         # (B, ch+3)
        value  = self.value_net(self.mlp_value(feats))                 # (B, 1)
        return value.squeeze(-1)                                       # (B,)


def export_onnx_value(model_zip_path: str, out_path: str):
    policy = MaskablePPO.load(model_zip_path).policy.cpu().eval()
    wrapper = ValueWrapper(policy).eval()

    dummy_board = torch.zeros(1, 1, ROWS, COLS)
    dummy_aux   = torch.zeros(1, 3)

    torch.onnx.export(
        wrapper,
        (dummy_board, dummy_aux),
        out_path,
        input_names  = ["board", "aux"],
        output_names = ["score"],
        dynamic_axes = {
            "board": {0: "batch"},
            "aux"  : {0: "batch"},
            "score": {0: "batch"},
        },
        opset_version = 17,
    )
    print(f"ONNX value model exported: {out_path}  "
          f"({os.path.getsize(out_path)/1e6:.1f} MB)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sl",       default="model_sl.pt",  help="SL checkpoint (backbone init)")
    parser.add_argument("--out",      default="model_rl.zip",  help="output SB3 zip")
    parser.add_argument("--export",   default="",              help="export ONNX value model")
    parser.add_argument("--iters",    type=int,   default=200,  help="number of PPO update iterations")
    parser.add_argument("--n-envs",   type=int,   default=8,    help="parallel envs")
    parser.add_argument("--n-steps",  type=int,   default=2048, help="rollout steps per env per iter")
    parser.add_argument("--batch",    type=int,   default=512)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--lr",       type=float, default=3e-4)
    parser.add_argument("--channels", type=int,   default=128)
    parser.add_argument("--blocks",   type=int,   default=6)
    parser.add_argument("--seed",     type=int,   default=42)
    args = parser.parse_args()

    # ── check SL checkpoint ─────────────────────────────────────────────────
    if not os.path.exists(args.sl):
        raise FileNotFoundError(f"SL checkpoint not found: {args.sl}")

    sl_cfg = torch.load(args.sl, map_location="cpu").get(
        "config", {"channels": args.channels, "blocks": args.blocks}
    )
    channels = sl_cfg.get("channels", args.channels)
    n_blocks = sl_cfg.get("blocks",   args.blocks)
    print(f"SL config: channels={channels}, blocks={n_blocks}")

    # ── build envs ──────────────────────────────────────────────────────────
    env_fns = [make_env(i) for i in range(args.n_envs)]
    # SubprocVecEnv is faster for CPU-heavy envs; DummyVecEnv is simpler
    vec_env = SubprocVecEnv(env_fns) if args.n_envs > 1 else DummyVecEnv(env_fns)

    # ── policy kwargs ───────────────────────────────────────────────────────
    policy_kwargs = dict(
        features_extractor_class  = AppleNetExtractor,
        features_extractor_kwargs = dict(
            channels = channels,
            n_blocks = n_blocks,
            sl_ckpt  = args.sl,
        ),
        net_arch = dict(pi=[256, 128], vf=[256, 128]),
    )

    total_timesteps = args.iters * args.n_envs * args.n_steps

    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        learning_rate     = args.lr,
        n_steps           = args.n_steps,
        batch_size        = args.batch,
        n_epochs          = args.ppo_epochs,
        gamma             = 1.0,           # no discounting (maximise total removed)
        gae_lambda        = 0.95,
        clip_range        = 0.2,
        ent_coef          = 0.01,
        vf_coef           = 0.5,
        max_grad_norm     = 1.0,
        policy_kwargs     = policy_kwargs,
        verbose           = 0,
        seed              = args.seed,
        device            = "auto",
    )

    print(f"\nMaskablePPO  envs={args.n_envs}  "
          f"n_steps={args.n_steps}  iters={args.iters}  "
          f"total_steps={total_timesteps:,}")
    print(f"Policy device: {model.device}\n")

    callback = ScoreLogger()

    t0 = time.time()
    model.learn(
        total_timesteps = total_timesteps,
        callback        = callback,
        reset_num_timesteps = True,
    )
    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed/60:.1f} min")

    model.save(args.out)
    print(f"Saved: {args.out}")

    if args.export:
        export_onnx_value(args.out, args.export)


if __name__ == "__main__":
    main()
