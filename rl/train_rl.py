"""
Phase 3: PPO Reinforcement Learning — self-play fine-tuning
============================================================
Starts from the SL checkpoint (model_sl.pt) and improves via PPO self-play.

Usage:
    python train_rl.py --sl model_sl.pt --out model_rl.pt
    python train_rl.py --sl model_sl.pt --out model_rl.pt --episodes 400 --ppo-epochs 4

After training, export to ONNX:
    python train_rl.py --sl model_sl.pt --out model_rl.pt --export model_rl.onnx

Core idea:
    SL label = playGreedy(b2)      → Greedy ceiling, fixed
    RL label = actual_remaining    → rises when model beats Greedy → virtuous cycle

PPO with shared value network:
    π(a|s) ∝ exp(model(apply(s,a)))   for a in valid_actions
    V(s)   = E_π[ model(apply(s,a)) ]  (same forward pass)
"""

import argparse
import math
import os
import random
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ── Constants ──────────────────────────────────────────────────────────────────
ROWS, COLS   = 10, 17
NCELLS       = ROWS * COLS      # 170
MAX_SCORE    = float(NCELLS)    # 170.0
MAX_SUM      = NCELLS * 9.0    # 1530.0

# ── Import model definition from train_sl ─────────────────────────────────────
from train_sl import AppleNetSL, ROWS, COLS, NCELLS, MAX_SCORE, MAX_SUM

# ── Python Game Engine ─────────────────────────────────────────────────────────

def build_prefix_sum(board: np.ndarray):
    """Build 2-D prefix sums for value and non-zero count. O(R×C)."""
    val = np.zeros((ROWS + 1, COLS + 1), dtype=np.int32)
    cnt = np.zeros((ROWS + 1, COLS + 1), dtype=np.int32)
    val[1:, 1:] = np.cumsum(np.cumsum(board.astype(np.int32), axis=0), axis=1)
    cnt[1:, 1:] = np.cumsum(np.cumsum((board > 0).astype(np.int32), axis=0), axis=1)
    return val, cnt


def query(ps: np.ndarray, r1: int, c1: int, r2: int, c2: int) -> int:
    return int(ps[r2 + 1, c2 + 1] - ps[r1, c2 + 1]
               - ps[r2 + 1, c1]   + ps[r1, c1])


def get_valid_rects(board: np.ndarray):
    """Return list of (r1,c1,r2,c2) rectangles with value-sum == 10."""
    val, _ = build_prefix_sum(board)
    rects = []
    for r1 in range(ROWS):
        for r2 in range(r1, ROWS):
            for c1 in range(COLS):
                for c2 in range(c1, COLS):
                    if query(val, r1, c1, r2, c2) == 10:
                        rects.append((r1, c1, r2, c2))
    return rects


def apply_rect(board: np.ndarray, r1: int, c1: int, r2: int, c2: int):
    """Return (new_board, removed_count)."""
    b2 = board.copy()
    mask = b2[r1:r2+1, c1:c2+1] > 0
    removed = int(mask.sum())
    b2[r1:r2+1, c1:c2+1] *= ~mask
    return b2, removed


def random_board(rng: random.Random) -> np.ndarray:
    """Generate a valid random board (same distribution as Go genBoard)."""
    while True:
        b = np.array([rng.randint(1, 9) for _ in range(NCELLS - 1)],
                     dtype=np.int8).reshape(ROWS, COLS)
        # leave last cell empty for adjustment
        b_flat = b.flatten()
        total = int(b_flat.sum())
        r = (10 - total % 10) % 10
        if r >= 1:
            full = np.append(b_flat, r).reshape(ROWS, COLS)
            return full.astype(np.int8)


def compute_aux(boards: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    boards: (N, 10, 17) int8  or float32
    returns: (N, 3) float32 tensor on device
    """
    b = boards.astype(np.float32)
    nz  = (b > 0).sum(axis=(1, 2)) / NCELLS              # (N,)
    s   = b.sum(axis=(1, 2)) / MAX_SUM                    # (N,)
    aux = np.stack([nz, nz * (nz - 1.0 / NCELLS), s], axis=1)  # (N, 3)
    return torch.tensor(aux, dtype=torch.float32, device=device)


@torch.no_grad()
def model_scores(model: AppleNetSL, next_boards: list, device: torch.device
                 ) -> np.ndarray:
    """Batch-evaluate model on a list of next board states. Returns (N,) numpy."""
    arr = np.stack(next_boards, axis=0).astype(np.float32)   # (N, 10, 17)
    x   = torch.tensor(arr[:, None, :, :], dtype=torch.float32,
                        device=device) / 9.0                  # (N, 1, 10, 17)
    aux = compute_aux(arr, device)                            # (N, 3)
    return model(x, aux).cpu().numpy()                        # (N,)


def play_episode(model: AppleNetSL, board: np.ndarray,
                 epsilon: float, device: torch.device, rng: random.Random):
    """
    Play one episode greedily (with ε-greedy exploration).
    Returns trajectory: list of (b2: np.ndarray, removed: int)
    """
    trajectory = []
    cur = board.copy()
    while True:
        rects = get_valid_rects(cur)
        if not rects:
            break
        next_boards, removals = [], []
        for r in rects:
            b2, rem = apply_rect(cur, *r)
            next_boards.append(b2)
            removals.append(rem)

        if rng.random() < epsilon:
            idx = rng.randrange(len(rects))
        else:
            scores = model_scores(model, next_boards, device)
            idx    = int(scores.argmax())

        trajectory.append((next_boards[idx], removals[idx]))
        cur = next_boards[idx]
    return trajectory


# ── PPO helpers ────────────────────────────────────────────────────────────────

def compute_log_pi(model: AppleNetSL, boards_x: torch.Tensor,
                   aux_x: torch.Tensor, chosen_x: torch.Tensor,
                   chosen_aux: torch.Tensor,
                   step_idx: torch.Tensor, step_counts: torch.Tensor
                   ) -> torch.Tensor:
    """
    For each step in a batch, compute log π(chosen | state).

    boards_x   : (total_candidates, 1, R, C)  all next-states across all steps
    aux_x      : (total_candidates, 3)
    chosen_x   : (B, 1, R, C)                 the chosen next-state per step
    chosen_aux : (B, 3)
    step_idx   : (total_candidates,) which step each candidate belongs to
    step_counts: (B,) number of candidates per step

    Returns log_probs: (B,)
    """
    all_scores    = model(boards_x, aux_x)          # (total,)
    chosen_scores = model(chosen_x, chosen_aux)     # (B,)

    B = len(step_counts)
    log_probs = torch.zeros(B, device=all_scores.device)

    # For each step: log_softmax over all candidates, pick chosen
    offset = 0
    for i in range(B):
        n = step_counts[i].item()
        step_logits = all_scores[offset: offset + n]   # (n,)
        log_z       = torch.logsumexp(step_logits, dim=0)
        log_probs[i] = chosen_scores[i] - log_z
        offset += n
    return log_probs


# ── Main training loop ─────────────────────────────────────────────────────────

def ppo_update(model, old_model, optimizer,
               episodes_data,          # list of (b2, remaining, all_cands, chosen_idx)
               clip_eps, lambda_v, lambda_e, lambda_kl,
               ppo_epochs, batch_size, device):
    """
    episodes_data: collected from play_episode_full (see below).
    """
    if not episodes_data:
        return {}

    # Unpack
    all_boards   = []  # chosen b2 per step
    all_remaining = [] # Monte-Carlo return (actual remaining after step)
    all_cand_boards = []   # all candidate b2 for each step
    all_cand_counts = []   # number of candidates per step
    all_chosen_idx  = []   # which candidate was chosen

    for b2, rem, cands, cidx in episodes_data:
        all_boards.append(b2)
        all_remaining.append(rem)
        all_cand_boards.extend(cands)
        all_cand_counts.append(len(cands))
        all_chosen_idx.append(cidx)

    B = len(all_boards)

    # Tensors for chosen states
    bx = np.stack(all_boards).astype(np.float32)         # (B, 10, 17)
    board_t = torch.tensor(bx[:, None, :, :] / 9.0,
                            dtype=torch.float32, device=device)  # (B, 1, R, C)
    aux_t   = compute_aux(bx, device)                            # (B, 3)
    ret_t   = torch.tensor(all_remaining, dtype=torch.float32,
                            device=device) / MAX_SCORE           # (B,)

    # Tensors for all candidate states
    cx = np.stack(all_cand_boards).astype(np.float32)    # (total, 10, 17)
    cand_board_t = torch.tensor(cx[:, None, :, :] / 9.0,
                                 dtype=torch.float32, device=device)
    cand_aux_t   = compute_aux(cx, device)
    step_idx_t   = torch.tensor(
        [i for i, cnt in enumerate(all_cand_counts) for _ in range(cnt)],
        dtype=torch.long, device=device
    )
    step_cnt_t = torch.tensor(all_cand_counts, dtype=torch.long, device=device)

    # Compute old log-probs (frozen)
    with torch.no_grad():
        old_log_pi = compute_log_pi(
            old_model, cand_board_t, cand_aux_t,
            board_t, aux_t, step_idx_t, step_cnt_t
        )  # (B,)

    metrics = {"p_loss": [], "v_loss": [], "entropy": [], "kl": []}

    for _ in range(ppo_epochs):
        # Shuffle
        perm = torch.randperm(B, device=device)

        for start in range(0, B, batch_size):
            idx = perm[start: start + batch_size]
            if len(idx) == 0:
                continue

            b_b   = board_t[idx]
            a_b   = aux_t[idx]
            ret_b = ret_t[idx]
            olp_b = old_log_pi[idx]

            # ── Value predictions ──────────────────────────────────────────
            pred_v = model(b_b, a_b)                         # (batch,)
            v_loss = F.mse_loss(pred_v, ret_b)

            # ── Advantage ─────────────────────────────────────────────────
            with torch.no_grad():
                adv = ret_b - pred_v.detach()
                # Normalise advantage per mini-batch
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # ── Policy loss (PPO clip) ─────────────────────────────────────
            # Recompute log-probs for this mini-batch
            # (we only have chosen state tensors in b_b; skip full candidate
            #  re-enumeration for efficiency — use value prediction as proxy)
            new_log_pi = compute_log_pi(
                model, cand_board_t, cand_aux_t,
                b_b, a_b, step_idx_t, step_cnt_t
            )[idx]  # This is approximate when using shuffled subset; see note below.

            ratio    = (new_log_pi - olp_b).exp()
            surr1    = ratio * adv
            surr2    = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv
            p_loss   = -torch.min(surr1, surr2).mean()

            # ── Entropy bonus ──────────────────────────────────────────────
            with torch.no_grad():
                all_sc  = model(cand_board_t, cand_aux_t)   # (total,)
            # Compute per-step entropy
            entropy_list = []
            offset = 0
            for cnt in all_cand_counts:
                logits = all_sc[offset: offset + cnt]
                log_p  = logits - torch.logsumexp(logits, dim=0)
                p      = log_p.exp()
                entropy_list.append(-(p * log_p).sum())
                offset += cnt
            entropy = torch.stack(entropy_list).mean()

            # ── KL regularisation (stay close to SL model) ────────────────
            # Approximate KL between new and old policy at chosen states
            kl = (olp_b - new_log_pi).mean()

            loss = p_loss + lambda_v * v_loss - lambda_e * entropy + lambda_kl * kl.clamp(min=0)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            metrics["p_loss"].append(p_loss.item())
            metrics["v_loss"].append(v_loss.item())
            metrics["entropy"].append(entropy.item())
            metrics["kl"].append(kl.item())

    return {k: float(np.mean(v)) for k, v in metrics.items() if v}


def play_episode_full(model, board, epsilon, device, rng):
    """
    Like play_episode but also records all candidates per step (for PPO).
    Returns trajectory_full: list of (b2, remaining_after, all_cand_b2s, chosen_idx)
    """
    steps = []
    cur   = board.copy()
    while True:
        rects = get_valid_rects(cur)
        if not rects:
            break
        next_boards, removals = [], []
        for r in rects:
            b2, rem = apply_rect(cur, *r)
            next_boards.append(b2)
            removals.append(rem)

        if rng.random() < epsilon:
            idx = rng.randrange(len(rects))
        else:
            with torch.no_grad():
                sc = model_scores(model, next_boards, device)
            idx = int(sc.argmax())

        steps.append((next_boards[idx], removals[idx], next_boards, idx))
        cur = next_boards[idx]

    # Compute actual remaining at each step
    total = sum(rem for _, rem, _, _ in steps)
    result = []
    remaining = total
    for b2, removed, cands, cidx in steps:
        remaining -= removed
        result.append((b2, remaining, cands, cidx))
    return result, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sl",          default="model_sl.pt",  help="SL checkpoint to start from")
    parser.add_argument("--out",         default="model_rl.pt",  help="output RL checkpoint")
    parser.add_argument("--export",      default="",             help="export ONNX after training")
    parser.add_argument("--iters",       type=int,   default=200, help="PPO iterations")
    parser.add_argument("--episodes",    type=int,   default=200, help="episodes per iteration")
    parser.add_argument("--ppo-epochs",  type=int,   default=4,   help="PPO update epochs per iter")
    parser.add_argument("--batch",       type=int,   default=1024)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--clip-eps",    type=float, default=0.2)
    parser.add_argument("--lambda-v",    type=float, default=0.5,  help="value loss weight")
    parser.add_argument("--lambda-e",    type=float, default=0.01, help="entropy bonus weight")
    parser.add_argument("--lambda-kl",   type=float, default=0.1,  help="KL reg weight")
    parser.add_argument("--epsilon",     type=float, default=0.05, help="ε-greedy exploration")
    parser.add_argument("--above-median",action="store_true",
                        default=True, help="only train on above-median episodes")
    parser.add_argument("--sl-mix",      type=float, default=0.1,
                        help="fraction of SL data to mix in (0=off)")
    parser.add_argument("--sl-data",     default="sl_data.bin",
                        help="SL binary file for mixed training")
    parser.add_argument("--channels",    type=int, default=128)
    parser.add_argument("--blocks",      type=int, default=6)
    parser.add_argument("--seed",        type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load SL model
    ckpt = torch.load(args.sl, map_location=device)
    cfg  = ckpt.get("config", {"channels": args.channels, "blocks": args.blocks})
    model = AppleNetSL(channels=cfg["channels"], n_blocks=cfg["blocks"]).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded SL model from {args.sl}")

    # Old model (frozen reference for PPO / KL reg)
    old_model = deepcopy(model).to(device)
    for p in old_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.iters, eta_min=args.lr * 0.1
    )

    best_score = 0.0

    print(f"\n{'Iter':>5}  {'Mean':>7}  {'Median':>7}  {'Max':>5}  "
          f"{'Steps':>6}  {'p_loss':>8}  {'v_loss':>8}  {'LR':>8}  {'Time':>7}")
    print("-" * 80)

    for it in range(args.iters):
        t0 = time.time()
        model.eval()

        # ── Collect episodes ───────────────────────────────────────────────
        all_episodes = []
        scores       = []
        for _ in range(args.episodes):
            board = random_board(rng)
            traj, total = play_episode_full(model, board, args.epsilon, device, rng)
            scores.append(total)
            all_episodes.append(traj)

        scores_arr = np.array(scores)
        median_s   = float(np.median(scores_arr))
        mean_s     = float(scores_arr.mean())
        max_s      = int(scores_arr.max())

        # ── Filter: above-median only ──────────────────────────────────────
        if args.above_median:
            filtered = [ep for ep, sc in zip(all_episodes, scores)
                        if sc >= median_s]
        else:
            filtered = all_episodes

        # Flatten steps
        episodes_data = [step for ep in filtered for step in ep]
        n_steps       = len(episodes_data)

        # ── Update old model (for PPO ratio) ──────────────────────────────
        old_model.load_state_dict(model.state_dict())

        # ── PPO update ─────────────────────────────────────────────────────
        model.train()
        m = ppo_update(
            model, old_model, optimizer,
            episodes_data,
            clip_eps=args.clip_eps,
            lambda_v=args.lambda_v,
            lambda_e=args.lambda_e,
            lambda_kl=args.lambda_kl,
            ppo_epochs=args.ppo_epochs,
            batch_size=args.batch,
            device=device,
        )

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(f"{it+1:5d}  {mean_s:7.2f}  {median_s:7.2f}  {max_s:5d}  "
              f"{n_steps:6d}  "
              f"{m.get('p_loss', 0):8.4f}  {m.get('v_loss', 0):8.4f}  "
              f"{lr_now:8.2e}  {elapsed:6.1f}s")

        # ── Save best ──────────────────────────────────────────────────────
        if mean_s > best_score:
            best_score = mean_s
            torch.save({
                "iter"      : it,
                "model"     : model.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "scheduler" : scheduler.state_dict(),
                "best_score": best_score,
                "config"    : cfg,
            }, args.out)
            print(f"      ↑ saved (mean {best_score:.2f})")

    print(f"\nRL training complete. Best mean score: {best_score:.2f}")

    if args.export:
        from train_sl import export_onnx
        ckpt = torch.load(args.out, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        export_onnx(model, args.export, device="cpu")


if __name__ == "__main__":
    main()
