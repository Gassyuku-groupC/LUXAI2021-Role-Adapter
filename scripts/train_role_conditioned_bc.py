#!/usr/bin/env python3
"""Weighted offline BC for Role-Conditioned Local Adapter and role biases."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from imitation_learning_utils import load_agent_flags  # noqa: E402
from train_imitation_bc import bc_loss, to_device  # noqa: E402
from lux_ai.nns import create_model  # noqa: E402


def shard_paths(path: Path):
    result = sorted(path.glob("shard_*.pt"))
    if not result:
        raise ValueError(f"No shards found in {path}")
    return result


def make_batch(shard, indices, device):
    idx = torch.tensor(indices, dtype=torch.long)
    select = lambda values: {k: v.index_select(0, idx) for k, v in values.items()}
    model_input = {
        "obs": to_device(select(shard["obs"]), device),
        "info": {
            "input_mask": shard["input_mask"].index_select(0, idx).to(device),
            "available_actions_mask": to_device(select(shard["available_actions_mask"]), device),
            "role_bias_codes": to_device(select(shard["role_bias_codes"]), device),
            "role_bias_scale": shard["role_bias_scale"].index_select(0, idx).to(device),
        },
    }
    return (
        model_input,
        to_device(select(shard["actions_taken"]), device),
        shard["weights"].index_select(0, idx).to(device),
    )


def delta_metrics(deltas, role_codes):
    values = [
        value[role_codes[space].to(value.device) != 0].abs().reshape(-1)
        for space, value in deltas.items()
        if space in role_codes and (role_codes[space] != 0).any()
    ]
    if not values:
        zero = torch.zeros(())
        return zero, 0.0
    merged = torch.cat(values)
    return merged.mean(), float(merged.max().detach().item())


def active_bc_loss(policy_logits, targets, weights, role_codes):
    total = torch.zeros((), device=weights.device)
    denominator = torch.zeros((), device=weights.device)
    sample_weights = weights.view(-1, 1, 1, 1, 1)
    for space, logits in policy_logits.items():
        if space not in role_codes:
            continue
        target = targets[space].float()
        active_tile = role_codes[space].ne(0).any(dim=-1)
        action_count = target.sum(dim=-1)
        active = active_tile & action_count.gt(0)
        if not active.any():
            continue
        safe_logits = torch.nan_to_num(logits, nan=-30.0, neginf=-30.0, posinf=30.0)
        per_tile = -(F.log_softmax(safe_logits, dim=-1) * target).sum(dim=-1)
        weighted = active.float() * sample_weights
        total = total + (per_tile * weighted).sum()
        denominator = denominator + weighted.sum()
    return total / denominator.clamp(min=1.0)


def teacher_kl_loss(student_logits, teacher_logits, role_codes):
    total = next(iter(student_logits.values())).new_zeros(())
    count = total.clone()
    for space, student in student_logits.items():
        if space not in role_codes:
            continue
        teacher = teacher_logits[space]
        active = role_codes[space].ne(0).any(dim=-1)
        legal = torch.isfinite(student) & torch.isfinite(teacher)
        student_safe = torch.where(legal, student, torch.full_like(student, -1e4))
        teacher_safe = torch.where(legal, teacher, torch.full_like(teacher, -1e4))
        student_prob = F.softmax(student_safe, dim=-1)
        per_tile = (student_prob * (
            F.log_softmax(student_safe, dim=-1) - F.log_softmax(teacher_safe, dim=-1)
        )).sum(dim=-1)
        total = total + per_tile[active].sum()
        count = count + active.sum()
    return total / count.clamp(min=1.0)


@torch.no_grad()
def evaluate(model, paths, batch_size, device):
    model.eval()
    loss_sum = active_loss_sum = delta_sum = 0.0
    delta_max = samples = 0
    for path in paths:
        shard = torch.load(path, map_location="cpu")
        count = int(shard["weights"].shape[0])
        for start in range(0, count, batch_size):
            indices = list(range(start, min(start + batch_size, count)))
            inputs, targets, weights = make_batch(shard, indices, device)
            outputs = model(inputs, sample=False)
            loss, _ = bc_loss(outputs["policy_logits"], targets, weights)
            active_loss = active_bc_loss(
                outputs["policy_logits"], targets, weights, inputs["info"]["role_bias_codes"]
            )
            delta_mean, batch_max = delta_metrics(
                outputs["role_local_deltas"], inputs["info"]["role_bias_codes"]
            )
            loss_sum += float(loss.item()) * len(indices)
            active_loss_sum += float(active_loss.item()) * len(indices)
            delta_sum += float(delta_mean.item()) * len(indices)
            delta_max = max(delta_max, batch_max)
            samples += len(indices)
    return {"loss": loss_sum / samples, "active_loss": active_loss_sum / samples,
            "delta_abs_mean": delta_sum / samples,
            "delta_abs_max": delta_max, "samples": samples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-shards", type=Path, required=True)
    parser.add_argument("--validation-shards", type=Path, required=True)
    parser.add_argument("--agent-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--local-lr", type=float, default=5e-5)
    parser.add_argument("--role-lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--delta-l2-cost", type=float, default=1e-3)
    parser.add_argument("--teacher-kl-cost", type=float, default=0.01)
    parser.add_argument("--local-max-delta", type=float, default=0.05)
    parser.add_argument("--delta-stop-mean", type=float, default=0.015)
    parser.add_argument("--min-validation-improvement", type=float, default=0.002)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--reset-local-projection", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    flags = load_agent_flags(args.agent_dir)
    if not getattr(flags, "role_local_adapter_enabled", False):
        raise ValueError("agent config must enable role_local_adapter")
    model = create_model(flags, device)
    checkpoint = torch.load(args.agent_dir / "lux_ai/rl_agent/candidate_weights.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if args.reset_local_projection:
        for head in model.role_local_adapter.output_heads.values():
            torch.nn.init.zeros_(head.weight)
            torch.nn.init.zeros_(head.bias)
    model.role_local_adapter.max_delta = float(args.local_max_delta)
    teacher = copy.deepcopy(model).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.role_local_adapter.parameters():
        parameter.requires_grad_(True)
    for parameter in model.role_bias_layer.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW([
        {"params": model.role_local_adapter.parameters(), "lr": args.local_lr},
        {"params": model.role_bias_layer.parameters(), "lr": args.role_lr},
    ], weight_decay=args.weight_decay)
    train = shard_paths(args.train_shards)
    validation = shard_paths(args.validation_shards)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    initial = evaluate(model, validation, args.batch_size, device)
    print("initial_validation=" + json.dumps(initial), flush=True)
    history, step = [], 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train)
        loss_sum = samples = 0
        for path in train:
            shard = torch.load(path, map_location="cpu")
            indices = list(range(int(shard["weights"].shape[0])))
            random.shuffle(indices)
            for start in range(0, len(indices), args.batch_size):
                chosen = indices[start:start + args.batch_size]
                inputs, targets, weights = make_batch(shard, chosen, device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs, sample=False)
                bc, _ = bc_loss(outputs["policy_logits"], targets, weights)
                with torch.no_grad():
                    teacher_outputs = teacher(inputs, sample=False)
                kl = teacher_kl_loss(
                    outputs["policy_logits"], teacher_outputs["policy_logits"],
                    inputs["info"]["role_bias_codes"],
                )
                delta_mean, delta_max = delta_metrics(
                    outputs["role_local_deltas"], inputs["info"]["role_bias_codes"]
                )
                loss = bc + args.teacher_kl_cost * kl + args.delta_l2_cost * delta_mean.square()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step {step}")
                loss.backward()
                params = list(model.role_local_adapter.parameters()) + list(model.role_bias_layer.parameters())
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                optimizer.step()
                step += 1
                loss_sum += float(bc.detach().item()) * len(chosen)
                samples += len(chosen)
                if step == 1 or step % args.log_interval == 0:
                    print(f"step={step} epoch={epoch} train_loss={bc.item():.6f} kl={kl.item():.6f} "
                          f"delta_abs_mean={delta_mean.item():.6g} delta_abs_max={delta_max:.6g}", flush=True)
                if args.max_train_batches and step >= args.max_train_batches:
                    break
            if args.max_train_batches and step >= args.max_train_batches:
                break
        metrics = {"epoch": epoch, "step": step, "train_loss": loss_sum / samples,
                   "validation": evaluate(model, validation, args.batch_size, device)}
        history.append(metrics)
        print("epoch_summary=" + json.dumps(metrics), flush=True)
        torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics}, args.output_dir / f"epoch_{epoch:02d}.pt")
        (args.output_dir / "train_summary.json").write_text(
            json.dumps({"initial_validation": initial, "history": history, "config": vars(args)},
                       indent=2, default=str), encoding="utf-8")
        validation_improvement = (
            initial["active_loss"] - metrics["validation"]["active_loss"]
        ) / max(abs(initial["active_loss"]), 1e-8)
        stop_reasons = []
        if metrics["validation"]["delta_abs_mean"] > args.delta_stop_mean:
            stop_reasons.append("delta_mean_exceeded")
        if validation_improvement < args.min_validation_improvement:
            stop_reasons.append("validation_improvement_below_minimum")
        if stop_reasons:
            print("early_stop=" + json.dumps({
                "reasons": stop_reasons,
                "active_validation_improvement": validation_improvement,
            }), flush=True)
            break
        if args.max_train_batches:
            break


if __name__ == "__main__":
    main()
