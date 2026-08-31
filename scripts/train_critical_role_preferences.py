#!/usr/bin/env python3
"""Critical-state DPO/Focal-BC training for Role bias and Local Adapter only."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from lux_ai.nns import create_model  # noqa: E402
from lux_ai.rl_agent.role_assignment import RoleCityBiasParams  # noqa: E402
from lux_ai.torchbeast.monobeast import load_training_model_state  # noqa: E402
from lux_ai.utils import flags_to_namespace  # noqa: E402
from train_imitation_bc import to_device  # noqa: E402


def shard_paths(path: Path) -> list[Path]:
    paths = sorted(path.glob("shard_*.pt"))
    if not paths:
        raise ValueError(f"No shards found in {path}")
    return paths


def make_batch(shard: dict, indices: list[int], device: torch.device):
    idx = torch.tensor(indices, dtype=torch.long)
    select = lambda values: {key: value.index_select(0, idx) for key, value in values.items()}
    inputs = {
        "obs": to_device(select(shard["obs"]), device),
        "info": {
            "input_mask": shard["input_mask"].index_select(0, idx).to(device),
            "available_actions_mask": to_device(select(shard["available_actions_mask"]), device),
            "role_bias_codes": to_device(select(shard["role_bias_codes"]), device),
            "role_bias_scale": shard["role_bias_scale"].index_select(0, idx).to(device),
        },
    }
    return {
        "inputs": inputs,
        "preferred": to_device(select(shard["actions_taken"]), device),
        "rejected": to_device(select(shard["rejected_actions"]), device),
        "weights": shard["weights"].index_select(0, idx).to(device),
        "sample_type": shard["sample_type"].index_select(0, idx).to(device),
    }


def selected_log_prob(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    safe = torch.nan_to_num(logits, nan=-30.0, neginf=-30.0, posinf=30.0)
    log_probs = F.log_softmax(safe, dim=-1)
    count = mask.sum(dim=-1).clamp(min=1)
    return (log_probs * mask.float()).sum(dim=-1) / count


def preference_terms(outputs, teacher_outputs, batch, beta: float, focal_gamma: float):
    device = batch["weights"].device
    dpo_sum = torch.zeros((), device=device)
    dpo_weight = torch.zeros((), device=device)
    focal_sum = torch.zeros((), device=device)
    focal_weight = torch.zeros((), device=device)
    margin_sum = torch.zeros((), device=device)
    margin_count = torch.zeros((), device=device)
    preferred_rank = torch.zeros((), device=device)
    flip_count = torch.zeros((), device=device)

    strict_count = torch.zeros((), device=device)
    weak_count = torch.zeros((), device=device)
    for space, logits in outputs["policy_logits"].items():
        preferred = batch["preferred"][space]
        rejected = batch["rejected"][space]
        role_codes = batch["inputs"]["info"]["role_bias_codes"].get(space)
        if role_codes is None:
            continue
        role_active = role_codes.ne(0).any(dim=-1)
        preferred_active = preferred.any(dim=-1) & role_active
        rejected_active = rejected.any(dim=-1) & role_active
        broadcast_shape = (preferred_active.shape[0],) + (1,) * (preferred_active.dim() - 1)
        sample_weights = batch["weights"].view(broadcast_shape)
        sample_type = batch["sample_type"].view(broadcast_shape)
        paired = preferred_active & rejected_active & sample_type.ne(2)
        strict = paired & sample_type.eq(0)
        weak = paired & sample_type.eq(1)
        focal = preferred_active & sample_type.eq(2)

        student_pref = selected_log_prob(logits, preferred)
        student_rej = selected_log_prob(logits, rejected)
        teacher_logits = teacher_outputs["policy_logits"][space]
        teacher_pref = selected_log_prob(teacher_logits, preferred)
        teacher_rej = selected_log_prob(teacher_logits, rejected)
        relative_margin = (student_pref - student_rej) - (teacher_pref - teacher_rej)

        if paired.any():
            pair_weight = sample_weights * torch.where(
                sample_type.eq(0), torch.ones_like(sample_weights), torch.full_like(sample_weights, 0.25)
            )
            dpo = F.softplus(-beta * relative_margin)
            dpo_sum = dpo_sum + (dpo * paired.float() * pair_weight).sum()
            dpo_weight = dpo_weight + (paired.float() * pair_weight).sum()
            margin_sum = margin_sum + relative_margin[paired].sum()
            margin_count = margin_count + paired.sum()
            strict_count = strict_count + strict.sum()
            weak_count = weak_count + weak.sum()
            student_argmax = torch.nan_to_num(logits, nan=-1e4, neginf=-1e4).argmax(dim=-1)
            teacher_argmax = torch.nan_to_num(teacher_logits, nan=-1e4, neginf=-1e4).argmax(dim=-1)
            student_pref_rank = preferred.gather(-1, student_argmax.unsqueeze(-1)).squeeze(-1)
            teacher_pref_rank = preferred.gather(-1, teacher_argmax.unsqueeze(-1)).squeeze(-1)
            preferred_rank = preferred_rank + student_pref_rank[paired].float().sum()
            flip_count = flip_count + (student_pref_rank & ~teacher_pref_rank)[paired].float().sum()

        if focal.any():
            probability = student_pref.exp().clamp(0.0, 1.0)
            focal_loss = -((1.0 - probability) ** focal_gamma) * student_pref
            focal_sum = focal_sum + (focal_loss * focal.float() * sample_weights).sum()
            focal_weight = focal_weight + (focal.float() * sample_weights).sum()

    return {
        "dpo_loss": dpo_sum / dpo_weight.clamp(min=1e-8),
        "focal_loss": focal_sum / focal_weight.clamp(min=1e-8),
        "relative_margin": margin_sum / margin_count.clamp(min=1.0),
        "preferred_rank_rate": preferred_rank / margin_count.clamp(min=1.0),
        "target_flip_rate": flip_count / margin_count.clamp(min=1.0),
        "dpo_weight": dpo_weight,
        "paired_count": margin_count,
        "strict_count": strict_count,
        "weak_count": weak_count,
        "focal_count": focal_weight,
    }


def active_teacher_kl(student_logits, teacher_logits, role_codes):
    total = next(iter(student_logits.values())).new_zeros(())
    count = total.clone()
    for space, student in student_logits.items():
        codes = role_codes.get(space)
        if codes is None:
            continue
        teacher = teacher_logits[space]
        active = codes.ne(0).any(dim=-1)
        legal = torch.isfinite(student) & torch.isfinite(teacher)
        student_safe = torch.where(legal, student, torch.full_like(student, -1e4))
        teacher_safe = torch.where(legal, teacher, torch.full_like(teacher, -1e4))
        probability = F.softmax(student_safe, dim=-1)
        kl = (probability * (
            F.log_softmax(student_safe, dim=-1) - F.log_softmax(teacher_safe, dim=-1)
        )).sum(dim=-1)
        total = total + kl[active].sum()
        count = count + active.sum()
    return total / count.clamp(min=1.0)


def delta_mean(outputs, role_codes):
    values = []
    for space, delta in outputs["role_local_deltas"].items():
        codes = role_codes.get(space)
        if codes is not None and codes.ne(0).any():
            values.append(delta[codes.ne(0)].abs())
    return torch.cat(values).mean() if values else next(iter(outputs["policy_logits"].values())).new_zeros(())


@torch.no_grad()
def evaluate(model, teacher, paths, batch_size, device, beta, focal_gamma):
    model.eval()
    totals = {
        "dpo_loss": 0.0, "focal_loss": 0.0, "relative_margin": 0.0,
        "preferred_rank_rate": 0.0, "target_flip_rate": 0.0,
        "dpo_weight": 0.0, "paired_count": 0.0, "strict_count": 0.0,
        "weak_count": 0.0, "focal_count": 0.0, "samples": 0,
    }
    for path in paths:
        shard = torch.load(path, map_location="cpu")
        count = int(shard["weights"].shape[0])
        for start in range(0, count, batch_size):
            indices = list(range(start, min(start + batch_size, count)))
            batch = make_batch(shard, indices, device)
            outputs = model(batch["inputs"], sample=False)
            teacher_outputs = teacher(batch["inputs"], sample=False)
            terms = preference_terms(outputs, teacher_outputs, batch, beta, focal_gamma)
            paired = float(terms["paired_count"].item())
            dpo_weight = float(terms["dpo_weight"].item())
            focal = float(terms["focal_count"].item())
            totals["dpo_loss"] += float(terms["dpo_loss"].item()) * dpo_weight
            totals["relative_margin"] += float(terms["relative_margin"].item()) * paired
            totals["preferred_rank_rate"] += float(terms["preferred_rank_rate"].item()) * paired
            totals["target_flip_rate"] += float(terms["target_flip_rate"].item()) * paired
            totals["focal_loss"] += float(terms["focal_loss"].item()) * focal
            totals["paired_count"] += paired
            totals["dpo_weight"] += dpo_weight
            totals["strict_count"] += float(terms["strict_count"].item())
            totals["weak_count"] += float(terms["weak_count"].item())
            totals["focal_count"] += focal
            totals["samples"] += len(indices)
    totals["dpo_loss"] /= totals["dpo_weight"] if totals["dpo_weight"] > 0 else 1.0
    for key in ("relative_margin", "preferred_rank_rate", "target_flip_rate"):
        totals[key] /= max(totals["paired_count"], 1.0)
    totals["focal_loss"] /= totals["focal_count"] if totals["focal_count"] > 0 else 1.0
    return totals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-shards", type=Path, required=True)
    parser.add_argument("--validation-shards", type=Path, required=True)
    parser.add_argument("--agent-dir", type=Path, required=True)
    parser.add_argument("--config-name", default="conv_role_local_stage3")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--local-lr", type=float, default=1e-5)
    parser.add_argument("--role-lr", type=float, default=2e-6)
    parser.add_argument("--dpo-beta", type=float, default=2.0)
    parser.add_argument("--dpo-cost", type=float, default=1.0)
    parser.add_argument("--focal-cost", type=float, default=1.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--teacher-kl-cost", type=float, default=0.01)
    parser.add_argument("--delta-l2-cost", type=float, default=1e-3)
    parser.add_argument("--min-margin-improvement", type=float, default=0.02)
    parser.add_argument("--min-rank-improvement", type=float, default=0.03)
    parser.add_argument("--min-strict-pairs-for-head-unlock", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    with initialize_config_dir(config_dir=str((ROOT / "conf").resolve()), version_base=None):
        resolved = OmegaConf.to_container(compose(config_name=args.config_name), resolve=True)
    flags = flags_to_namespace(resolved)
    if not getattr(flags, "role_local_adapter_enabled", False):
        raise ValueError("agent must enable role_local_adapter")
    model = create_model(flags, device)
    checkpoint = torch.load(
        args.agent_dir / "lux_ai/rl_agent/candidate_weights.pt", map_location=device
    )
    load_training_model_state(model, checkpoint["model_state_dict"], flags)
    role_yaml = args.agent_dir / "lux_ai/rl_agent/role_city_bias_params.yaml"
    role_values = RoleCityBiasParams.from_yaml(role_yaml).to_mapping()
    with torch.no_grad():
        for name, value in role_values.items():
            model.role_bias_layer.bias_params[name].fill_(float(value))
    teacher = copy.deepcopy(model).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = list(model.role_local_adapter.parameters()) + list(model.role_bias_layer.parameters())
    for parameter in trainable:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW([
        {"params": model.role_local_adapter.parameters(), "lr": args.local_lr},
        {"params": model.role_bias_layer.parameters(), "lr": args.role_lr},
    ], weight_decay=1e-4)

    train_paths = shard_paths(args.train_shards)
    validation_paths = shard_paths(args.validation_shards)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    initial = evaluate(
        model, teacher, validation_paths, args.batch_size, device, args.dpo_beta, args.focal_gamma
    )
    print("initial_validation=" + json.dumps(initial), flush=True)
    history, step = [], 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_paths)
        for path in train_paths:
            shard = torch.load(path, map_location="cpu")
            indices = list(range(int(shard["weights"].shape[0])))
            random.shuffle(indices)
            for start in range(0, len(indices), args.batch_size):
                chosen = indices[start:start + args.batch_size]
                batch = make_batch(shard, chosen, device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(batch["inputs"], sample=False)
                with torch.no_grad():
                    teacher_outputs = teacher(batch["inputs"], sample=False)
                terms = preference_terms(outputs, teacher_outputs, batch, args.dpo_beta, args.focal_gamma)
                kl = active_teacher_kl(
                    outputs["policy_logits"], teacher_outputs["policy_logits"],
                    batch["inputs"]["info"]["role_bias_codes"],
                )
                delta = delta_mean(outputs, batch["inputs"]["info"]["role_bias_codes"])
                loss = (
                    args.dpo_cost * terms["dpo_loss"]
                    + args.focal_cost * terms["focal_loss"]
                    + args.teacher_kl_cost * kl
                    + args.delta_l2_cost * delta.square()
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite preference loss at step={step}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 5.0)
                optimizer.step()
                step += 1
                if step == 1 or step % args.log_interval == 0:
                    print(
                        f"step={step} loss={loss.item():.6f} dpo={terms['dpo_loss'].item():.6f} "
                        f"focal={terms['focal_loss'].item():.6f} kl={kl.item():.6f} "
                        f"margin={terms['relative_margin'].item():.6f} "
                        f"flip={terms['target_flip_rate'].item():.4f} delta={delta.item():.6g}",
                        flush=True,
                    )
                if args.max_batches and step >= args.max_batches:
                    break
            if args.max_batches and step >= args.max_batches:
                break
        validation = evaluate(
            model, teacher, validation_paths, args.batch_size, device, args.dpo_beta, args.focal_gamma
        )
        margin_improvement = validation["relative_margin"] - initial["relative_margin"]
        rank_improvement = validation["preferred_rank_rate"] - initial["preferred_rank_rate"]
        offline_eligible = (
            validation["strict_count"] >= args.min_strict_pairs_for_head_unlock
            and (
                margin_improvement >= args.min_margin_improvement
                or rank_improvement >= args.min_rank_improvement
            )
        )
        metrics = {
            "epoch": epoch, "step": step, "validation": validation,
            "margin_improvement": margin_improvement,
            "preferred_rank_improvement": rank_improvement,
            "policy_head_unlock_eligible_offline": offline_eligible,
            "requires_extreme_seed_replay_validation": True,
        }
        history.append(metrics)
        print("epoch_summary=" + json.dumps(metrics), flush=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
            },
            args.output_dir / f"epoch_{epoch:02d}.pt",
        )
        (args.output_dir / "train_summary.json").write_text(
            json.dumps({"initial_validation": initial, "history": history, "config": vars(args)}, indent=2, default=str),
            encoding="utf-8",
        )
        if args.max_batches:
            break


if __name__ == "__main__":
    main()
