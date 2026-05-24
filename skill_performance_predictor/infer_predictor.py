"""Run inference for one raw predictor sample JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from dataset_schema import NUMERIC_FEATURE_NAMES
from evaluate_predictor import _build_model_from_checkpoint, _load_checkpoint
from feature_extractor import extract_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict performance for one JSON sample.")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--sample_json", required=True, type=str)
    parser.add_argument("--device", default="auto", type=str)
    args = parser.parse_args()

    device = _resolve_device(args.device)
    checkpoint = _load_checkpoint(Path(args.checkpoint).expanduser(), device)
    model = _build_model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with Path(args.sample_json).expanduser().open("r", encoding="utf-8") as f:
        raw_sample = json.load(f)
    processed = extract_sample(raw_sample)
    batch = _make_batch(processed, checkpoint, device)

    with torch.no_grad():
        outputs = model(batch["numeric_features"], batch["skill_id"], batch["target_id"])
    print(json.dumps(_format_prediction(outputs, checkpoint), indent=2, ensure_ascii=False))


def _make_batch(processed: dict[str, Any], checkpoint: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    stats = checkpoint["feature_stats"]
    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    std = torch.tensor(stats["std"], dtype=torch.float32)
    numeric = torch.tensor(processed["numeric_features"], dtype=torch.float32)
    numeric = (numeric - mean) / std

    vocab = checkpoint["vocab"]
    skill_id = vocab["skill"].get(processed["skill"], vocab["skill"]["unknown"])
    target_id = vocab["target"].get(processed["target"], vocab["target"]["unknown"])
    return {
        "numeric_features": numeric.reshape(1, len(NUMERIC_FEATURE_NAMES)).to(device),
        "skill_id": torch.tensor([skill_id], dtype=torch.long, device=device),
        "target_id": torch.tensor([target_id], dtype=torch.long, device=device),
    }


def _format_prediction(outputs: dict[str, torch.Tensor], checkpoint: dict[str, Any]) -> dict[str, Any]:
    failure_vocab = checkpoint["vocab"]["failure_reason"]
    failure_names = {idx: name for name, idx in failure_vocab.items()}
    failure_probs = torch.softmax(outputs["failure_reason_logits"], dim=-1)[0].detach().cpu()
    failure_idx = int(torch.argmax(failure_probs).item())
    regression = outputs["regression"][0].detach().cpu().tolist()
    return {
        "success_probability": float(torch.sigmoid(outputs["success_logits"])[0].detach().cpu().item()),
        "timeout_probability": float(torch.sigmoid(outputs["timeout_logits"])[0].detach().cpu().item()),
        "failure_reason": failure_names.get(failure_idx, "unknown"),
        "failure_reason_probabilities": {
            failure_names.get(idx, str(idx)): float(prob) for idx, prob in enumerate(failure_probs.tolist())
        },
        "regression": {
            name: float(value) for name, value in zip(checkpoint["regression_target_names"], regression)
        },
    }


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


if __name__ == "__main__":
    main()
