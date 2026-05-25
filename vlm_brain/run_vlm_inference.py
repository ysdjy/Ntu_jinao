"""Run Qwen3-VL blueprint generation or a prompt-only dry run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from blueprint_parser import extract_json_from_text
from blueprint_validator import validate_blueprint
from qwen3vl_loader import generate_response, load_qwen3vl_model

VLM_DIR = Path(__file__).resolve().parent
REPO_ROOT = VLM_DIR.parents[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a skill_blueprint JSON with Qwen3-VL.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--image", default=None)
    parser.add_argument("--scene_state", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--validate", default="true")
    parser.add_argument("--dry_run", default="false")
    args = parser.parse_args()

    config = _read_json(Path(args.config))
    scene_state = _read_json(Path(args.scene_state))
    output_json = Path(args.output_json).expanduser()
    output_dir = output_json.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_generation_prompt(args.task, scene_state)

    if _str_to_bool(args.dry_run):
        (output_dir / "dry_run_prompt.txt").write_text(prompt, encoding="utf-8")
        print(f"[INFO] Dry-run prompt written to: {(output_dir / 'dry_run_prompt.txt').resolve()}")
        return

    model, processor = load_qwen3vl_model(config)
    raw_response = generate_response(model, processor, args.image, prompt, config)
    (output_dir / "raw_response.txt").write_text(raw_response, encoding="utf-8")
    parse_result = extract_json_from_text(raw_response)
    if parse_result.extracted_text:
        (output_dir / "parsed_blueprint.json").write_text(parse_result.extracted_text + "\n", encoding="utf-8")

    validation_report: dict[str, Any] = {"parse_ok": parse_result.ok, "parse_error": parse_result.error}
    if parse_result.ok and parse_result.blueprint is not None and _str_to_bool(args.validate):
        validation = validate_blueprint(parse_result.blueprint)
        validation_report.update(validation.to_dict())
        if validation.valid:
            _write_json(output_json, parse_result.blueprint)
    elif parse_result.ok and parse_result.blueprint is not None:
        validation_report.update({"valid": None, "errors": [], "warnings": ["validation_skipped"]})
        _write_json(output_json, parse_result.blueprint)

    _write_json(output_dir / "validation_report.json", validation_report)
    if not parse_result.ok:
        raise RuntimeError(f"Failed to parse VLM output: {parse_result.error}")
    if _str_to_bool(args.validate) and not validation_report.get("valid"):
        raise RuntimeError(f"Generated blueprint failed validation: {validation_report.get('errors')}")


def build_generation_prompt(task: str, scene_state: dict[str, Any]) -> str:
    system_prompt = _read_text(VLM_DIR / "prompts" / "system_prompt.md")
    generation_prompt = _read_text(VLM_DIR / "prompts" / "blueprint_generation_prompt.md")
    schema = _read_json(VLM_DIR / "schemas" / "skill_blueprint_schema.json")
    example = _read_json(VLM_DIR / "examples" / "sample_blueprint.json")
    return (
        f"{system_prompt}\n\n"
        f"{generation_prompt}\n\n"
        f"Task instruction:\n{task}\n\n"
        f"Scene state JSON:\n{json.dumps(scene_state, indent=2, ensure_ascii=False)}\n\n"
        f"Available skills:\nmove_above, reach, descend, grasp, lift, place, retreat, wait, align_orientation\n\n"
        f"Available logic:\nskill, condition, parallel, terminal\n\n"
        f"Performance metrics:\nsuccess, execution_steps, execution_time, trajectory_length, final_ee_position, "
        f"target_position, final_ee_position_error, final_ee_orientation_error, final_ee_linear_speed, "
        f"average_ee_linear_speed, final_object_position, object_target_xy_error, object_target_position_error, "
        f"timeout, failure_reason\n\n"
        f"Output schema:\n{json.dumps(schema, indent=2, ensure_ascii=False)}\n\n"
        f"Example blueprint:\n{json.dumps(example, indent=2, ensure_ascii=False)}\n"
    )


def _read_json(path: Path) -> dict[str, Any]:
    with path.expanduser().open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_text(path: Path) -> str:
    return path.expanduser().read_text(encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.expanduser().open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes", "y"}


if __name__ == "__main__":
    main()
