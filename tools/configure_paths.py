"""Generate a machine-local resource configuration for the archived Walker source."""
import argparse
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--davis-root", required=True, type=Path)
    parser.add_argument("--vlm-checkpoint", required=True, type=Path)
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument("--vlm-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path, default=ROOT / "configs/walker_walk.local.yaml")
    args = parser.parse_args()
    resources = {
        "davis_root": args.davis_root.expanduser().resolve(),
        "vlm_checkpoint": args.vlm_checkpoint.expanduser().resolve(),
        "internvideo2_base_checkpoint": args.base_checkpoint.expanduser().resolve(),
        "fine_turn_root": ROOT / "external_vlm/get_data/fine-turn",
    }
    for name, path in resources.items():
        if not path.exists():
            parser.error(f"{name} does not exist: {path}")
    vlm_python = args.vlm_python.expanduser().resolve()
    if not vlm_python.is_file():
        parser.error(f"VLM Python does not exist: {vlm_python}")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    config = yaml.safe_load((ROOT / "configs/walker_walk.yaml").read_text(encoding="utf-8"))
    config["paths"].update({name: str(path) for name, path in resources.items()})
    config["env"]["davis"]["root"] = str(resources["davis_root"])
    config["external_vlm"]["python_executable"] = str(vlm_python)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Created {args.output.resolve()}")
    print("Set these variables in the environment used by the VLM process:")
    print(f"INTERNVIDEO2_ROOT={ROOT / 'external_vlm/InternVideo/InternVideo2'}")
    print(f"INTERNVIDEO2_WEIGHTS_ROOT={resources['internvideo2_base_checkpoint'].parent}")
    print(f"INTERNVIDEO2_CLIP_CHECKPOINT={resources['internvideo2_base_checkpoint']}")


if __name__ == "__main__":
    main()
