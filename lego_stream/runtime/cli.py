from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .pipeline import (
    run_causal_stream,
)


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config JSON must contain an object")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Causal MOSS rolling ASR with ERes2NetV2 global speaker association"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    child = subparsers.add_parser("run-stream")
    child.add_argument("--config", type=Path, required=True)
    child.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load_config(args.config)
    result = run_causal_stream(config, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
