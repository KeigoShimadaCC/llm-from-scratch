from __future__ import annotations

import argparse
import json
import platform

import torch


def environment_report(optional_mps: bool) -> dict[str, object]:
    mps_available = bool(torch.backends.mps.is_available()) if hasattr(torch.backends, "mps") else False
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cpu_available": True,
        "mps_available": mps_available,
        "mps_required": not optional_mps,
        "status": "pass" if optional_mps or mps_available else "fail",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report local PyTorch/Mac environment capabilities.")
    parser.add_argument("--optional-mps", action="store_true", help="Report MPS availability without requiring it.")
    args = parser.parse_args(argv)
    report = environment_report(optional_mps=args.optional_mps)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
