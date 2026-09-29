"""Read-only Phase 1D annotation workspace audit."""

import argparse
import json
from pathlib import Path
import sys

# Allow direct invocation from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from yolo_conveyor.annotation import audit_annotations  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path, help="annotation workspace directory")
    args = parser.parse_args(argv)
    result = audit_annotations(args.workspace)
    print(json.dumps(result.to_dict(), indent=2))
    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
