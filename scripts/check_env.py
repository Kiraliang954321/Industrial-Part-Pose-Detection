"""Read-only local environment check; never imports or downloads the model."""

import importlib.util
import json
import sys


def main() -> int:
    print(json.dumps({"python": sys.version.split()[0],
                      "ultralytics_installed": importlib.util.find_spec("ultralytics") is not None}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
