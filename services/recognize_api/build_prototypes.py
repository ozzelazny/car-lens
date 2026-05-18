"""In-container shim for the ``build-prototypes`` console script.

Most users will invoke ``build-prototypes`` directly from the host
(it's registered in ``pyproject.toml [project.scripts]``). This script
exists so that a one-shot rebuild can also be triggered from inside the
running container:

    docker compose run --rm recognize-api \
        python build_prototypes.py --source compcars \
        --checkpoint /app/models/mobileclip_s2_compcars_epoch09_top1_91.9.pt \
        --output /app/cache/prototypes.pt --device cpu

The CLI implementation lives in
``car_lense_engine.eval.build_prototypes_cli`` so both entry points
share the same flag parsing + validation.
"""

from __future__ import annotations

import sys

from car_lense_engine.eval.build_prototypes_cli import main

if __name__ == "__main__":
    sys.exit(main())
