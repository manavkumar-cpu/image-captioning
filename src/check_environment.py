"""Phase 1 check. No dataset/model downloads and no training."""

import argparse
import importlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone
from uuid import uuid4

from src.utils.config import load_config
from src.utils.exceptions import (
    CaptioningError,
    EnvironmentCheckError,
)
from src.utils.logger import get_logger


PACKAGES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "numpy": "numpy",
    "PIL": "Pillow",
    "yaml": "PyYAML",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--config",
        default="configs/baseline_debug.yaml",
    )

    parser.add_argument(
        "--config-only",
        action="store_true",
    )

    args = parser.parse_args()
    logger = get_logger()

    try:
        if sys.version_info < (3, 10):
            raise EnvironmentCheckError(
                "Use Python 3.10+; "
                "Python 3.11 is our setup target."
            )

        config = load_config(args.config)

        logger.info("Configuration validation: PASS")

        if args.config_only:
            logger.info(
                "CONFIG-ONLY PASS; "
                "dependency/device checks have NOT run"
            )
            return 0

        versions = {}
        failures = []

        for module, distribution in PACKAGES.items():
            try:
                importlib.import_module(module)

                versions[distribution] = metadata.version(
                    distribution
                )

                logger.info(
                    "%s: %s",
                    distribution,
                    versions[distribution],
                )

            except Exception as exc:
                failures.append(f"{distribution}: {exc}")

        if failures:
            raise EnvironmentCheckError(
                "Dependency checks failed:\n"
                + "\n".join(failures)
            )

        import torch

        from src.utils.device import select_device
        from src.utils.seed import seed_everything

        seed_everything(config["project"]["seed"])

        device = select_device(
            config["runtime"]["device"]
        )

        # Check computation AND autograd on the selected device.
        x = torch.ones(
            (2, 3),
            device=device,
            requires_grad=True,
        )

        x.square().mean().backward()

        if (
            x.grad is None
            or not bool(torch.isfinite(x.grad).all().item())
        ):
            raise EnvironmentCheckError(
                "Device backward check failed"
            )

        logger.info("Selected device: %s", device)
        logger.info("Tensor forward/backward: PASS")

        run_id = (
            datetime.now(timezone.utc).strftime(
                "%Y%m%dT%H%M%S_%fZ"
            )
            + "_"
            + uuid4().hex[:8]
        )

        directory = (
            Path(config["paths"]["artifacts_root"])
            / "setup"
            / run_id
        )

        directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        report = {
            "status": "passed",
            "python": sys.version,
            "platform": platform.platform(),
            "device": str(device),
            "versions": versions,
            "config": config,
            "scope": (
                "Environment only; no data, pretrained "
                "weights, model, or training checked"
            ),
        }

        (directory / "environment.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )

        installed = sorted(
            f"{distribution.metadata['Name']}=="
            f"{distribution.version}"
            for distribution in metadata.distributions()
            if distribution.metadata.get("Name")
        )

        (directory / "installed-packages.txt").write_text(
            "\n".join(installed) + "\n"
        )

        logger.info(
            "Environment report: %s",
            directory / "environment.json",
        )

        logger.info(
            "PHASE 1 ENVIRONMENT CHECK: PASS"
        )

        return 0

    except (
        CaptioningError,
        OSError,
        RuntimeError,
    ) as exc:
        logger.error("CHECK FAILED: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())