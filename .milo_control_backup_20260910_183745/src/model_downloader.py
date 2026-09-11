#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download


ROOT = Path(
    os.getenv(
        "JETSON_FRIEND_ROOT",
        Path(__file__).resolve().parents[1],
    )
).resolve()

MODELS_DIR = ROOT / "models"
CATALOG_PATH = MODELS_DIR / "catalog.json"


def load_catalog():
    if not CATALOG_PATH.is_file():
        raise FileNotFoundError(
            f"Model catalog not found: {CATALOG_PATH}"
        )

    with CATALOG_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = json.load(handle)

    models = data.get("models")

    if not isinstance(models, dict):
        raise ValueError(
            "catalog.json must contain a 'models' object."
        )

    return models


def target_path(file_info):
    return (
        MODELS_DIR
        / str(file_info["target"])
    ).resolve()


def model_installed(info):
    files = info.get("files", [])

    return bool(files) and all(
        target_path(item).is_file()
        and target_path(item).stat().st_size > 0
        for item in files
    )


def free_gb():
    usage = shutil.disk_usage(MODELS_DIR)
    return usage.free / (1024 ** 3)


def print_models(models):
    for model_id, info in models.items():
        installed = model_installed(info)

        marker = "installed" if installed else "not installed"

        print(
            f"{model_id:24} "
            f"{marker:13} "
            f"{info.get('type', 'text'):7} "
            f"{info.get('name', model_id)}"
        )

        notes = str(
            info.get(
                "notes",
                "",
            )
        ).strip()

        if notes:
            print(
                "  "
                + notes
            )


def download_model(model_id, models):
    if model_id not in models:
        raise KeyError(
            f"Unknown model id: {model_id}"
        )

    info = models[model_id]
    repo = str(info["repo"])
    files = info.get("files", [])

    if not files:
        raise ValueError(
            f"No files configured for {model_id}"
        )

    print(
        f"Installing: {info.get('name', model_id)}"
    )
    print(
        f"Free disk space: {free_gb():.1f} GB"
    )

    for file_info in files:
        filename = str(
            file_info["filename"]
        )

        destination = target_path(
            file_info
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if (
            destination.is_file()
            and destination.stat().st_size > 0
        ):
            print(
                "Already installed:",
                destination,
            )
        else:
            print(
                "Downloading:",
                filename,
            )

            downloaded = hf_hub_download(
                repo_id=repo,
                filename=filename,
            )

            temporary = destination.with_suffix(
                destination.suffix + ".part"
            )

            if temporary.exists():
                temporary.unlink()

            shutil.copy2(
                downloaded,
                temporary,
            )

            if (
                not temporary.is_file()
                or temporary.stat().st_size == 0
            ):
                temporary.unlink(
                    missing_ok=True
                )

                raise RuntimeError(
                    "Downloaded file is empty: "
                    + filename
                )

            temporary.replace(
                destination
            )

            print(
                "Installed:",
                destination,
            )

    print(
        f"Done: {model_id}"
    )


def delete_model(model_id, models):
    if model_id not in models:
        raise KeyError(
            f"Unknown model id: {model_id}"
        )

    removed = False

    for file_info in models[
        model_id
    ].get(
        "files",
        [],
    ):
        path = target_path(
            file_info
        )

        if path.exists():
            path.unlink()
            print(
                "Deleted:",
                path,
            )
            removed = True

    if not removed:
        print(
            "Nothing to delete."
        )


def show_model(model_id, models):
    if model_id not in models:
        raise KeyError(
            f"Unknown model id: {model_id}"
        )

    info = dict(
        models[model_id]
    )

    info["id"] = model_id
    info["installed"] = model_installed(
        info
    )

    info["paths"] = [
        str(
            target_path(
                item
            )
        )
        for item in info.get(
            "files",
            [],
        )
    ]

    print(
        json.dumps(
            info,
            indent=2,
            ensure_ascii=False,
        )
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Download and manage local MILO "
            "LLM/VLM models."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    subparsers.add_parser(
        "list",
        help="List models from the catalog.",
    )

    download_parser = subparsers.add_parser(
        "download",
        help="Download one model.",
    )

    download_parser.add_argument(
        "model_id",
    )

    delete_parser = subparsers.add_parser(
        "delete",
        help="Delete one installed model.",
    )

    delete_parser.add_argument(
        "model_id",
    )

    show_parser = subparsers.add_parser(
        "show",
        help="Show model metadata and paths.",
    )

    show_parser.add_argument(
        "model_id",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    models = load_catalog()

    if args.command == "list":
        print_models(
            models
        )

    elif args.command == "download":
        download_model(
            args.model_id,
            models,
        )

    elif args.command == "delete":
        delete_model(
            args.model_id,
            models,
        )

    elif args.command == "show":
        show_model(
            args.model_id,
            models,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            "ERROR:",
            exc,
            file=sys.stderr,
        )
        raise SystemExit(1)
