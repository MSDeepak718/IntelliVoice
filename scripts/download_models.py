#!/usr/bin/env python3
"""
IntelliVoice — Model Download Script

Downloads all required models from HuggingFace Hub.
Run this before starting the application for the first time.

Usage:
    python scripts/download_models.py              # Download all models
    python scripts/download_models.py --model vad   # Download specific model
    python scripts/download_models.py --status      # Check download status
    python scripts/download_models.py --budget      # Show VRAM budget
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings
from config.model_registry import ModelRegistry
from backend.services.model_loader import ModelLoader


def main():
    parser = argparse.ArgumentParser(description="IntelliVoice Model Downloader")
    parser.add_argument(
        "--model",
        type=str,
        help="Download a specific model by name (e.g., silero_vad, xlsr_1b, qwen3_moe)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show download status of all models",
    )
    parser.add_argument(
        "--budget",
        action="store_true",
        help="Show VRAM budget breakdown",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached",
    )
    args = parser.parse_args()

    loader = ModelLoader()

    if args.budget:
        ModelRegistry.print_vram_budget()
        return

    if args.status:
        print("\n=== Model Download Status ===\n")
        status = loader.get_download_status()
        for name, info in status.items():
            icon = "✅" if info["downloaded"] else "❌"
            vram = f"{info['vram_mb']}MB"
            print(f"  {icon} {name:<25} {vram:<10} {info['hf_id']}")
            if info["downloaded"]:
                print(f"     └─ {info['path']}")
        return

    if args.model:
        # Download specific model
        model_map = {m.name: m for m in ModelRegistry.get_all_models()}
        if args.model not in model_map:
            print(f"Unknown model: {args.model}")
            print(f"Available: {', '.join(model_map.keys())}")
            sys.exit(1)

        model_config = model_map[args.model]
        print(f"\nDownloading {model_config.name} ({model_config.hf_model_id})...")
        loader.download_model(model_config, force=args.force)
        print("Done!")
        return

    # Download all models
    print("\n=== Downloading All IntelliVoice Models ===\n")
    ModelRegistry.print_vram_budget()

    print("\n\nStarting downloads...\n")
    results = loader.download_all_models(force=args.force)

    print("\n=== Download Results ===\n")
    for name, result in results.items():
        icon = "✅" if result["status"] == "ok" else "❌"
        print(f"  {icon} {name}: {result.get('path', result.get('error', ''))}")

    errors = [n for n, r in results.items() if r["status"] != "ok"]
    if errors:
        print(f"\n⚠️  {len(errors)} model(s) failed to download: {', '.join(errors)}")
        sys.exit(1)
    else:
        print("\n✅ All models downloaded successfully!")


if __name__ == "__main__":
    main()
