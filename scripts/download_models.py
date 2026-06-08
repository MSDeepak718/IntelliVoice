#!/usr/bin/env python3
"""
IntelliVoice — Model Download Script

Downloads all required models from HuggingFace Hub.
Run this before starting the application for the first time.

Usage:
    python scripts/download_models.py              # Download all HF models
    python scripts/download_models.py --model bge_m3
    python scripts/download_models.py --status      # Check download status
    python scripts/download_models.py --budget      # Show VRAM budget
    python scripts/download_models.py --list        # List all registered models
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings
from config.model_registry import ModelRegistry, LoadingOrder
from backend.services.model_loader import ModelLoader


def main():
    parser = argparse.ArgumentParser(description="IntelliVoice Model Downloader")
    parser.add_argument("--model", type=str, help="Download a specific model by name")
    parser.add_argument("--status", action="store_true", help="Show download status")
    parser.add_argument("--budget", action="store_true", help="Show VRAM budget breakdown")
    parser.add_argument("--list", action="store_true", help="List all registered models")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()

    loader = ModelLoader()

    if args.budget:
        ModelRegistry.print_vram_budget()
        return

    if args.list:
        print("\n=== Registered models ===\n")
        for m in ModelRegistry.get_all_models():
            print(
                f"  {m.name:<14} "
                f"order={m.order.name:<13} "
                f"precision={m.precision.value:<5} "
                f"vram≈{m.estimated_vram_mb:>5}MB  "
                f"{m.hf_model_id}"
            )
        return

    if args.status:
        print("\n=== Model Download Status ===\n")
        status = loader.get_download_status()
        for name, info in status.items():
            icon = "[OK]" if info["downloaded"] else "[--]"
            vram = f"{info['vram_mb']}MB"
            print(f"  {icon} {name:<16} {vram:<10} {info['hf_id']}")
            if info["downloaded"]:
                print(f"        -> {info['path']}")
        return

    if args.model:
        model_map = {m.name: m for m in ModelRegistry.get_all_models()}
        if args.model not in model_map:
            print(f"Unknown model: {args.model}")
            print(f"Available: {', '.join(model_map.keys())}")
            sys.exit(1)
        cfg = model_map[args.model]
        print(f"\nDownloading {cfg.name} ({cfg.hf_model_id})...")
        loader.download_model(cfg, force=args.force)
        print("Done!")
        return

    # Download all
    print("\n=== IntelliVoice Model Downloader ===\n")
    ModelRegistry.print_vram_budget()
    print("\nStarting downloads...\n")
    results = loader.download_all_models(force=args.force)
    print("\n=== Download Results ===\n")
    for name, result in results.items():
        icon = "[OK]" if result["status"] == "ok" else "[ERR]"
        print(f"  {icon} {name}: {result.get('path', result.get('error', ''))}")
    errors = [n for n, r in results.items() if r["status"] != "ok"]
    if errors:
        print(f"\n[!] {len(errors)} model(s) failed: {', '.join(errors)}")
        sys.exit(1)
    print("\n[OK] All models downloaded successfully!")


if __name__ == "__main__":
    main()
