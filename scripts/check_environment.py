#!/usr/bin/env python3
"""Report whether a host can run V-JEPA 2 without loading a checkpoint."""
import argparse, importlib.util, json, platform, sys

REQUIRED_MODULES = ("torch", "torchvision", "yaml", "numpy")

def inspect_environment():
    modules = {name: importlib.util.find_spec(name) is not None for name in REQUIRED_MODULES}
    report = {"python": platform.python_version(), "python_supported": sys.version_info >= (3, 11), "architecture": platform.machine(), "platform": platform.system(), "modules": modules, "ready": sys.version_info >= (3, 11) and all(modules.values()), "accelerator": {"kind": "none", "available": False}}
    if modules["torch"]:
        import torch
        if torch.cuda.is_available():
            report["accelerator"] = {"kind": "cuda", "available": True, "device_count": torch.cuda.device_count(), "device": torch.cuda.get_device_name(0)}
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            report["accelerator"] = {"kind": "mps", "available": True}
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--require-accelerator", action="store_true")
    args = parser.parse_args(); report = inspect_environment()
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else f"Python {report['python']} on {report['architecture']}\nDependencies ready: {report['ready']}\nAccelerator: {report['accelerator']}")
    return 1 if not report["ready"] else (2 if args.require_accelerator and not report["accelerator"]["available"] else 0)

if __name__ == "__main__": raise SystemExit(main())
