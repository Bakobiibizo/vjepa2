#!/usr/bin/env python3
"""Download a V-JEPA checkpoint atomically, with optional SHA-256 verification."""
import argparse, hashlib, os, tempfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()

def download(url, destination, expected_sha256=None):
    if urlparse(url).scheme != "https": raise ValueError("checkpoint URL must use HTTPS")
    destination = destination.expanduser().resolve(); destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if expected_sha256 and sha256(destination) != expected_sha256.lower(): raise ValueError(f"existing file checksum mismatch: {destination}")
        return destination
    request = Request(url, headers={"User-Agent": "vjepa2-integration/1"})
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent); temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as target, urlopen(request, timeout=60) as response:
            while chunk := response.read(1024 * 1024): target.write(chunk)
        if expected_sha256 and sha256(temporary) != expected_sha256.lower(): raise ValueError("downloaded checkpoint checksum mismatch")
        temporary.replace(destination)
    finally: temporary.unlink(missing_ok=True)
    return destination

def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("url"); parser.add_argument("destination", type=Path); parser.add_argument("--sha256")
    args = parser.parse_args(); print(download(args.url, args.destination, args.sha256)); return 0
if __name__ == "__main__": raise SystemExit(main())
