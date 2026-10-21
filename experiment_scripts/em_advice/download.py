"""Fetch only the three advice files from the authors' protected data release.

Install the authors' unpacker first: uv pip install easy-dataset-share.
The public extraction password and canary-removal procedure are in their README.
"""
import argparse
import hashlib
import subprocess
import urllib.request
from pathlib import Path

from run import ARCHIVE_SHA256, FILES, SOURCE_REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/em_advice"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    archive = args.out / "training_datasets.zip.enc"
    url = ("https://raw.githubusercontent.com/clarifying-EM/model-organisms-for-EM/"
           f"{SOURCE_REVISION}/em_organism_dir/data/training_datasets.zip.enc")
    if not archive.exists():
        urllib.request.urlretrieve(url, archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Released archive checksum mismatch")
    subprocess.run(["easy-dataset-share", "unprotect-dir", str(archive),
                    "-p", "model-organisms-em-datasets", "--remove-canaries", "--no-gitignore"], check=True)
    for filename in FILES.values():
        source = Path(str(archive) + ".extracted") / filename
        target = args.out / filename
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError(f"Refusing to overwrite different source: {target}")
        target.write_bytes(source.read_bytes())
    print("Prepared three advice files; data is ignored by Git.")


if __name__ == "__main__":
    main()
