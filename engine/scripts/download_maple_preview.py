#!/usr/bin/env python3
"""Download the reviewed Maple-Preview revision without loading model weights."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


REVISION = "ac1ddd79d2b5cb4406f5d2bebdf95406ce505a07"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    args = parser.parse_args()
    path = snapshot_download(
        repo_id="deepgrove/maple-preview",
        revision=REVISION,
        local_dir=args.output,
        max_workers=4,
    )
    Path(path, ".kvswap_revision").write_text(REVISION + "\n")
    print(path)


if __name__ == "__main__":
    main()
