#!/usr/bin/env python3
"""Pack or verify a lossless aligned Qwen3-MoE expert store."""

import argparse
import json

from expert_store import (
    ExpertStore,
    pack_qwen3_expert_store,
    qwen3_checkpoint_digest,
)
from model_config import get_model_config
from moe_weights import SafetensorCheckpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pack = subparsers.add_parser("pack")
    pack.add_argument("checkpoint")
    pack.add_argument("output")
    pack.add_argument("--source-revision", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("checkpoint")
    verify.add_argument("store")
    verify.add_argument("--source-revision", required=True)
    args = parser.parse_args()

    config = get_model_config(args.checkpoint)
    if args.command == "pack":
        store = pack_qwen3_expert_store(
            SafetensorCheckpoint(args.checkpoint),
            config,
            args.output,
            source_revision=args.source_revision,
        )
        print(json.dumps({"extents": len(store.extents), "store": str(store.root)}))
    else:
        checkpoint = SafetensorCheckpoint(args.checkpoint)
        store = ExpertStore(
            args.store,
            config=config,
            expected_source_revision=args.source_revision,
            expected_checkpoint_digest=qwen3_checkpoint_digest(
                checkpoint, config
            ),
        )
        count = store.verify_checksums()
        print(json.dumps({"verified_extents": count, "store": str(store.root)}))


if __name__ == "__main__":
    main()
