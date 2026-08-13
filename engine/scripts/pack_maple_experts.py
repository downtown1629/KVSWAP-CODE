#!/usr/bin/env python3
"""Pack or verify the BF16 expert bank used by the Maple engine adapter."""

import argparse
import json

from expert_store import (
    ExpertStore,
    pack_bf16_expert_store,
    qwen3_checkpoint_digest,
    qwen3_fixed_checkpoint_digest,
)
from model_config import get_model_config
from moe_weights import SafetensorCheckpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("pack")
    pack.add_argument("checkpoint")
    pack.add_argument("output")
    pack.add_argument("--source-revision", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("checkpoint")
    verify.add_argument("store")
    verify.add_argument("--source-revision", required=True)
    args = parser.parse_args()

    config = get_model_config(args.checkpoint)
    if config.model_type != "maple":
        raise ValueError("pack_maple_experts.py requires model_type=maple")
    checkpoint = SafetensorCheckpoint(args.checkpoint)
    if args.command == "pack":
        store = pack_bf16_expert_store(
            checkpoint,
            config,
            args.output,
            source_revision=args.source_revision,
        )
        result = {"extents": len(store.extents), "store": str(store.root)}
    else:
        store = ExpertStore(
            args.store,
            config=config,
            expected_source_revision=args.source_revision,
            expected_fixed_checkpoint_digest=qwen3_fixed_checkpoint_digest(
                checkpoint, config
            ),
        )
        if store.checkpoint_digest != qwen3_checkpoint_digest(checkpoint, config):
            raise ValueError("Maple store full checkpoint digest mismatch")
        result = {"verified_extents": store.verify_checksums(), "store": str(store.root)}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
