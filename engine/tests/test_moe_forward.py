import unittest

import torch
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeSparseMoeBlock,
)

from moe import (
    ResidentExpertProvider,
    qwen3_moe_forward,
    qwen3_moe_layer_forward,
    qwen3_moe_layer_forward_chunked,
    rms_norm_reference,
)


class Qwen3MoeForwardTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        config = Qwen3MoeConfig(
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=12,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_experts=4,
            num_experts_per_tok=2,
            norm_topk_prob=True,
        )
        self.reference = Qwen3MoeSparseMoeBlock(config).float().eval()
        self.hidden = torch.randn(2, 5, config.hidden_size)
        self.gate_proj = torch.stack(
            [expert.gate_proj.weight.detach() for expert in self.reference.experts]
        )
        self.up_proj = torch.stack(
            [expert.up_proj.weight.detach() for expert in self.reference.experts]
        )
        self.down_proj = torch.stack(
            [expert.down_proj.weight.detach() for expert in self.reference.experts]
        )
        self.provider = ResidentExpertProvider(
            self.gate_proj, self.up_proj, self.down_proj
        )

    def test_resident_provider_returns_original_bank(self):
        selected = torch.tensor([[0, 3], [1, 0]])
        materialized = self.provider.materialize(selected)
        self.assertEqual(materialized.gate_proj.data_ptr(), self.gate_proj.data_ptr())
        self.assertEqual(materialized.up_proj.data_ptr(), self.up_proj.data_ptr())
        self.assertEqual(materialized.down_proj.data_ptr(), self.down_proj.data_ptr())

    def test_resident_provider_rejects_invalid_expert(self):
        with self.assertRaises(IndexError):
            self.provider.materialize(torch.tensor([[0, 4]]))

    def test_forward_matches_transformers_reference(self):
        with torch.inference_mode():
            expected, expected_logits = self.reference(self.hidden)
            actual, routing = qwen3_moe_forward(
                self.hidden,
                self.reference.gate.weight,
                self.provider,
                top_k=self.reference.top_k,
                norm_topk_prob=self.reference.norm_topk_prob,
            )

        torch.testing.assert_close(routing.router_logits, expected_logits)
        torch.testing.assert_close(actual, expected)

        expected_weights = torch.softmax(expected_logits, dim=-1, dtype=torch.float32)
        expected_weights, expected_ids = torch.topk(
            expected_weights, self.reference.top_k, dim=-1
        )
        expected_weights /= expected_weights.sum(dim=-1, keepdim=True)
        self.assertTrue(torch.equal(routing.topk_ids, expected_ids))
        torch.testing.assert_close(routing.topk_weights, expected_weights)

    def test_layer_wrapper_matches_reference_norm_and_residual(self):
        norm_weight = torch.randn(self.hidden.shape[-1])
        eps = 1e-6
        with torch.inference_mode():
            normalized = rms_norm_reference(self.hidden, norm_weight, eps)
            expected_moe, _ = self.reference(normalized)
            actual, _ = qwen3_moe_layer_forward(
                self.hidden,
                norm_weight,
                self.reference.gate.weight,
                self.provider,
                top_k=self.reference.top_k,
                norm_topk_prob=self.reference.norm_topk_prob,
                rms_norm_eps=eps,
            )
        torch.testing.assert_close(actual, self.hidden + expected_moe)

    def test_reference_matrix_covers_topk_batch_and_decode_shapes(self):
        for shape, top_k, normalize in (
            ((2, 1, 16), 1, False),
            ((2, 3, 16), 2, True),
            ((1, 7, 16), 2, False),
        ):
            with self.subTest(shape=shape, top_k=top_k, normalize=normalize):
                config = Qwen3MoeConfig(
                    hidden_size=16,
                    intermediate_size=32,
                    moe_intermediate_size=12,
                    num_hidden_layers=1,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    num_experts=4,
                    num_experts_per_tok=top_k,
                    norm_topk_prob=normalize,
                )
                reference = Qwen3MoeSparseMoeBlock(config).float().eval()
                hidden = torch.randn(*shape)
                provider = ResidentExpertProvider(
                    torch.stack(
                        [expert.gate_proj.weight.detach() for expert in reference.experts]
                    ),
                    torch.stack(
                        [expert.up_proj.weight.detach() for expert in reference.experts]
                    ),
                    torch.stack(
                        [expert.down_proj.weight.detach() for expert in reference.experts]
                    ),
                )
                with torch.inference_mode():
                    expected, expected_logits = reference(hidden)
                    actual, routing = qwen3_moe_forward(
                        hidden,
                        reference.gate.weight,
                        provider,
                        top_k=top_k,
                        norm_topk_prob=normalize,
                    )
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(routing.router_logits, expected_logits)

    def test_duplicate_routing_and_unused_experts_are_deterministic(self):
        repeated_hidden = self.hidden[:1, :1].expand(2, 4, -1).clone()
        first, first_routing = qwen3_moe_forward(
            repeated_hidden,
            self.reference.gate.weight,
            self.provider,
            top_k=2,
            norm_topk_prob=True,
        )
        second, second_routing = qwen3_moe_forward(
            repeated_hidden,
            self.reference.gate.weight,
            self.provider,
            top_k=2,
            norm_topk_prob=True,
        )
        self.assertLess(torch.unique(first_routing.topk_ids).numel(), 4)
        self.assertTrue(torch.equal(first_routing.topk_ids, second_routing.topk_ids))
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_chunked_layer_matches_unchunked(self):
        norm_weight = torch.randn(self.hidden.shape[-1])
        unchunked, unchunked_routing = qwen3_moe_layer_forward(
            self.hidden,
            norm_weight,
            self.reference.gate.weight,
            self.provider,
            top_k=2,
            norm_topk_prob=True,
            rms_norm_eps=1e-6,
        )
        chunked, chunked_routing = qwen3_moe_layer_forward_chunked(
            self.hidden,
            norm_weight,
            self.reference.gate.weight,
            self.provider,
            top_k=2,
            norm_topk_prob=True,
            rms_norm_eps=1e-6,
            token_chunk_size=3,
        )
        # GEMM batch shape can change the final FP32 rounding even though each
        # token is independent. Routing IDs remain the exact semantic gate.
        torch.testing.assert_close(chunked, unchunked, rtol=1e-6, atol=1e-7)
        self.assertTrue(
            torch.equal(chunked_routing.topk_ids, unchunked_routing.topk_ids)
        )
        torch.testing.assert_close(
            chunked_routing.topk_weights,
            unchunked_routing.topk_weights,
            rtol=1e-6,
            atol=1e-7,
        )


if __name__ == "__main__":
    unittest.main()
