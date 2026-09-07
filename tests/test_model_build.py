"""Model construction: the supported forecasting backbones and the RSSD layer."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rssd.models.lstm import Seq2SeqLSTM                                # noqa: E402


def build(**overrides):
    kwargs = dict(input_dim=4, hidden_dim=8, output_dim=1, num_layers=1, pred_len=7)
    kwargs.update(overrides)
    return Seq2SeqLSTM(**kwargs)


class BackboneTests(unittest.TestCase):
    def test_recurrent_backbone_forecasts_every_lead_day_in_one_pass(self):
        model = build()
        self.assertIsInstance(model.encoder, torch.nn.LSTM)
        self.assertIsNone(model.decoder)
        # one head, seven lead days
        self.assertEqual(tuple(model.fc2_direct.weight.shape), (7, 8))

    def test_transformer_backbone_builds_a_full_encoder_decoder(self):
        model = build(backbone="transformer_seq2seq", n_heads=2, tin=30)
        self.assertIsInstance(model.encoder, torch.nn.TransformerEncoder)
        self.assertIsInstance(model.decoder, torch.nn.TransformerDecoder)
        # one positional query per lead day, so every horizon attends to the full history
        self.assertEqual(tuple(model.pos_tgt.weight.shape), (7, 8))

    def test_unknown_backbone_is_rejected(self):
        with self.assertRaises(ValueError):
            build(backbone="gru")

    def test_transformer_hidden_dim_must_divide_by_heads(self):
        with self.assertRaises(ValueError):
            build(backbone="transformer_seq2seq", n_heads=3, tin=30)


class RssdLayerTests(unittest.TestCase):
    """The decomposition the method is named after: h_dyn -> h_shr + h_spc -> h_rec."""

    def test_the_two_components_sum_back_to_the_dynamic_state(self):
        torch.manual_seed(0)
        model = build(use_darsd=True, lcib_k=3)
        h_dyn = torch.randn(5, 8)
        h_shr, h_spc, _assignments = model._lcib_decompose(h_dyn)
        torch.testing.assert_close(h_shr + h_spc, h_dyn, rtol=1e-5, atol=1e-6)

    def test_recomposition_attenuates_only_the_site_specific_component(self):
        torch.manual_seed(0)
        model = build(use_darsd=True, lcib_k=3)
        h_dyn = torch.randn(5, 8)
        h_shr, h_spc, _ = model._lcib_decompose(h_dyn)
        gate = torch.sigmoid(model.lcib_gate)
        # h_rec = h_shr + (1 - g) h_spc, the form the method is described in
        torch.testing.assert_close(model._lcib_forward(h_dyn), h_shr + (1.0 - gate) * h_spc,
                                   rtol=1e-5, atol=1e-6)

    def test_assignment_weights_form_a_distribution_over_the_shared_basis(self):
        torch.manual_seed(0)
        model = build(use_darsd=True, lcib_k=3)
        _, _, assignments = model._lcib_decompose(torch.randn(5, 8))
        self.assertEqual(tuple(assignments.shape), (5, 3))
        torch.testing.assert_close(assignments.sum(dim=-1), torch.ones(5), rtol=1e-5, atol=1e-6)
        self.assertTrue(bool((assignments >= 0).all()))


if __name__ == "__main__":
    unittest.main()
