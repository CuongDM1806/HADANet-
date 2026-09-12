import unittest

import numpy as np
import torch

from hadanet import HADANet, HADANetLoss, HADANetRaw, HADANetV1, HADANetV2
from hadanet.physionet import TRIAL_SAMPLES, build_loso_fold
from train_physionet_loso import make_loaders


class HADANetArchitectureTest(unittest.TestCase):
    def test_physionet_trial_length_is_4p1_seconds(self):
        self.assertEqual(TRIAL_SAMPLES, 656)

    def test_raw_model_is_the_active_architecture(self):
        self.assertIs(HADANet, HADANetRaw)
        self.assertGreater(
            sum(parameter.numel() for parameter in HADANetV2().parameters()),
            sum(parameter.numel() for parameter in HADANetV1().parameters()),
        )

    def test_v2_front_end_preserves_expected_shapes(self):
        model = HADANetV2().eval()
        inputs = torch.randn(2, 64, 5, 4)
        with torch.no_grad():
            hcnn_features = model.hierarchical_cnn(inputs)
            attended_features = model.attention(hcnn_features)
        self.assertEqual(hcnn_features.shape, inputs.shape)
        self.assertEqual(attended_features.shape, inputs.shape)

    def test_complete_loss_is_finite_and_backpropagates(self):
        torch.manual_seed(1)
        model = HADANet()
        source = torch.randn(4, 64, TRIAL_SAMPLES)
        target = torch.randn(4, 64, TRIAL_SAMPLES)
        labels = torch.tensor([0, 1, 2, 3])
        outputs = model.forward_domains(source, target, alpha=0.5)
        losses = HADANetLoss()(model, outputs, labels)
        self.assertEqual(outputs["source_logits"].shape, (4, 4))
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        losses["total"].backward()
        self.assertIsNotNone(model.classifier.fc2.weight.grad)
        self.assertIsNotNone(model.domain_discriminator.network[0].weight.grad)

    def test_source_and_target_share_one_batchnorm_pass(self):
        torch.manual_seed(4)
        model = HADANet().train()
        source = torch.randn(4, 64, TRIAL_SAMPLES)
        target = torch.randn(4, 64, TRIAL_SAMPLES) + 2.0

        first_batch_norm = model.hierarchical_cnn.stem[1]
        self.assertEqual(first_batch_norm.num_batches_tracked.item(), 0)
        outputs = model.forward_domains(source, target, alpha=0.5)

        self.assertEqual(first_batch_norm.num_batches_tracked.item(), 1)
        self.assertEqual(outputs["source_features"].shape, (4, 1280))
        self.assertEqual(outputs["target_features"].shape, (4, 1280))

    def test_orthogonal_loss_uses_final_fc_input_dimension(self):
        model = HADANet()
        weight = model.classifier.fc2.weight
        gram = weight @ weight.transpose(0, 1)
        identity = torch.eye(gram.size(0))
        expected = (gram - identity).square().sum() / weight.size(1) ** 2
        torch.testing.assert_close(model.orthogonal_loss(), expected)

    def test_raw_front_end_requires_4p1_second_trials(self):
        model = HADANet()
        trials = torch.randn(2, 64, TRIAL_SAMPLES)
        self.assertEqual(model(trials).shape, (2, 4))
        with self.assertRaises(ValueError):
            model(trials[..., :-1])


class LOSOSplitTest(unittest.TestCase):
    def test_target_is_excluded_and_training_loader_has_no_target_labels(self):
        rng = np.random.default_rng(3)
        pool = {}
        for subject in (1, 2, 3):
            labels = np.tile(np.arange(4), 8)
            features = rng.standard_normal(
                (len(labels), 64, TRIAL_SAMPLES)
            ).astype(np.float32)
            pool[subject] = (features, labels)

        fold = build_loso_fold(pool, target_subject=2, seed=7)
        self.assertEqual(fold.source_subjects, (1, 3))
        self.assertNotIn(2, fold.source_subjects)
        self.assertEqual(len(fold.source_train_y), 56)
        self.assertEqual(len(fold.source_val_y), 8)
        self.assertEqual(len(fold.target_train_x), 32)
        self.assertEqual(len(fold.target_test_y), 32)
        np.testing.assert_array_equal(fold.target_train_x, pool[2][0])
        np.testing.assert_array_equal(fold.target_test_x, pool[2][0])
        source_loader, val_loader, target_loader, test_loader = make_loaders(
            fold, batch_size=4, num_workers=0
        )
        self.assertEqual(len(next(iter(source_loader))), 2)
        self.assertEqual(len(next(iter(val_loader))), 2)
        self.assertEqual(len(next(iter(target_loader))), 1)
        self.assertEqual(len(next(iter(test_loader))), 2)


if __name__ == "__main__":
    unittest.main()
