import unittest

import numpy as np
import torch

from hadanet.model import HADANet, HADANetLoss
from hadanet.physionet import build_loso_fold, differential_entropy_features
from train_physionet_loso import make_loaders


class HADANetArchitectureTest(unittest.TestCase):
    def test_complete_loss_is_finite_and_backpropagates(self):
        torch.manual_seed(1)
        model = HADANet()
        source = torch.randn(4, 64, 5, 4)
        target = torch.randn(4, 64, 5, 4)
        labels = torch.tensor([0, 1, 2, 3])
        outputs = model.forward_domains(source, target, alpha=0.5)
        losses = HADANetLoss()(model, outputs, labels)
        self.assertEqual(outputs["source_logits"].shape, (4, 4))
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        losses["total"].backward()
        self.assertIsNotNone(model.classifier.fc2.weight.grad)
        self.assertIsNotNone(model.domain_discriminator.network[0].weight.grad)

    def test_de_feature_shape(self):
        rng = np.random.default_rng(2)
        trials = rng.standard_normal((2, 64, 640)).astype(np.float32)
        features = differential_entropy_features(trials)
        self.assertEqual(features.shape, (2, 64, 5, 4))
        self.assertTrue(np.isfinite(features).all())


class LOSOSplitTest(unittest.TestCase):
    def test_target_is_excluded_and_training_loader_has_no_target_labels(self):
        rng = np.random.default_rng(3)
        pool = {}
        for subject in (1, 2, 3):
            labels = np.tile(np.arange(4), 8)
            features = rng.standard_normal((len(labels), 64, 5, 4)).astype(np.float32)
            pool[subject] = (features, labels)

        fold = build_loso_fold(pool, target_subject=2, seed=7)
        self.assertEqual(fold.source_subjects, (1, 3))
        self.assertNotIn(2, fold.source_subjects)
        source_loader, val_loader, target_loader, test_loader = make_loaders(
            fold, batch_size=4, num_workers=0
        )
        self.assertEqual(len(next(iter(source_loader))), 2)
        self.assertEqual(len(next(iter(val_loader))), 2)
        self.assertEqual(len(next(iter(target_loader))), 1)
        self.assertEqual(len(next(iter(test_loader))), 2)


if __name__ == "__main__":
    unittest.main()
