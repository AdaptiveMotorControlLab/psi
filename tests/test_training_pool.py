"""Pool controls preserve record alignment and leave held-out data untouched."""
from types import SimpleNamespace

import torch

from psi.data.module import PSIDataModule


def test_base_pool_uses_provenance_not_pairs_or_targets(tmp_path):
    (tmp_path / "train").mkdir()
    # Base 1 is ineligible; duplicate base IDs must not duplicate observations.
    torch.save({"base_index": torch.tensor([0, 0, 1, 4, 4])}, tmp_path / "train/pairs.pt")
    correct = torch.tensor([0, 2, 3, 4, 5])
    records = {"activation": correct[:, None].float(), "state_features": correct[:, None].float() + 10}
    cache = SimpleNamespace(root=tmp_path, records=lambda split: records,
                            outcomes=lambda split: SimpleNamespace(correct_indices=correct))
    for pool in ("base_only", "matched_all"):
        module = PSIDataModule({"training_pool": pool, "seed": 11}, cache)
        module.setup("fit")
        selected = module.train_dataset.records
        assert len(selected["activation"]) == 2
        torch.testing.assert_close(selected["state_features"], selected["activation"] + 10)
        assert module.val_dataset.records["activation"] is records["activation"]
        if pool == "base_only":
            assert selected["activation"].flatten().tolist() == [0, 4]
        before = selected["activation"].clone()
        module.setup("fit")
        torch.testing.assert_close(before, module.train_dataset.records["activation"])
