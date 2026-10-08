"""Tests for the dataset registry: approved vs excluded sources (no silent downloads)."""
from ml.data.download import APPROVED, EXCLUDED, RAW_DIR, MANIFEST_PATH


def test_approved_registry_is_exactly_the_v0_plan():
    assert set(APPROVED) == {"mage", "hc3", "raid"}
    for spec in APPROVED.values():
        assert spec["license"]
        assert spec["role"]
        assert spec["homepage"].startswith("https://huggingface.co/datasets/")
        assert spec["files"]


def test_excluded_datasets_are_never_downloadable():
    assert "m4" in EXCLUDED and "llm-detectaive" in EXCLUDED
    assert "m4" not in APPROVED
    assert "llm-detectaive" not in APPROVED


def test_raid_train_and_extra_files_are_not_downloaded():
    assert "train.csv" not in APPROVED["raid"]["files"]
    assert "extra.csv" not in APPROVED["raid"]["files"]
    assert "test.csv" in APPROVED["raid"]["files"]
    assert "not_downloaded" in APPROVED["raid"]  # recorded, not silent


def test_roles_match_training_policy():
    assert "training" in APPROVED["mage"]["role"] or "corpus" in APPROVED["mage"]["role"]
    assert "never-trained-on" in APPROVED["hc3"]["role"]
    assert "never-trained-on" in APPROVED["raid"]["role"]


def test_manifest_path_is_committed_location():
    # manifest lives in git-tracked dir; raw data does not
    assert "manifests" in str(MANIFEST_PATH)
    assert "raw" in str(RAW_DIR)
    assert MANIFEST_PATH != RAW_DIR
