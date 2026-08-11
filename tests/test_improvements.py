"""Invariants for the accuracy/methodology changes.

These target the failure modes that stay silent. Augmentation that desynchronizes an image from
its mask still trains; TTA that forgets to un-flip still produces a plausible-looking volume; a
guard that does not fire still writes a file. Each of those degrades a result without raising
anything, so the check has to be explicit.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fedbrats.config import PRESETS, Config, apply_preset
from fedbrats.data import _augment_intensity, _augment_spatial
from fedbrats.logging_utils import RunExistsError, guard_run_dir
from fedbrats.train import _probs_2d, _probs_3d, postprocess, remove_small_components


class PassThrough(torch.nn.Module):
    """Returns the first three input channels as logits.

    Exactly equivariant to flips, which is what makes it a probe: averaging flip views of an
    equivariant model must reproduce the un-augmented output. Any error in un-flipping shows up
    as a blur between a view and its mirror instead of an exception.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :3]


def gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# --------------------------------------------------------------------------------------
# LR schedule
# --------------------------------------------------------------------------------------

def test_constant_schedule_is_unchanged():
    """The baseline recipe must keep returning exactly cfg.lr, or old runs stop reproducing."""
    cfg = Config(rounds=25, lr=1e-3, lr_schedule="constant")
    assert all(cfg.round_lr(r) == 1e-3 for r in range(1, 26))


def test_cosine_spans_lr_to_floor_and_never_rises():
    cfg = Config(rounds=25, lr=1e-3, lr_schedule="cosine", lr_min_factor=0.05)
    lrs = [cfg.round_lr(r) for r in range(1, 26)]
    assert lrs[0] == pytest.approx(1e-3)
    assert lrs[-1] == pytest.approx(1e-3 * 0.05)
    assert all(a >= b for a, b in zip(lrs, lrs[1:])), "cosine schedule must be monotone decreasing"


def test_single_round_schedule_does_not_divide_by_zero():
    assert Config(rounds=1, lr_schedule="cosine", report_last_k=1).round_lr(1) == 1e-3


# --------------------------------------------------------------------------------------
# augmentation
# --------------------------------------------------------------------------------------

def test_spatial_augmentation_keeps_image_and_mask_aligned():
    """The one failure that silently destroys a segmentation dataset.

    Flipping the image but not its mask still trains, still converges, and still reports a Dice
    -- just a much worse one, for a reason no error message will ever mention.
    """
    cfg = Config(augment=True, aug_flip_p=1.0, aug_rot90_p=0.0)
    x = np.arange(4 * 6 * 6, dtype=np.float32).reshape(4, 6, 6)
    y = (x % 7 == 0).astype(np.float32)[:3]
    out = _augment_spatial(np.concatenate([x, y]), cfg, gen(), axes=(1, 2), rot_axes=(1, 2))

    expected = (out[:4] % 7 == 0).astype(np.float32)[:3]
    assert np.array_equal(out[4:], expected)


def test_flip_probability_zero_is_identity():
    cfg = Config(augment=True, aug_flip_p=0.0, aug_rot90_p=0.0)
    a = np.random.default_rng(0).normal(size=(7, 5, 5)).astype(np.float32)
    assert np.array_equal(_augment_spatial(a.copy(), cfg, gen(), axes=(1, 2)), a)


def test_rot90_preserves_voxel_multiset():
    cfg = Config(augment=True, aug_flip_p=0.0, aug_rot90_p=1.0)
    a = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    out = _augment_spatial(a.copy(), cfg, gen(3), axes=(), rot_axes=(1, 2))
    assert out.shape == a.shape
    assert np.array_equal(np.sort(out.ravel()), np.sort(a.ravel()))


def test_intensity_augmentation_preserves_exact_zero_background():
    """Preprocessing writes background as exactly 0.0 and evaluation relies on it.

    A shift or a noise draw that lifts background off zero trains the model on inputs it will
    never be evaluated on -- augmentation manufacturing a train/test gap instead of closing one.
    """
    cfg = Config(augment=True, aug_intensity_p=1.0, aug_intensity_shift=0.5, aug_noise_std=0.1)
    x = np.zeros((4, 8, 8), dtype=np.float32)
    x[:, 2:6, 2:6] = 1.5                                   # a "brain"; everything else background

    out = _augment_intensity(x.copy(), cfg, gen())
    background = np.ones((8, 8), dtype=bool)
    background[2:6, 2:6] = False
    assert np.all(out[:, background] == 0.0)
    assert not np.allclose(out[:, 2:6, 2:6], x[:, 2:6, 2:6]), "foreground should have changed"


def test_intensity_augmentation_is_off_by_default():
    cfg = Config()                                          # baseline recipe
    x = np.random.default_rng(1).normal(size=(4, 5, 5)).astype(np.float32)
    assert np.array_equal(_augment_intensity(x.copy(), cfg, gen()), x)


# --------------------------------------------------------------------------------------
# post-processing
# --------------------------------------------------------------------------------------

def test_small_components_removed_and_large_kept():
    mask = np.zeros((20, 20, 20), dtype=bool)
    mask[2:12, 2:12, 2:12] = True                           # 1000 voxels
    mask[18, 18, 18] = True                                 # 1 voxel, disconnected
    out = remove_small_components(mask, min_voxels=50)
    assert out[2:12, 2:12, 2:12].all()
    assert not out[18, 18, 18]


def test_filter_never_empties_a_nonempty_mask():
    """An empty prediction scores 0.0 against non-empty truth, so erasing everything costs more
    than the false positives it removes. The largest component survives regardless."""
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[5, 5, 5] = True                                    # smaller than any sane threshold
    assert remove_small_components(mask, min_voxels=1000).sum() == 1


def test_zero_threshold_is_a_no_op():
    mask = np.random.default_rng(0).random((8, 8, 8)) > 0.5
    assert np.array_equal(remove_small_components(mask, 0), mask)


def test_postprocess_restores_region_nesting():
    """WT >= TC >= ET is what the BraTS regions mean, but three independent sigmoids do not
    enforce it and filtering each channel separately can break it."""
    cfg = Config(postproc_min_voxels=10)
    pred = np.zeros((3, 12, 12, 12), dtype=bool)
    pred[0, 0:4, 0:4, 0:4] = True                           # WT here
    pred[1, 6:10, 6:10, 6:10] = True                        # TC somewhere else entirely
    pred[2, 6:10, 6:10, 6:10] = True
    out = postprocess(pred, cfg)
    assert not (out[1] & ~out[0]).any(), "TC must lie inside WT"
    assert not (out[2] & ~out[1]).any(), "ET must lie inside TC"


# --------------------------------------------------------------------------------------
# test-time augmentation
# --------------------------------------------------------------------------------------

def test_tta_2d_undoes_its_own_flips():
    """Averaging flip views of a flip-equivariant model must equal the un-augmented output.

    Forget to flip the prediction back and each view lands mirrored; the mean is then a blur of
    the volume with itself, which still has the right shape, the right range, and a plausible
    Dice. Nothing downstream would notice.
    """
    x = np.random.default_rng(0).normal(size=(3, 4, 16, 16)).astype(np.float32)
    model, dev = PassThrough(), torch.device("cpu")
    plain = _probs_2d(model, x, dev, tta=False)
    tta = _probs_2d(model, x, dev, tta=True)
    assert tta.shape == plain.shape
    assert np.allclose(tta, plain, atol=1e-6)


def test_tta_2d_on_an_asymmetric_input_is_not_accidentally_symmetric():
    """Guards the guard: a symmetric input would satisfy the test above even if un-flipping
    were broken, so confirm the probe input actually distinguishes the two."""
    x = np.zeros((1, 4, 8, 8), dtype=np.float32)
    x[0, :, 0, 0] = 5.0                                     # a single asymmetric corner
    out = _probs_2d(PassThrough(), x, torch.device("cpu"), tta=True)
    assert out[0, 0, 0, 0] > out[0, 0, -1, -1], "un-flipping lost the input's orientation"


def test_tta_3d_undoes_its_own_flips():
    x = np.random.default_rng(1).normal(size=(1, 4, 32, 32, 32)).astype(np.float32)
    xt = torch.from_numpy(x)
    cfg = Config(dim="3d", patch_size=16, sw_overlap=0.25)
    plain = _probs_3d(PassThrough(), xt, cfg, tta=False)
    tta = _probs_3d(PassThrough(), xt, cfg, tta=True)
    assert tta.shape == plain.shape
    assert torch.allclose(tta, plain, atol=1e-5)


# --------------------------------------------------------------------------------------
# run guard + config plumbing
# --------------------------------------------------------------------------------------

def test_guard_blocks_writing_into_an_existing_run(tmp_path):
    (tmp_path / "metrics.jsonl").write_text('{"round": 1}\n')
    with pytest.raises(RunExistsError, match="--tag"):
        guard_run_dir(tmp_path)


def test_guard_allows_empty_dir_and_explicit_overwrite(tmp_path):
    guard_run_dir(tmp_path)                                 # nothing there yet
    (tmp_path / "metrics.jsonl").write_text('{"round": 1}\n')
    guard_run_dir(tmp_path, overwrite=True)                 # explicitly asked for


def test_tag_namespaces_the_run_directory():
    plain, tagged = Config(), Config(tag="v2")
    assert plain.run_dir("fedavg").name == "fedavg_2d_42"
    assert tagged.run_dir("fedavg").parent.name == "v2"
    assert tagged.run_dir("fedavg") != plain.run_dir("fedavg")
    assert tagged.run_id("fedavg") == plain.run_id("fedavg"), "run_id must stay seed-parseable"


def test_baseline_preset_matches_bare_defaults():
    """`--preset baseline` has to stay identical to no preset, or the frozen numbers stop
    being reproducible by the documented command."""
    assert apply_preset("baseline").to_dict() == Config().to_dict()


def test_3d_records_no_rotation_even_when_the_preset_asks_for_it():
    """PatchDataset never rotates, so a non-zero aug_rot90_p in a 3D config.json would describe
    a run that did not happen. 2D must keep the preset's value."""
    assert apply_preset("v2", {"dim": "3d"}).aug_rot90_p == 0.0
    assert apply_preset("v2", {"dim": "2d"}).aug_rot90_p > 0.0


def test_v2_preset_sets_every_documented_lever():
    cfg = apply_preset("v2")
    assert (cfg.lr_schedule, cfg.augment, cfg.tta) == ("cosine", True, True)
    assert cfg.postproc_min_voxels > 0 and cfg.val_per_hospital > 0
    assert cfg.select_by == "best_val" and cfg.report_last_k > 1


def test_explicit_overrides_beat_the_preset():
    assert apply_preset("v2", {"augment": False, "rounds": 3, "report_last_k": 2}).augment is False


def test_to_dict_is_json_typed_not_stringified():
    d = Config(rounds=25).to_dict()
    assert d["rounds"] == 25 and d["augment"] is False
    assert isinstance(d["paths"], dict)


@pytest.mark.parametrize("kwargs", [
    {"lr_schedule": "linear"},
    {"select_by": "best"},
    {"select_by": "best_val", "val_per_hospital": 0},       # selection with nothing to select on
    {"rounds": 3, "report_last_k": 5},                      # averaging more rounds than exist
])
def test_invalid_configs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_presets_only_reference_real_config_fields():
    """A typo in a preset key would otherwise surface as an unrelated TypeError at run time."""
    fields = set(vars(Config()))
    for name, preset in PRESETS.items():
        assert set(preset) <= fields, f"preset {name!r} sets unknown fields"
