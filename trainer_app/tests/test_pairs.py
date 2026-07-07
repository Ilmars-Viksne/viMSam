import pytest

np = pytest.importorskip("numpy")
imageio = pytest.importorskip("imageio.v3")

from vimsam_trainer.core.errors import InputValidationError
from vimsam_trainer.data.pairs import collect_image_mask_pairs


def test_collect_raw_mask_pairs_by_stem(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()

    (images / "a.raw").write_bytes(np.zeros((4, 4), dtype=np.uint16).tobytes())
    imageio.imwrite(masks / "a.tif", np.zeros((4, 4), dtype=np.uint8))

    pairs = collect_image_mask_pairs(
        images_path=images,
        masks_path=masks,
        workflow="raw_frames",
    )

    assert len(pairs) == 1
    assert pairs[0].image_path.name == "a.raw"
    assert pairs[0].mask_path.name == "a.tif"


def test_collect_pairs_rejects_duplicate_mask_stems(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()

    (images / "a.raw").write_bytes(b"")
    imageio.imwrite(masks / "a.tif", np.zeros((4, 4), dtype=np.uint8))
    imageio.imwrite(masks / "a.png", np.zeros((4, 4), dtype=np.uint8))

    with pytest.raises(InputValidationError):
        collect_image_mask_pairs(
            images_path=images,
            masks_path=masks,
            workflow="raw_frames",
        )
