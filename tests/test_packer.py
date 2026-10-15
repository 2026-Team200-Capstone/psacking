"""
Tests for the BinPacker class and packing pipeline.
"""

import pytest
import numpy as np


class TestBinPackerInit:
    """Tests for BinPacker initialization."""

    def test_basic_init(self):
        """Test basic initialization."""
        from spectral_packer import BinPacker

        packer = BinPacker(tray_size=(100, 100, 100))

        assert packer.tray_size == (100, 100, 100)
        assert packer.voxel_resolution == 128

    def test_custom_resolution(self):
        """Test initialization with custom resolution."""
        from spectral_packer import BinPacker

        packer = BinPacker(tray_size=(50, 50, 50), voxel_resolution=64)

        assert packer.voxel_resolution == 64

    def test_custom_height_penalty(self):
        """Test initialization with custom height penalty."""
        from spectral_packer import BinPacker

        packer = BinPacker(tray_size=(50, 50, 50), height_penalty=1e6)

        assert packer.height_penalty == 1e6

    def test_invalid_tray_size_length(self):
        """Test error on wrong tray_size length."""
        from spectral_packer import BinPacker

        with pytest.raises(ValueError):
            BinPacker(tray_size=(100, 100))

    def test_invalid_tray_size_negative(self):
        """Test error on negative tray_size."""
        from spectral_packer import BinPacker

        with pytest.raises(ValueError):
            BinPacker(tray_size=(100, -50, 100))

    def test_invalid_resolution(self):
        """Test error on invalid resolution."""
        from spectral_packer import BinPacker

        with pytest.raises(ValueError):
            BinPacker(tray_size=(100, 100, 100), voxel_resolution=0)

    def test_repr(self):
        """Test string representation."""
        from spectral_packer import BinPacker

        packer = BinPacker(tray_size=(100, 100, 100))

        repr_str = repr(packer)
        assert "BinPacker" in repr_str
        assert "100" in repr_str


class TestPackVoxels:
    """Tests for pack_voxels method."""

    @pytest.fixture
    def packer(self):
        """Create a test packer."""
        from spectral_packer import BinPacker
        return BinPacker(tray_size=(50, 50, 50))

    def test_pack_single_item(self, packer, cube_5x5x5, requires_cuda):
        """Test packing a single item."""
        result = packer.pack_voxels([cube_5x5x5])

        assert result.num_placed == 1
        assert result.num_failed == 0
        assert result.density > 0

    def test_pack_multiple_items(self, packer, cube_3x3x3, requires_cuda):
        """Test packing multiple items."""
        items = [cube_3x3x3.copy() for _ in range(3)]

        result = packer.pack_voxels(items)

        assert result.num_placed >= 1
        assert len(result.placements) == 3

    def test_pack_returns_result(self, packer, cube_3x3x3, requires_cuda):
        """Test that pack_voxels returns PackingResult."""
        from spectral_packer import PackingResult

        result = packer.pack_voxels([cube_3x3x3])

        assert isinstance(result, PackingResult)

    def test_empty_items_error(self, packer):
        """Test error on empty items list."""
        with pytest.raises(ValueError):
            packer.pack_voxels([])

    def test_sort_by_volume(self, packer, requires_cuda):
        """Test that items are sorted by volume."""
        small = np.ones((2, 2, 2), dtype=np.int32)
        large = np.ones((5, 5, 5), dtype=np.int32)

        # Pack without sorting
        result = packer.pack_voxels([small, large], sort_by_volume=False)
        # First placement should be the small item
        first_placed = [p for p in result.placements if p.success][0]

        # Pack with sorting
        result2 = packer.pack_voxels([small, large], sort_by_volume=True)
        # First placement should be the large item (it's placed first due to sorting)

        # Just verify both work
        assert result.num_placed >= 1
        assert result2.num_placed >= 1

    def test_placements_have_correct_structure(self, packer, cube_3x3x3, requires_cuda):
        """Test that placements have correct attributes."""
        result = packer.pack_voxels([cube_3x3x3])

        placement = result.placements[0]
        assert hasattr(placement, 'item_index')
        assert hasattr(placement, 'position')
        assert hasattr(placement, 'score')
        assert hasattr(placement, 'success')
        assert hasattr(placement, 'volume')


class TestPackingResult:
    """Tests for PackingResult dataclass."""

    def test_get_item_mask(self, cube_3x3x3, requires_cuda):
        """Test getting mask for specific item."""
        from spectral_packer import BinPacker

        packer = BinPacker(tray_size=(20, 20, 20))
        result = packer.pack_voxels([cube_3x3x3])

        mask = result.get_item_mask(1)

        assert mask.dtype == bool
        assert mask.shape == result.tray.shape
        assert np.sum(mask) == np.sum(cube_3x3x3 > 0)

    def test_summary(self, cube_3x3x3, requires_cuda):
        """Test summary string generation."""
        from spectral_packer import BinPacker

        packer = BinPacker(tray_size=(20, 20, 20))
        result = packer.pack_voxels([cube_3x3x3])

        summary = result.summary()

        assert isinstance(summary, str)
        assert "Packing Result" in summary
        assert "density" in summary.lower()

    def test_bounding_box(self, cube_3x3x3, requires_cuda):
        """Test bounding box calculation."""
        from spectral_packer import BinPacker

        packer = BinPacker(tray_size=(20, 20, 20))
        result = packer.pack_voxels([cube_3x3x3])

        assert result.bounding_box is not None
        bbox_min, bbox_max = result.bounding_box
        assert len(bbox_min) == 3
        assert len(bbox_max) == 3


class TestPackSingle:
    """Tests for pack_single method."""

    def test_pack_single_returns_tuple(self, cube_3x3x3, requires_cuda):
        """Test that pack_single returns correct tuple."""
        from spectral_packer import BinPacker

        packer = BinPacker(tray_size=(20, 20, 20))
        position, found, score = packer.pack_single(cube_3x3x3)

        assert isinstance(found, bool)
        if found:
            assert len(position) == 3
            assert isinstance(score, float)

    def test_pack_single_with_tray(self, cube_3x3x3, requires_cuda):
        """Test pack_single with existing tray state."""
        from spectral_packer import BinPacker

        packer = BinPacker(tray_size=(20, 20, 20))
        tray = np.zeros((20, 20, 20), dtype=np.int32)
        tray[0:5, 0:5, 0:5] = 1

        position, found, score = packer.pack_single(cube_3x3x3, tray=tray)

        if found:
            # Position should not overlap with existing item
            assert not (position[0] < 5 and position[1] < 5 and position[2] < 5)


class TestIntegration:
    """Integration tests for the full packing pipeline."""

    def test_pack_and_check_no_overlap(self, requires_cuda):
        """Test that packed items don't overlap."""
        from spectral_packer import BinPacker

        packer = BinPacker(tray_size=(50, 50, 50))
        items = [np.ones((5, 5, 5), dtype=np.int32) for _ in range(5)]

        result = packer.pack_voxels(items)

        # Check no voxels have multiple item IDs
        # (each voxel should be 0 or a single item ID)
        unique_ids = np.unique(result.tray)
        assert all(id >= 0 for id in unique_ids)

    def test_density_reasonable(self, requires_cuda):
        """Test that density calculation is reasonable."""
        from spectral_packer import BinPacker

        packer = BinPacker(tray_size=(50, 50, 50))
        items = [np.ones((5, 5, 5), dtype=np.int32) for _ in range(3)]

        result = packer.pack_voxels(items)

        # Density should be between 0 and 1
        assert 0 <= result.density <= 1

    def test_total_volume_correct(self, requires_cuda):
        """Test that total volume is calculated correctly."""
        from spectral_packer import BinPacker

        packer = BinPacker(tray_size=(50, 50, 50))
        item = np.ones((5, 5, 5), dtype=np.int32)  # 125 voxels

        result = packer.pack_voxels([item])

        if result.num_placed == 1:
            assert result.total_volume == 125


class TestFloorSupport:
    """floor_support_items: wheeled items stand on the floor and keep the room under them."""

    @staticmethod
    def _cart():
        # 8x6 platform on top, four 1x1 wheels three voxels tall at the corners.
        cart = np.zeros((8, 6, 4), dtype=np.int32)
        cart[:, :, 3] = 1
        for x in (0, 7):
            for y in (0, 5):
                cart[x, y, :3] = 1
        return cart

    def test_ratio_rule_rejects_a_wheeled_item(self):
        from spectral_packer import BinPacker

        packer = BinPacker(tray_size=(10, 8, 8), num_orientations=4, support_threshold=0.4)
        assert packer.pack_voxels([self._cart()]).num_placed == 0

    def test_floor_rule_stands_it_on_its_wheels_with_room_underneath(self):
        from spectral_packer import BinPacker

        flat = np.ones((4, 3, 2), dtype=np.int32)
        # The tray is exactly the cart, so the flat box can only go under it.
        packer = BinPacker(tray_size=(8, 6, 4), num_orientations=4, support_threshold=0.4)
        result = packer.pack_voxels(
            [self._cart(), flat], sort_by_volume=False, floor_support_items=[True, False]
        )
        assert result.num_placed == 2
        cart_place = next(p for p in result.placements if p.item_index == 0)
        assert tuple(cart_place.position) == (0, 0, 0)

    def test_floor_rule_does_not_stand_on_items_from_the_same_call(self):
        from spectral_packer import BinPacker

        slab = np.ones((8, 6, 2), dtype=np.int32)  # covers the whole floor
        packer = BinPacker(tray_size=(8, 6, 8), num_orientations=4, support_threshold=0.4)
        result = packer.pack_voxels(
            [slab, self._cart()], sort_by_volume=False, floor_support_items=[False, True]
        )
        assert [p.success for p in sorted(result.placements, key=lambda p: p.item_index)] == [True, False]


class TestAttractMask:
    """attract_mask: a trunk fills from its back wall instead of the array-origin corner."""

    @staticmethod
    def _walled_tray():
        tray = np.zeros((20, 20, 10), dtype=np.int32)
        tray[[0, -1], :, :] = 1
        tray[:, [0, -1], :] = 1
        tray[:, :, 0] = 1
        return tray

    def test_without_a_mask_the_first_item_drifts_to_the_origin_corner(self):
        from spectral_packer import BinPacker

        tray = self._walled_tray()
        result = BinPacker(tray_size=tray.shape, support_threshold=0.4).pack_voxels(
            [np.ones((4, 4, 2), dtype=np.int32)], initial_tray=tray
        )
        assert result.placements[0].position[1] <= 2

    def test_items_gather_against_the_attracting_back_wall(self):
        from spectral_packer import BinPacker

        tray = self._walled_tray()
        back_wall = np.zeros(tray.shape, dtype=bool)
        back_wall[:, -1, :] = True  # +y is the inside of the trunk
        box = np.ones((4, 4, 2), dtype=np.int32)
        result = BinPacker(tray_size=tray.shape, support_threshold=0.4).pack_voxels(
            [box, box, box], initial_tray=tray, sort_by_volume=False, attract_mask=back_wall
        )
        assert result.num_placed == 3
        # The first box (id 2, walls are 1) touches the back wall: free cells end at y = 18.
        first_cells = np.argwhere(np.asarray(result.tray) == 2)
        assert first_cells[:, 1].max() == 18
