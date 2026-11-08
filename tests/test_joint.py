import numpy as np
import pytest

from fallpre.weekly.joint import (
    TargetTracker,
    associate_rgb_depth,
    depth_to_xyz,
    fit_floor,
    person_root,
    read_timestamps,
    run_joint,
    to_ground,
)


def test_timestamp_association_is_bounded_one_to_one_and_keeps_source_index():
    rgb = [(0., "rgb/0.png"), (.033, "rgb/1.png"), (.066, "rgb/2.png")]
    depth = [(.005, "depth/0.png"), (.05, "depth/1.png")]
    pairs = associate_rgb_depth(rgb, depth)
    assert [p["source_frame_id"] for p in pairs] == [0, 2]
    assert len({p["depth_path"] for p in pairs}) == 2
    assert pairs[-1]["sync_error_s"] == pytest.approx(.016)
    assert associate_rgb_depth([(1., "r")], [(2., "d")]) == []
    assert len(associate_rgb_depth([(0., "r")], [(.02, "d")])) == 1


def test_timestamp_parser_rejects_nonmonotonic_and_path_escape(tmp_path):
    p = tmp_path / "rgb.txt"
    p.write_text("# header\n2 rgb/2.png\n1 rgb/1.png\n")
    with pytest.raises(ValueError, match="increasing"):
        read_timestamps(p)
    p.write_text("1 ../escape.png\n")
    with pytest.raises(ValueError, match="unsafe"):
        read_timestamps(p)


def test_tum_depth_scale_and_missing_depth_and_pixel_calibration():
    k = {"fx": 2., "fy": 4., "cx": 1., "cy": 0., "width": 3, "height": 2,
         "depth_units_per_m": 5000.}
    depth = np.array([[5000, 10000, 0], [10000, 10000, 10000]], dtype=np.uint16)
    xyz = depth_to_xyz(depth, k)
    assert xyz[0, 0] == pytest.approx([-.5, 0, 1])
    assert xyz[1, 2] == pytest.approx([1, .5, 2])
    assert np.isnan(xyz[0, 2]).all()
    with pytest.raises(ValueError, match="uint16"):
        depth_to_xyz(depth.astype(np.uint8), k)
    with pytest.raises(ValueError, match="uint16"):
        depth_to_xyz(None)
    with pytest.raises(ValueError, match="dimensions"):
        depth_to_xyz(depth)


def floor_data():
    y, x = np.indices((40, 40))
    xyz = np.stack([(x-20)/10, np.full_like(x, 1.3, dtype=float), 1+y/10], axis=-1)
    labels = np.full((40, 40), 3)
    labels[:12] = 15  # dominant tabletop pixels are never candidate floor.
    return xyz, labels, np.ones(labels.shape)


def test_semantic_ransac_rejects_tabletop_and_transforms_height():
    xyz, labels, confidence = floor_data()
    xyz[:12, :, 1] = .5
    xyz[12:16, :, 1] = .5  # minority table pixels misclassified as floor
    plane = fit_floor(xyz, labels, confidence)
    assert plane["camera_height_m"] == pytest.approx(1.3)
    assert plane["normal_camera"] == pytest.approx([0, -1, 0], abs=1e-6)
    ground = to_ground(np.array([[0, 1.3, 2], [0, .3, 2]]), plane)
    assert ground[:, 2] == pytest.approx([0, 1.])
    assert np.linalg.norm(ground[0, :2]-ground[1, :2]) == pytest.approx(0)


def test_floor_fit_fails_without_semantic_or_mounting_support():
    xyz, labels, confidence = floor_data()
    with pytest.raises(ValueError, match="Insufficient"):
        fit_floor(xyz, np.full_like(labels, 15), confidence)
    xyz[..., 1], xyz[..., 2] = xyz[..., 2].copy(), xyz[..., 1].copy()
    with pytest.raises(ValueError, match="mounting"):
        fit_floor(xyz, labels, confidence)


def test_ceiling_camera_prior_accepts_top_image_floor_and_metric_horizontal_disk():
    y, x = np.indices((40, 40))
    xyz = np.stack([(x-20)/10, (y-20)/10, np.full_like(x, 3., dtype=float)], axis=-1)
    labels = np.full((40, 40), 3)
    labels[28:] = 15  # upper-image floor is valid for a downward camera
    confidence = np.ones(labels.shape)
    plane = fit_floor(xyz, labels, confidence, mounting="ceiling")
    assert plane["camera_height_m"] == pytest.approx(3.)
    assert plane["normal_camera"] == pytest.approx([0, 0, -1], abs=1e-6)
    assert plane["downward_pitch_degrees"] == pytest.approx(90.)
    ground = to_ground(np.array([[0.,0.,2.], [1.2,1.6,3.]]),plane)
    assert ground[:,2] == pytest.approx([1.,0.])
    assert np.linalg.norm(ground[1,:2]-ground[0,:2]) == pytest.approx(2.)
    with pytest.raises(ValueError,match="lower image"):
        fit_floor(xyz,labels,confidence,mounting="upright")


def test_metric_root_needs_detected_hips_and_person_depth_not_occluding_table():
    xyz, labels, confidence = floor_data()
    plane = fit_floor(xyz, labels, confidence)
    person = {"kpts": [[0., 0., 0.] for _ in range(17)]}
    person["kpts"][11], person["kpts"][12] = [12., 20., .9], [27., 20., .9]
    xyz[15:26, 7:18] = [-.15, .3, 2.]
    xyz[15:26, 22:33] = [.15, .3, 2.]
    with pytest.raises(ValueError, match="person-labeled"):
        person_root(person, xyz, labels, plane)
    labels[15:26, 7:18] = labels[15:26, 22:33] = 12
    root = person_root(person, xyz, labels, plane)
    assert root["ground_xyz_m"][2] == pytest.approx(1)
    assert root["body_center_ground_truth"] is False
    xyz[15:26, 22:33, 2] = 4.
    with pytest.raises(ValueError, match="inconsistent"):
        person_root(person, xyz, labels, plane)


def test_target_does_not_change_when_other_detection_becomes_more_confident():
    tracker = TargetTracker()
    a = {"box": [0, 0, 100, 200], "det": .9}
    b = {"box": [200, 0, 300, 200], "det": .8}
    assert tracker.update([a, b], 0)[0] == 0
    a["det"], b["det"] = .5, .99
    assert tracker.update([b, a], .033)[0] == 1
    assert tracker.epoch == 1
    assert tracker.update([], .066) == (None, "missing_person")
    assert tracker.update([b], .1)[0] == 0
    assert tracker.epoch == 2


def test_target_ambiguity_resets_and_existing_output_is_preserved(tmp_path):
    tracker = TargetTracker()
    a = {"box": [0, 0, 100, 200], "det": .9}
    tracker.update([a], 0)
    assert tracker.update([a, dict(a)], .033) == (None, "ambiguous_identity_reset")
    with pytest.raises(FileExistsError):
        run_joint(tmp_path)


@pytest.mark.parametrize("arguments", [
    {"start_s": -1}, {"start_s": float("nan")}, {"duration_s": float("inf")},
    {"radius_m": float("nan")}, {"scene_stride": 1.5}, {"scene_stride": 0},
])
def test_invalid_run_parameters_fail_before_loading_models(tmp_path, arguments):
    with pytest.raises(ValueError, match="finite"):
        run_joint(tmp_path / "unused", **arguments)
