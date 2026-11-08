import numpy as np
import pytest

from fallpre.weekly.scene import (
    analyze_scene,
    binary_iou,
    estimate_floor_height,
    evaluate_semantics,
    run_scene,
)


def analyze(xyz, labels=None, **kwargs):
    labels = np.full(xyz.shape[:2], 19) if labels is None else labels
    return analyze_scene(labels, np.ones(labels.shape), xyz, (0.0, 0.0),
                         {3: "floor", 19: "chair"}, floor_z_m=0.0,
                         min_region_pixels=1, **kwargs)


def test_disk_tests_nearest_surface_not_centroid():
    # One connected chair straddles the radius; its centroid lies outside 2 m.
    xyz = np.array([[[1.9, 0, 0.5], [2.5, 0, 0.5], [3, 0, 0.5]]])
    report, _ = analyze(xyz)
    assert len(report["obstacles"]) == 1
    obstacle = report["obstacles"][0]
    assert obstacle["centroid_xy_m"][0] > 2
    assert obstacle["nearest_visible_surface_xy_distance_m"] == pytest.approx(1.9)
    assert obstacle["visible_pixels_inside_disk"] == 1


def test_metric_radius_boundary_and_height_filter():
    xyz = np.array([[[2.0, 0, 0.5], [2.01, 0, 0.5], [0.2, 0, 2.5], [0.1, 0, 0.02]]])
    labels = np.array([[19, 15, 19, 19]])
    report, _ = analyze(xyz, labels)
    assert len(report["obstacles"]) == 1
    assert report["obstacles"][0]["nearest_visible_surface_xy_distance_m"] == 2


def test_unknown_is_not_filled_and_obstacle_takes_precedence():
    xyz = np.array([[[0.01, 0.01, 0.0], [0.01, 0.01, 0.5], [np.nan, 0, 0.5]]])
    report, grid = analyze(xyz, np.array([[3, 19, 19]]))
    coverage = report["coverage"]
    assert coverage["unknown_fraction"] > 0.99
    assert coverage["visible_floor_fraction"] == 0
    assert np.sum(grid == 2) == 1
    assert sum(coverage[k] for k in ["unknown_fraction", "visible_floor_fraction",
                                   "visible_elevated_surface_fraction"]) == pytest.approx(1)


def test_floor_fit_rejects_elevated_misclassified_pixels():
    xyz = np.zeros((20, 20, 3))
    xyz[:4, :, 2] = 1.0
    height, support = estimate_floor_height(xyz, np.full((20, 20), 3), np.ones((20, 20)))
    assert height == 0
    assert support["inlier_pixels"] == 320


def test_floor_fit_requires_model_evidence():
    with pytest.raises(ValueError, match="floor support"):
        estimate_floor_height(np.zeros((20, 20, 3)), np.full((20, 20), 19), np.ones((20, 20)))


def test_semantic_evaluation_ignores_unlabeled_gt_and_groups_chair_variants():
    predicted = np.array([[3, 3, 3, 30]])
    truth = np.array([[2, -1, 1, 5]])
    evaluation = evaluate_semantics(predicted, truth)
    assert evaluation["matched_class_pixel_iou"]["floor"]["iou"] == 0.5
    assert evaluation["matched_class_pixel_iou"]["chair"]["iou"] == 1
    assert evaluation["ground_truth_used_for_inference"] is False
    assert binary_iou(np.zeros((1, 1), bool), np.zeros((1, 1), bool)) is None


def test_missing_metric_center_and_nonpositive_radius_are_rejected():
    xyz = np.zeros((1, 1, 3))
    with pytest.raises(ValueError, match="radius"):
        analyze(xyz, radius_m=0)
    with pytest.raises(ValueError, match="finite metric"):
        analyze_scene(np.array([[19]]), np.ones((1, 1)), xyz,
                      (np.nan, 0), {19: "chair"}, floor_z_m=0)


def test_horizontal_distance_is_independent_of_world_origin_and_height():
    xyz = np.array([[[101.2, -48.4, 8.4]]])
    report, _ = analyze_scene(np.array([[19]]), np.ones((1, 1)), xyz,
                              (100, -50), {19: "chair"}, floor_z_m=8,
                              min_region_pixels=1)
    assert report["obstacles"][0]["nearest_visible_surface_xy_distance_m"] == pytest.approx(2)


def test_focus_person_mask_does_not_become_its_own_obstacle():
    xyz = np.array([[[0, 0, 1], [1, 0, 0.5]]], dtype=float)
    labels = np.array([[12, 19]])
    report, grid = analyze_scene(labels, np.ones((1, 2)), xyz, (0, 0), {12: "person", 19: "chair"},
                                 floor_z_m=0, min_region_pixels=1,
                                 subject_mask=np.array([[True, False]]))
    assert len(report["obstacles"]) == 1
    assert np.sum(grid == 2) == 1


def test_disk_smaller_than_grid_resolution_is_reported():
    with pytest.raises(ValueError, match="grid resolution"):
        analyze(np.zeros((1, 1, 3)), radius_m=0.001)


def test_existing_result_directory_is_preserved(tmp_path):
    marker = tmp_path / "scene-result.json"
    marker.write_text("existing result")
    with pytest.raises(FileExistsError, match="Output already exists"):
        run_scene(tmp_path)
    assert marker.read_text() == "existing result"
