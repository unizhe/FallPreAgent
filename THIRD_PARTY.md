# Open-source provenance

| Component | Pinned source / role | License |
|---|---|---|
| Ultralytics | `third_party/ultralytics`, tag `v8.4.63`, commit `c9271a4967322f8cab976448f049400dc617e598`; editable runtime, YOLO26/YOLO11 Pose/Detect and integrated ByteTrack | AGPL-3.0; upstream also offers enterprise licensing |
| Original ByteTrack | `third_party/ByteTrack`, commit `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99`; source reference, not installed as a second tracker | MIT |
| YOLO26n Pose / YOLO11n Pose / YOLO11n Detect weights | Official `ultralytics/assets` v8.4.0 release; files and SHA-256 in `models.lock.json` | Follow upstream Ultralytics model licensing |
| MMAction2 / ST-GCN++ source | `third_party/algorithms/mmaction2`, commit `a5a167dff2399e2d182a60332325f9c0d4663517`; source, model configuration and NTU60 label map | Apache-2.0 for repository code; optional dependencies have their own terms |
| ST-GCN++ NTU60 2D joint checkpoint | Official OpenMMLab checkpoint ending in `20221228-86e1e77a.pth`; URL, size and SHA-256 in `configs/runtime-resources.json` | Checkpoint and NTU60 dataset terms have not been separately audited; the source-code license does not establish permission to redistribute them |
| OpenCV, NumPy, SciPy, Shapely, Pydantic, PyTorch, LAP | Installed wheels; exact resolved versions/hashes in `uv.lock` | See the respective installed distribution metadata and upstream licenses |

The core source pins are in `sources.lock.json`; MMAction2 and runtime resource pins are in `configs/runtime-resources.json`. The earlier source-only audit is preserved in `algorithm-sources.2026-10-14.v1.json`. These records describe the local evidence collected for this project, not a fresh review of all upstream terms. Nested clones and weights are excluded from this parent repository's Git staging, but remain available locally. Preserve upstream LICENSE files and notices when redistributing. Integration with an AGPL component requires choosing a compatible distribution approach; this document does not select a license for the project's own code or relicense upstream code.

The current camera entry point defaults to **YOLO26n-Pose**. YOLO11n-Pose remains a historical comparison option, and YOLO11n Detect supplies person detections for the optional RGB branch. Historical YOLO11 experiment records do not describe the current default.

- [Ultralytics repository](https://github.com/ultralytics/ultralytics)
- [ByteTrack repository](https://github.com/FoundationVision/ByteTrack)
- [Official model assets](https://github.com/ultralytics/assets/releases/tag/v8.4.0)
- [MMAction2 at the recorded commit](https://github.com/open-mmlab/mmaction2/tree/a5a167dff2399e2d182a60332325f9c0d4663517)

The Kalman, temporal state machines, baseline selection, geometric composition, policy and tool adapters in `src/fallpre/` are project integration code, not copied fall-detection pretrained models. No claim of a clinically validated fall classifier is made.

## ST-GCN++ runtime and project-trained artifacts

The pretrained backbone is the official `stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d` model. Its recorded checkpoint SHA-256 is `86e1e77a7b27dd53c6ccc1600a4cc3c321ec8e04d17f25a41fbbb7896026945a`. The source-only audit dated 2026-10-14 recorded an undownloaded checkpoint; the current runtime inventory records the subsequently acquired artifact. The checkpoint's 60-class output is an NTU action score, not a calibrated fall probability. The project's causal history adapter, quality gates, fusion and event management are separate integration code.

The current inventory distinguishes public upstream downloads from project-trained `team-artifact` resources. Listing a download URL or an artifact hash is provenance, not a grant of redistribution rights.

| Project artifact | Training origin and role | Distribution boundary |
|---|---|---|
| Frozen V7 completion classifier (`action-v7`) | Project V3–V7 nested classifiers; source manifests are `configs/action-v3-dataset.json` through `configs/action-v7-dataset.json`. These include URFD, whose recorded dataset terms are CC BY-NC-SA 4.0, and other sources with separately recorded terms. | Exact artifact hash and local/legacy locations are in `configs/runtime-resources.json`; no public download or independent derived-weight license verification is recorded. |
| C process head (`fall-head-c`, `head-c-st`) | Project-trained binary linear head on 256 frozen ST-GCN++ features, using URFD development sequences and weakly aligned official state labels. | Model parameters and normalization statistics are separate local artifacts. URFD data terms and pretrained-backbone provenance must remain attached to any proposed distribution; derived-weight redistribution terms have not been independently verified. |
| D process head (`fall-head-d`, `head-d-geometry`) | Project-trained binary linear head on the same frozen features plus 21 motion/pose geometry features, using the geometry-available URFD development subset. | Same boundary as C; the runtime inventory pins its separate SHA-256. It is not interchangeable with the C input representation. |

The C/D training records are retained locally under `runs/stgcn-fall-20261107/official-dev/` (`provenance.json`, per-head `report.json` and `README.md`). A `head-c-common` control uses the same samples as D and is not a registered runtime resource. The experiment uses historical YOLO11n skeleton caches from low-resolution RGB previews; official depth-frame labels were transferred by frame index without independently verifying physical RGB/depth synchronization. It does not train on the user's reviewed personal videos, establish independent-person validation, or prove an improvement in deployed alarms. The backbone and YOLO weights were frozen during this experiment. C/D outputs remain experimental diagnostics, not approved alarm policies.

The URFD source/license record is `configs/weekly-data.lock.json`; detailed training manifests retain source attribution and checksums. Recording a dataset license does not establish that all derived weights automatically have the same license. Resolve the applicable data, checkpoint and distribution terms before publishing trained artifacts. The source repository therefore carries manifests and integration/training code, while original data, personal recordings, review results and model artifacts remain outside its publication set.

## V6 slow-motion source screening (2026-10-19)

[Assistive-Robotics-Lab STS_Dataset](https://github.com/Assistive-Robotics-Lab/STS_Dataset), Tom Bennett and Virginia Ruiz Garate, 2022, DOI 10.5281/zenodo.6400938. License: **CC BY-NC 4.0** (local copy `runs/action-v6/inputs/xsens-LICENSE.txt`). b01–b06 archives were read solely to screen native sit-to-stand kinematics; no samples admitted to V6 camera-model training or reported RGB test. Source checksums and member-level audit: `runs/action-v6/inputs/xsens-slow-screen.json`.


## V7 新增数据核查（2026-10-19）

UL-RED，Nikhil Reji、Kristiaan D’Août、Sebastiano Fichera、Paolo Paoletti，University of Liverpool；[官方数据](https://datacat.liverpool.ac.uk/2729/)，[论文](https://doi.org/10.1038/s41597-025-05099-1)，数据许可 CC BY 4.0。本地仅下载 S01/S02/S03 的 SitToStandR3 marker clean、marker-less raw/clean 共 9 个 AMC 成员，CRC/SHA256 在 `runs/action-v7/inputs/ulred/acquisition.json`。转换为坐标曲线和第三次上升时长分析；没有重定时，未输入模型或参与训练。作者 viewer/notebooks 仅阅读，未执行。

REMAP Open，Craddock 等，University of Bristol，[DOI 10.5523/bris.21h9f9e30v9cl2fapjggz4q1x7](https://data.bris.ac.uk/data/dataset/21h9f9e30v9cl2fapjggz4q1x7)，CC BY 4.0。本轮只成功核查在线元数据与 README；下载超时/空响应，未取得样本，未纳入训练测试。

Telkom-STS 本轮新增人物 7/8/9，原始来源及 MD5 位于 `runs/action-v7/inputs/acquisition.json`；固定前 20 秒派生片段及助手暂定阶段标注，未改时间轴。原始许可和归属沿用此前 Telkom 记录。

## Independent live fallen detector experiment (2026-10-23)

The live RGB branch uses [kamalchibrani/yolov8_fall_detection_25](https://huggingface.co/kamalchibrani/yolov8_fall_detection_25), revision `607a55e0f2a3c056cf1afb82a9a6f8e65a97bc0b`, SHA256 `4a7eba2e982955d7c5f2ea65968f23332c123275116e75e2bac915f40ba749e0`. Its author card declares OpenRAIL; the underlying Ultralytics model/runtime licensing also applies. The author does not disclose usable training-set membership or splits: overlap with URFD cannot be ruled out. Model output is only current-image posture evidence; project code performs identity, descent, co-motion and duration checks. The official YOLO11n person model supplies initial body boxes; both weight hashes and sources are pinned in `configs/rgb-fall-model.json`. Original YOLO11/26 pose and frozen V7 remain separate.

Exploratory alternatives were not selected: official YOLO11s/YOLO26s Pose, [melihuzunoglu/human-fall-detection](https://huggingface.co/melihuzunoglu/human-fall-detection) (author declares AGPL-3.0), and [Manako fallen/crouching model](https://huggingface.co/manako/Detect-non-violent-fallen-and-crouching-person-detection) (ONNX metadata declares AGPL-3.0; card gives no license). Fixed revisions, SHA256, preprocessing, author metadata and observed failures are retained in `runs/live-fall-branch-20261022/model-probes/` and `rgb-probes/`. No author repository code was executed and no dependency upgrade was made for these experiments. These inspected clips are development data, not a blind independent validation set.
