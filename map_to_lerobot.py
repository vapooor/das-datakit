#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os.path as osp
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.video_utils import get_safe_default_codec

from utils.interpolate import get_inter_data
from utils.io import load_json
from utils.mcaploader import McapLoader

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

REF_TOPIC = "/robot0/sensor/camera0/compressed"
RIGHT_CAMERA_TOPIC = "/robot1/sensor/camera0/compressed"

PROCESS_TOPIC = [
    "/robot0/sensor/camera0/compressed",
    "/robot1/sensor/camera0/compressed",
    "/robot0/sensor/imu",
    "/robot1/sensor/imu",
    "/robot0/sensor/magnetic_encoder",
    "/robot1/sensor/magnetic_encoder",
    "/robot0/vio/eef_pose",
    "/robot1/vio/eef_pose",
]



def _get_image_dimensions(mcap_file: str) -> tuple[int, int]:
    bag = McapLoader(mcap_file)
    bag.load_topics([REF_TOPIC, RIGHT_CAMERA_TOPIC], auto_sync=False)

    ref_topic_data = bag.get_topic_data(REF_TOPIC)
    for item in ref_topic_data:
        image = item.get("decode_data")
        if image is not None:
            height, width = image.shape[:2]
            return height, width

    right_topic_data = bag.get_topic_data(RIGHT_CAMERA_TOPIC)
    for item in right_topic_data:
        image = item.get("decode_data")
        if image is not None:
            height, width = image.shape[:2]
            return height, width

    raise ValueError("No valid images found to infer height/width.")


def _ensure_2d(array: np.ndarray) -> np.ndarray:
    if array.ndim == 1:
        return array[:, None]
    return array


def _stack_ee_data(eef_pose_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized conversion:
    eef_pose_data: (N, 7) with [x, y, z, qx, qy, qz, qw]
    returns:
      ee_pos: (N, 3)
      ee_rot: (N, 3) rotvec
    """
    poses = np.asarray(eef_pose_data, dtype=np.float32)
    ee_pos = poses[:, :3]
    quat = poses[:, 3:7]
    ee_rot = R.from_quat(quat).as_rotvec().astype(np.float32)
    return ee_pos, ee_rot


def _match_nearest_indices(src_ts: np.ndarray, tgt_ts: np.ndarray) -> np.ndarray:
    """
    For each timestamp in src_ts, find nearest timestamp index in tgt_ts.
    src_ts: (N,), tgt_ts: (M,) must be sorted ascending (as in bag order).
    Returns indices into tgt_ts: (N,)
    """
    src_ts = np.asarray(src_ts, dtype=np.int64)
    tgt_ts = np.asarray(tgt_ts, dtype=np.int64)

    if len(tgt_ts) == 0:
        raise ValueError("Target timestamps empty; cannot sync topics.")

    if len(tgt_ts) == 1:
        return np.zeros_like(src_ts, dtype=np.int64)

    idx = np.searchsorted(tgt_ts, src_ts, side="left")
    idx = np.clip(idx, 1, len(tgt_ts) - 1)
    prev = idx - 1
    choose_prev = (src_ts - tgt_ts[prev]) <= (tgt_ts[idx] - src_ts)
    return np.where(choose_prev, prev, idx).astype(np.int64)

def iter_topic_times(self, topic: str, which: str = "log_time"):
    """
    which: 'log_time' 或 'publish_time'
    直接从 mcap message 取时间戳，不 ParseFromString、不解码图片。
    """
    self._reset_stream()
    for _, _, message in self._mcap_reader.iter_messages(topics=[topic]):
        yield getattr(message, which)



def convert_single_mcap_to_lerobot(mcap_file: str, dataset: LeRobotDataset, task: str) -> None:
    bag = McapLoader(mcap_file)
    bag.load_topics(PROCESS_TOPIC, auto_sync=False)

    # Load camera topics once
    # ref_topic_data = bag.get_topic_data(REF_TOPIC)
    # right_topic_data = bag.get_topic_data(RIGHT_CAMERA_TOPIC)

    # if len(ref_topic_data) == 0:
    #     raise ValueError(f"No frames in {REF_TOPIC}: {mcap_file}")
    # if len(right_topic_data) == 0:
    #     raise ValueError(f"No frames in {RIGHT_CAMERA_TOPIC}: {mcap_file}")

    # Extract timestamps
    # ref_timestamp = np.asarray([d["data"].header.timestamp for d in ref_topic_data], dtype=np.int64)
    # right_timestamp = np.asarray([d["data"].header.timestamp for d in right_topic_data], dtype=np.int64)
    ref_timestamp = np.fromiter(bag.iter_topic_times(REF_TOPIC, "log_time"), dtype=np.int64)
    right_timestamp = np.fromiter(bag.iter_topic_times(RIGHT_CAMERA_TOPIC, "log_time"), dtype=np.int64)
    
    # Pre-match right frame indices for all left frames (vectorized)
    matched_right_idx = _match_nearest_indices(ref_timestamp, right_timestamp)

    # Interpolate small data once (pose/imu/gripper)
    left_eef_pose = np.asarray(
        get_inter_data(bag, "/robot0/vio/eef_pose", ref_timestamp, inter_type="pose"), np.float32
    )
    right_eef_pose = np.asarray(
        get_inter_data(bag, "/robot1/vio/eef_pose", ref_timestamp, inter_type="pose"), np.float32
    )

    left_gripper = _ensure_2d(
        np.asarray(get_inter_data(bag, "/robot0/sensor/magnetic_encoder", ref_timestamp, inter_type="linear"), np.float32)
    )
    right_gripper = _ensure_2d(
        np.asarray(get_inter_data(bag, "/robot1/sensor/magnetic_encoder", ref_timestamp, inter_type="linear"), np.float32)
    )

    left_imu = np.asarray(get_inter_data(bag, "/robot0/sensor/imu", ref_timestamp, inter_type="linear"), np.float32)
    right_imu = np.asarray(get_inter_data(bag, "/robot1/sensor/imu", ref_timestamp, inter_type="linear"), np.float32)

    # Vectorized pose -> ee_pos/ee_rot
    left_ee_pos, left_ee_rot = _stack_ee_data(left_eef_pose)
    right_ee_pos, right_ee_rot = _stack_ee_data(right_eef_pose)

    ee_pos = np.concatenate([left_ee_pos, right_ee_pos], axis=1)       # (N, 6)
    ee_rot = np.concatenate([left_ee_rot, right_ee_rot], axis=1)       # (N, 6)
    gripper_w = np.concatenate([left_gripper, right_gripper], axis=1)  # (N, 2)
    imu = np.concatenate([left_imu, right_imu], axis=1)                # (N, 12)

    add_frame = dataset.add_frame  # speed: avoid attribute lookup in loop

    # Main loop: only index arrays (no per-frame sync function calls)
    for i, item in enumerate(ref_topic_data):
        left_img = item.get("decode_data")
        # print(left_img.dtype)
        left_img = left_img[..., ::-1].copy()  # BGR to RGB
        if left_img is None:
            raise ValueError(f"Left image None at idx={i}")

        right_img = right_topic_data[matched_right_idx[i]].get("decode_data")
        right_img = right_img[..., ::-1].copy()  # BGR to RGB
        if right_img is None:
            raise ValueError(f"Right image None at idx={i} (matched_right_idx={matched_right_idx[i]})")

        frame = {
            "left_image": left_img,
            "right_image": right_img,
            "state.ee_pos": ee_pos[i],
            "state.ee_rot": ee_rot[i],
            "state.gripper_w": gripper_w[i],
            "state.imu": imu[i],
            "task": task,
        }
        add_frame(frame)

    dataset.save_episode()


def _build_features(height: int, width: int) -> dict:
    return {
        "left_image": {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        },
        "right_image": {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        },
        "state.ee_pos": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["state.ee_pos"],
        },
        "state.ee_rot": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["state.ee_rot"],
        },
        "state.gripper_w": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["state.gripper_w"],
        },
        "state.imu": {
            "dtype": "float32",
            "shape": (12,),
            "names": ["state.imu"],
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MCAP files to LeRobot dataset format.")
    parser.add_argument("--mcap-file", type=str, default="", help="input mcap file path")
    parser.add_argument("--task-dir", type=str, default="", help="task dir containing vio_result.json")
    parser.add_argument("--out-path", type=str, default="", help="output directory for dataset")
    parser.add_argument("--repo-id", type=str, default="mcap_lerobot", help="dataset repo id")
    parser.add_argument("--task", type=str, default="default", help="task name for each episode")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="append episodes to an existing dataset if present",
    )

    # Speed knobs
    parser.add_argument("--image-writer-threads", type=int, default=16, help="threads for image/video writer")
    parser.add_argument("--image-writer-processes", type=int, default=16, help="processes for image/video writer")
    parser.add_argument("--batch-encoding-size", type=int, default=1, help="batch size for video encoding")
    parser.add_argument("--video-backend", type=str, default=None, help="override video backend/codec if needed")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    mcap_file = args.mcap_file
    task_dir = args.task_dir

    process_list: list[str] = []
    if not mcap_file:
        if not task_dir:
            raise ValueError("Either --mcap-file or --task-dir must be provided.")
        if (not osp.exists(task_dir)) or (not osp.isdir(task_dir)):
            raise FileNotFoundError(task_dir)
        mcap_list_file = osp.join(task_dir, "vio_result.json")
        if not osp.exists(mcap_list_file):
            raise FileNotFoundError(mcap_list_file)

        mcap_info = load_json(mcap_list_file)
        mcap_list = mcap_info.get("success_mcap_files", [])
        if len(mcap_list) == 0:
            raise ValueError("No mcap files need to be processed, success_mcap_files is empty.")
        process_list.extend(sorted(set(mcap_list)))
    else:
        if (not osp.exists(mcap_file)) or (not osp.isfile(mcap_file)):
            raise FileNotFoundError(mcap_file)
        process_list.append(mcap_file)

    first_mcap = process_list[0]
    height, width = _get_image_dimensions(first_mcap)
    features = _build_features(height, width)

    if args.out_path:
        base_dir = Path(args.out_path)
    elif task_dir:
        base_dir = Path(task_dir)
    else:
        base_dir = Path(first_mcap).parent or Path.cwd()

    dataset_root = base_dir / args.repo_id
    if dataset_root.exists():
        if not args.resume:
            raise FileExistsError(f"{dataset_root} already exists. Use --resume to append.")
        dataset = LeRobotDataset(repo_id=args.repo_id, root=dataset_root, download_videos=False)
    else:
        # meta = LeRobotDatasetMetadata.create(
        #     repo_id=args.repo_id,
        #     fps=30,
        #     features=features,
        #     robot_type="CobotMagic",
        #     root=dataset_root,
        #     data_files_size_in_mb=50,
        #     video_files_size_in_mb=10,
        # )
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            root=dataset_root,
            fps=30,
            features=features,
            robot_type="CobotMagic",
            use_videos=True,  # ✅ 关键：要编码视频就开这个
            image_writer_processes=args.image_writer_processes,
            image_writer_threads=args.image_writer_threads,
            batch_encoding_size=args.batch_encoding_size,
            video_backend=args.video_backend,  
        )
        dataset.vcodec = "h264"
        dataset.meta.update_chunk_settings(video_files_size_in_mb=10, data_files_size_in_mb=50)

    for idx, f_mcap in enumerate(process_list):
        print(f"######################## Process {idx + 1}/{len(process_list)} ########################")
        print(f"mcap: {f_mcap}")
        convert_single_mcap_to_lerobot(f_mcap, dataset, args.task)

    dataset.finalize()


if __name__ == "__main__":
    main()
