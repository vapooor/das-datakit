import argparse
import cv2
from utils.mcaploader import McapLoader
import pdb

PROCESS_TOPIC = [
    # mid fisheye camera
    "/robot0/sensor/camera0/compressed",
    # left stereo camera, sync with timestamp
    "/robot0/sensor/camera1/compressed",
    # right stereo camera, sync with timestamp
    "/robot0/sensor/camera2/compressed",
    # imu, sync with interpolation
    "/robot0/sensor/imu", 
    # tactile sensor, sync with interpolation
    "/robot0/sensor/tactile_left",
    "/robot0/sensor/tactile_right",
    # gripper width, sync with interpolation
    "/robot0/sensor/magnetic_encoder",
    # gripper pose, sync with interpolation
    "/robot0/vio/eef_pose",
]

def parse_mcap(mcap_file):
    bag = McapLoader(mcap_file)
    # step 1
    # parse all data we need
    bag.load_topics(PROCESS_TOPIC, auto_sync=False)
    print(bag)

    # decode images
    camera0_img_data = bag.get_topic_data("/robot0/sensor/camera0/compressed")
    for d in camera0_img_data:
        single_frame_img = dict(
            data=d["decode_data"], # [h, w, c], bgr
            timestamp=d["data"].header.timestamp
        )
        # save image
        # cv2.imwrite("debug.jpg", single_frame_img["data"])
    
    # decode vio pose
    vio_pose_data = bag.get_topic_data("/robot0/sensor/imu")
    for d in vio_pose_data:
        single_frame_pose = dict(
            data=d["decode_data"], # [Pos_X, Pos_Y, Pos_Z, Q_X, Q_Y, Q_Z, Q_W], detailed information can be found in README.md
            timestamp=d["data"].header.timestamp
        )

    # get aligned data by timestamp
    bag.register_sync_relation_with_time("/robot0/sensor/camera0/compressed", "/robot0/sensor/imu")
    camera0_idx = bag.get_topic_seq_num("/robot0/sensor/camera0/compressed") # anchor topic
    for seq_idx in camera0_idx:
        data = bag.get_topic_data_by_seq_num("/robot0/sensor/camera0/compressed", seq_idx, sync_topics=["/robot0/sensor/imu"])
        if data is None:
            continue
        frame_img = dict(
            data=data["/robot0/sensor/camera0/compressed"]["decode_data"], # [h, w, c], bgr
            timestamp=data["/robot0/sensor/camera0/compressed"]["data"].header.timestamp
        )
        frame_pose = dict(
            data=data["/robot0/sensor/imu"]["decode_data"], # [Pos_X, Pos_Y, Pos_Z, Q_X, Q_Y, Q_Z, Q_W], detailed information can be found in README.md
            timestamp=data["/robot0/sensor/imu"]["data"].header.timestamp
        )
        print("frame_img timestamp: ", frame_img["timestamp"], "frame_pose timestamp: ", frame_pose["timestamp"])

  
def parse_args():
    parser = argparse.ArgumentParser(description="Convert DAS mcap to h5 file")
    parser.add_argument("mcap_file", type=str, default="", help="input mcap file path")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    mcap_file = args.mcap_file
    parse_mcap(mcap_file)
