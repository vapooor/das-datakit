import numpy as np
import os.path as osp
import json
from mcap.reader import make_reader, McapReader, NonSeekingReader
from mcap.writer import Writer
from mcap_protobuf.decoder import DecoderFactory
from mcap_protobuf.writer import Writer
from .topic_parser import (
    img_parser,
    imu_parser,
    tactile_parser,
    pose_parser,
    magnetic_encoder_parser
)
from collections import defaultdict
from .sync_graph import RelationGraph
import io
import sys
sys.path.append(osp.join(osp.dirname(__file__), "../pb2"))
from pb2.IMUMeasurement_pb2 import IMUMeasurement
from pb2.CameraCalibration_pb2 import CameraCalibration
from pb2.SystemInfo_pb2 import SystemInfo
from pb2.TactileMeasurement_pb2 import TactileMeasurement
from pb2.MagneticEncoder_pb2 import MagneticEncoderMeasurement
from pb2.CompressedImage_pb2 import CompressedImage
from pb2.PoseInFrame_pb2 import PoseInFrame
from pb2.RobotInfo_pb2 import RobotInfo
import pdb

PROTO_MAPPING = {
    "/robot0/sensor/camera0/compressed": CompressedImage,
    "/robot0/sensor/camera1/compressed": CompressedImage,
    "/robot0/sensor/camera2/compressed": CompressedImage,
    "/robot0/sensor/imu": IMUMeasurement,
    "/robot0/sensor/camera0/camera_info": CameraCalibration,
    "/robot0/sensor/camera1/camera_info": CameraCalibration,
    "/robot0/sensor/camera2/camera_info": CameraCalibration,
    "/robot0/system_info": SystemInfo,
    "/robot0/sensor/tactile_left": TactileMeasurement,
    "/robot0/sensor/tactile_right": TactileMeasurement,
    "/robot0/sensor/magnetic_encoder": MagneticEncoderMeasurement,
    "/robot0/vio/eef_pose": PoseInFrame,
    "/robot0/sim/robot_info": RobotInfo,

    "/robot1/sensor/camera0/compressed": CompressedImage,
    "/robot1/sensor/camera1/compressed": CompressedImage,
    "/robot1/sensor/camera2/compressed": CompressedImage,
    "/robot1/sensor/imu": IMUMeasurement,
    "/robot1/sensor/camera0/camera_info": CameraCalibration,
    "/robot1/sensor/camera1/camera_info": CameraCalibration,
    "/robot1/sensor/camera2/camera_info": CameraCalibration,
    "/robot1/system_info": SystemInfo,
    "/robot1/sensor/tactile_left": TactileMeasurement,
    "/robot1/sensor/tactile_right": TactileMeasurement,
    "/robot1/sensor/magnetic_encoder": MagneticEncoderMeasurement,
    "/robot1/vio/eef_pose": PoseInFrame,
    "/robot1/sim/robot_info": RobotInfo,

    "/robot0/sensor/camera3/compressed": CompressedImage,
    "/robot0/sensor/camera4/compressed": CompressedImage,
    "/robot0/sensor/camera5/compressed": CompressedImage,
    "/robot0/sensor/camera3/camera_info": CameraCalibration,
    "/robot0/sensor/camera4/camera_info": CameraCalibration,
    "/robot0/sensor/camera5/camera_info": CameraCalibration,
}

def ns_to_s(ns):
    return float(ns) / 1e9

def parse_topic_data(reader: McapReader, topic: str):
    if topic not in PROTO_MAPPING:
        print(f"topic {topic} is not in PROTO_MAPPING.")
        return []
    proto_msg_class = PROTO_MAPPING[topic]
    topic_msgs = []
    for schema, channel, message in reader.iter_messages(topics=[topic]):
        proto_msg = proto_msg_class()
        proto_msg.ParseFromString(message.data)
        topic_msgs.append({
            "data": proto_msg,
            "log_time": message.log_time,
            "publish_time": message.publish_time,
        })
    return topic_msgs

class McapLoader:
    AUTO_DECOMPRESS_MAP = {
        "CompressedImage": img_parser,
        "CompressedVideo": img_parser,
        "IMUMeasurement": imu_parser,
        "MagneticEncoderMeasurement": magnetic_encoder_parser,
        "TactileMeasurement": tactile_parser,
        "PoseInFrame": pose_parser
    }
    
    def __init__(
        self,
        bag_path:str,
    ):
        self._bag_path = bag_path
        self.init_reader()
        self._bag_data = {}
        self.get_statistic_info()

        self.topic_sync_info = defaultdict(dict)
        self.seq2idx = defaultdict(dict)
        self.sync_graph = RelationGraph()

    def init_reader(self):
        self._stream = io.BytesIO(open(self._bag_path, "rb").read())
        self._mcap_reader = NonSeekingReader(self._stream, decoder_factories=[DecoderFactory()])
    
    def _reset_stream(self):
        self._stream.seek(0)
        self._mcap_reader = NonSeekingReader(self._stream, decoder_factories=[DecoderFactory()])
    
    def get_statistic_info(self):
        self._reset_stream()
        topic_summary = self._mcap_reader.get_summary()
        self._reset_stream()
        header = self._mcap_reader.get_header()
        self._topic_header = header
        # parse iter info
        self._reset_stream()
        meta_data = self._mcap_reader.iter_metadata()
        self._topic_meta = [meta for meta in meta_data]
        self._reset_stream()
        attachments = self._mcap_reader.iter_attachments()
        self._topic_attachments = [att for att in attachments]

        # read topic->id mapping
        topic_channels = topic_summary.channels
        topic2id = {}
        topic2schema_id = {}
        for v in topic_channels.values():
            if v.message_encoding != "protobuf":
                print(f"Unsupported message encoding: {v.message_encoding}")
            topic2id[v.topic] = v.id
            topic2schema_id[v.topic] = v.schema_id

        all_topic_names = list(topic2id.keys())

        topic_statistics = topic_summary.statistics
        topic_msg_count = topic_statistics.channel_message_counts
        msg_start_time = topic_statistics.message_start_time
        msg_end_time = topic_statistics.message_end_time

        # frequency
        bag_time_length = (msg_end_time - msg_start_time) / 1e9  # seconds
        topic_frequency_info = {}
        for topic_name in all_topic_names:
            msg_count = topic_msg_count.get(topic2id[topic_name], 0)
            if msg_count == 0:
                topic_frequency_info[topic_name] = 0
            else:
                topic_frequency_info[topic_name] = round(topic_msg_count[topic2id[topic_name]] / bag_time_length, 1)

        self.topic_schemas = {
            tn: topic_summary.schemas[topic2schema_id[tn]].name
            for tn in all_topic_names
        }
        self.all_topic_names = all_topic_names  
        self.topic_statistics = topic_statistics
        self.msg_start_time = msg_start_time
        self.msg_end_time = msg_end_time
        self.topic_frequency_info = topic_frequency_info

    def _update_seq2idx(self, topic_name: str, topic_data: list):
        for idx, sdata in enumerate(topic_data):
            if not hasattr(sdata["data"], "header"):
                continue
            header = sdata["data"].header
            sequence_num = header.sequence_num
            self.seq2idx[topic_name][sequence_num] = idx

    def _update_sync_info(self, topic_name: str, topic_data: list):
        for idx, sdata in enumerate(topic_data):
            if not hasattr(sdata["data"], "header"):
                continue
            header = sdata["data"].header
            sequence_num = header.sequence_num
            for input in header.inputs:
                self.sync_graph.add_relation(topic_name, sequence_num, {input.topic_name: input.sequence_num})

    def register_sync_relation_with_time(self, topic_name_1: str, topic_name_2: str, overwrite: bool = False) -> bool:
        if topic_name_1 == topic_name_2:
            print(f"two topics are same.")
            return True
        self.load_topics([topic_name_1, topic_name_2])
        
        topic_data_1 = self.get_topic_data(topic_name_1)
        if (topic_data_1 is None) or len(topic_data_1) == 0:
            print(f"topic {topic_name_1} is not in bag.")
            return False
        topic_data_2 = self.get_topic_data(topic_name_2)
        if (topic_data_2 is None) or len(topic_data_2) == 0:
            print(f"topic {topic_name_2} is not in bag.")
            return False
        
        seq_for_topic2 = []
        ts_for_topic2 = []
        for idx, sdata in enumerate(topic_data_2):
            # 有些topic没有header
            if not hasattr(sdata["data"], "header"):
                continue
            header = sdata["data"].header
            seq_for_topic2.append(header.sequence_num)
            ts_for_topic2.append(header.timestamp)
        
        if not ts_for_topic2:
            print(f"No valid timestamps found for {topic_name_2}")
            return False
        
        ts_for_topic2 = np.array(ts_for_topic2, dtype=np.int64)
        
        # start register sync relation
        valid_register_count = 0
        for idx, sdata in enumerate(topic_data_1):
            # 有些topic没有header
            if not hasattr(sdata["data"], "header"):
                continue
            header = sdata["data"].header
            sequence_num = header.sequence_num
            ts = header.timestamp

            # 找到对应的topic2的序号
            diff = np.abs(ts - ts_for_topic2)
            min_diff_idx = np.argmin(diff)

            # 注册同步关系
            self.sync_graph.add_relation(topic_name_1, sequence_num, {topic_name_2: seq_for_topic2[min_diff_idx]}, overwrite=overwrite)
            valid_register_count += 1
        
        self.sync_graph.deduce_relations()
        
        if valid_register_count == 0:
            print(f"no valid register relation between {topic_name_1} and {topic_name_2}")
        else:
            print(f"register [{valid_register_count}] sync relation between {topic_name_1}({len(topic_data_1)}), {topic_name_2}")
        return True
    
    def load_topics(self, topics: list = [], auto_decompress: bool = True, auto_sync: bool = False):
        """
        Load specified topics from a bag file.

        This method loads the specified topics from the bag file located at
        self._bag_path. If a topic is already loaded, it will not be loaded again.
        The method ensures that there are no duplicate topics being loaded.

        Args:
            topics (list): A list of topic names to be loaded. If a single topic
                   name is provided as a string, it will be converted to
                   a list containing that single topic.

        Returns:
            None
        """
        if not isinstance(topics, list):
            topics = [topics]
        # 防止重复解析
        not_loaded_topics = []
        for topic in topics:
            if topic not in self.all_topic_names:
                print(f" >> {topic} << is not protobug topic or not in bag, so it will not be loaded")
                continue
            if topic not in self._bag_data:
                not_loaded_topics.append(topic)
        # 去重
        not_loaded_topics = list(set(not_loaded_topics))
        if len(not_loaded_topics) == 0:
            return
        bag_data = {}
        for topic_name in not_loaded_topics:
            self._reset_stream()
            topic_msgs = parse_topic_data(self._mcap_reader, topic_name)
            # auto decompress
            if auto_decompress:
                try:
                    topic_msgs = self._auto_decompress(topic_msgs)
                except Exception as e:
                    print(f"auto decompress failed for topic {topic_name}: {e}")

            self._update_seq2idx(topic_name, topic_msgs)
            # update sync info
            if auto_sync:
                self._update_sync_info(topic_name, topic_msgs)
                self.sync_graph.deduce_relations()

            bag_data[topic_name] = topic_msgs

        if auto_sync:
            self.sync_graph.deduce_relations()
        self._bag_data.update(bag_data)
        
    def _auto_decompress(self, topic_data: list):
        def match_func(proto_desc):
            for k, v in self.AUTO_DECOMPRESS_MAP.items():
                if k in proto_desc:
                    return v
            return None
        if len(topic_data) > 0:
            proto_data = [d["data"] for d in topic_data]
            proto_desc = proto_data[0].DESCRIPTOR.name
            func = match_func(proto_desc)
            if func is not None:
                decompressed_data = func(proto_data)
                for idx, d in enumerate(decompressed_data):
                    topic_data[idx]["decode_data"] = d
                topic_data = [d for d in topic_data if d["decode_data"] is not None]
        return topic_data
    
    def get_bag_data(self):
        return self._bag_data

    def get_topic_schema(self, topic_name) -> str:
        if topic_name not in self.topic_schemas:
            print(f">> {topic_name} << is not in bag.")
            return ""
        return self.topic_schemas[topic_name]

    def get_topic_data(self, topic_name):
        self.load_topics(topic_name)
        if topic_name not in self._bag_data:
            return None
        return self._bag_data[topic_name]

    # Get a frame of data for a topic based on seq num
    def get_topic_data_by_seq_num(self, topic_name, seq_id, sync_topics=[]):
        """
        Retrieve data for a specific topic and sequence number, along with synchronized data for other topics.

        Args:
            topic_name (str): The name of the primary topic to retrieve data for.
            seq_id (int): The sequence number of the data to retrieve.
            sync_topics (list, optional): A list of topic names to retrieve synchronized data for. Defaults to an empty list.

        Returns:
            dict: A dictionary containing the data for the primary topic and any synchronized topics. 
              If the primary topic or any synchronized topic data is not found, the corresponding value will be None.
        """
        # load topic data
        self.load_topics(topic_name)
        self.load_topics(sync_topics)
        if topic_name not in self._bag_data:
            print(f"{topic_name} is noe in bag_data.")
            return None
        hit_data = {}
        list_idx = self.seq2idx[topic_name].get(seq_id, None)
        if list_idx is None:
            return None
        hit_data[topic_name] = self._bag_data[topic_name][list_idx]
        # get sync data
        if isinstance(sync_topics, str):
            sync_topics = [sync_topics]
        # deduplication
        sync_topics = [i for i in sync_topics if i!=topic_name]
        if len(sync_topics) > 0:
            cur_sync_info = self.sync_graph.get_relations(topic_name, seq_id)
            for sname in sync_topics:
                sync_topic_seq = cur_sync_info.get(sname, None)
                if sync_topic_seq is None:
                    hit_data[sname] = None
                    continue
                data_idx = self.seq2idx[sname].get(sync_topic_seq, None)
                if data_idx is None:
                    hit_data[sname] = None
                    continue
                sync_data = self._bag_data[sname][data_idx]
                hit_data[sname] = sync_data

        return hit_data

    def get_header(self):
        return self._topic_header
    def get_attachments(self):
        return self._topic_attachments
    def get_meta(self):
        return self._topic_meta

    def get_topic_seq_num(self, topic_name: str) -> list:
        self.load_topics(topic_name)
        return list(self.seq2idx[topic_name].keys())

    def get_valid_topic_names(self):
        return list(self._bag_data.keys())

    def get_all_topic_names(self):
        return self.all_topic_names

    def get_topic_frequency(self, topic_name) -> float:
        return self.topic_frequency_info[topic_name]

    def get_topic_msg_count(self, topic_name) -> int:
        return len(self._bag_data[topic_name])

    def get_bag_name(self)->str:
        return osp.basename(self._bag_path)

    def get_bag_path(self) -> str:
        return self._bag_path
    
    def __del__(self):
        self.close()
    
    def close(self):
        if not self._stream.closed:
            self._stream.close()

    def __repr__(self):
        msg_count = {}
        for tn in self._bag_data:
            msg_count[tn] = len(self._bag_data[tn])
        return (
            f"bag name: {osp.basename(self._bag_path)}\n"
            f"timestamp_range: [{self.msg_start_time}, {self.msg_end_time}]\n"
            f"bag time length: {ns_to_s(self.msg_end_time - self.msg_start_time):.1f} s\n"
            f"all topic names: {self.all_topic_names}\n"
            f"loaded topic names: {self.get_valid_topic_names()}\n"
            f"topic frequncy:\n"
            f"{json.dumps(self.topic_frequency_info, indent=4)}\n"
            f"topic msg count:\n"
            f"{json.dumps(msg_count, indent=4)}\n"
        )