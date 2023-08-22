"""Visualization code for point clouds and 3D bounding boxes with mayavi.

Modified by Charles R. Qi
Date: September 2017
"""

import argparse
import os, sys

# import mayavi.mlab as mlab
import numpy as np
import open3d as o3d
import torch
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize

sys.path.append(os.path.join(os.getcwd()))
sys.path.append(os.path.join(os.getcwd(), "../spvnas"))

from model_zoo import minkunet, spvcnn, spvnas_specialized

cmap = np.array(
    [
        [245, 150, 100, 255],
        [245, 230, 100, 255],
        [150, 60, 30, 255],
        [180, 30, 80, 255],
        [255, 0, 0, 255],
        [30, 30, 255, 255],
        [200, 40, 255, 255],
        [90, 30, 150, 255],
        [255, 0, 255, 255],
        [255, 150, 255, 255],
        [75, 0, 75, 255],
        [75, 0, 175, 255],
        [0, 200, 255, 255],
        [50, 120, 255, 255],
        [0, 175, 0, 255],
        [0, 60, 135, 255],
        [80, 240, 150, 255],
        [150, 240, 255, 255],
        [0, 0, 255, 255],
    ]
)
cmap = cmap[:, [2, 1, 0, 3]]  # convert bgra to rgba


label_name_mapping = {
    0: "unlabeled",
    1: "outlier",
    10: "car",
    11: "bicycle",
    13: "bus",
    15: "motorcycle",
    16: "on-rails",
    18: "truck",
    20: "other-vehicle",
    30: "person",
    31: "bicyclist",
    32: "motorcyclist",
    40: "road",
    44: "parking",
    48: "sidewalk",
    49: "other-ground",
    50: "building",
    51: "fence",
    52: "other-structure",
    60: "lane-marking",
    70: "vegetation",
    71: "trunk",
    72: "terrain",
    80: "pole",
    81: "traffic-sign",
    99: "other-object",
    252: "moving-car",
    253: "moving-bicyclist",
    254: "moving-person",
    255: "moving-motorcyclist",
    256: "moving-on-rails",
    257: "moving-bus",
    258: "moving-truck",
    259: "moving-other-vehicle",
}

kept_labels = [
    "road",
    "sidewalk",
    "parking",
    "other-ground",
    "building",
    "car",
    "truck",
    "bicycle",
    "motorcycle",
    "other-vehicle",
    "vegetation",
    "trunk",
    "terrain",
    "person",
    "bicyclist",
    "motorcyclist",
    "fence",
    "pole",
    "traffic-sign",
]


class BinVisualizer:
    def __init__(self):
        self.points = np.zeros((0, 3), dtype=np.float32)
        self.sem_label = np.zeros((0, 1), dtype=np.uint32)  # [m, 1]: label
        self.sem_label_color = np.zeros((0, 3), dtype=np.float32)  # [m ,3]: color

        # label map
        reverse_label_name_mapping = {}
        self.label_map = np.zeros(260)
        cnt = 0
        for label_id in label_name_mapping:
            if label_id > 250:
                if label_name_mapping[label_id].replace("moving-", "") in kept_labels:
                    self.label_map[label_id] = reverse_label_name_mapping[
                        label_name_mapping[label_id].replace("moving-", "")
                    ]
                else:
                    self.label_map[label_id] = 255
            elif label_id == 0:
                self.label_map[label_id] = 255
            else:
                if label_name_mapping[label_id] in kept_labels:
                    self.label_map[label_id] = cnt
                    reverse_label_name_mapping[label_name_mapping[label_id]] = cnt
                    cnt += 1
                else:
                    self.label_map[label_id] = 255
        self.reverse_label_name_mapping = reverse_label_name_mapping

    def read_pc_label(self, points, label):
        assert points.shape[0] == label.shape[0]
        label = label.reshape(-1)
        self.sem_label = label
        self.points = points[:, :3]

    def show_cloud(self, window_name="open3d"):
        # make color table
        color_dict = {}
        for i in range(19):
            color_dict[i] = cmap[i, :]
        color_dict[255] = [0, 0, 0, 255]

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(self.points)

        cloud_color = [color_dict[i] for i in list(self.sem_label)]

        self.sem_label_color = np.array(cloud_color).reshape((-1, 4))[:, :3] / 255
        pc.colors = o3d.utility.Vector3dVector(self.sem_label_color)

        # o3d.visualization.draw_geometries([pc], window_name)

    def run_visualize(self, points, label, window_name):
        self.read_pc_label(points, label)
        self.show_cloud(window_name)


class SPVNASnet:
    def __init__(self, args):
        self.args = args
        # self.output_dir = os.path.join(self.args.velodyne_dir, "outputs")
        # os.makedirs(self.output_dir, exist_ok=True)
        self.input_point_clouds = sorted(os.listdir(self.args.velodyne_dir))
        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        print(f"self.device : {self.device}")

    def getModel(self):
        if "MinkUNet" in self.args.model:
            model = minkunet(self.args.model, pretrained=True)
        elif "SPVCNN" in self.args.model:
            model = spvcnn(self.args.model, pretrained=True)
        elif "SPVNAS" in self.args.model:
            model = spvnas_specialized(self.args.model, pretrained=True)
        else:
            raise NotImplementedError

        model = model.to(self.device)
        return model

    def process_point_cloud(
        self, input_point_cloud, input_labels=None, voxel_size=0.05
    ):
        # input_point_cloud[:, 3] = input_point_cloud[:, 3]
        pc_ = np.round(input_point_cloud[:, :3] / voxel_size)
        pc_ -= pc_.min(0, keepdims=1)

        label_map = self.create_label_map()
        if input_labels is not None:
            labels_ = label_map[input_labels & 0xFFFF].astype(
                np.int64
            )  # semantic labels
        else:
            labels_ = np.zeros(pc_.shape[0], dtype=np.int64)

        feat_ = input_point_cloud

        if input_labels is not None:
            out_pc = input_point_cloud[labels_ != labels_.max(), :3]
            pc_ = pc_[labels_ != labels_.max()]
            feat_ = feat_[labels_ != labels_.max()]
            labels_ = labels_[labels_ != labels_.max()]
        else:
            out_pc = input_point_cloud
            pc_ = pc_

        coords_, inds, inverse_map = sparse_quantize(
            pc_, return_index=True, return_inverse=True
        )
        # TODO : what is sparse_quantize, sparse tensor
        pc = np.zeros((inds.shape[0], 4))
        pc[:, :3] = pc_[inds]

        feat = feat_[inds]
        labels = labels_[inds]
        lidar = SparseTensor(torch.from_numpy(feat).float(), torch.from_numpy(pc).int())
        return {
            "pc": out_pc,
            "lidar": lidar,
            "targets": labels,
            "targets_mapped": labels_,
            "inverse_map": inverse_map,
        }

    def create_label_map(self, num_classes=19):
        name_label_mapping = {
            "unlabeled": 0,
            "outlier": 1,
            "car": 10,
            "bicycle": 11,
            "bus": 13,
            "motorcycle": 15,
            "on-rails": 16,
            "truck": 18,
            "other-vehicle": 20,
            "person": 30,
            "bicyclist": 31,
            "motorcyclist": 32,
            "road": 40,
            "parking": 44,
            "sidewalk": 48,
            "other-ground": 49,
            "building": 50,
            "fence": 51,
            "other-structure": 52,
            "lane-marking": 60,
            "vegetation": 70,
            "trunk": 71,
            "terrain": 72,
            "pole": 80,
            "traffic-sign": 81,
            "other-object": 99,
            "moving-car": 252,
            "moving-bicyclist": 253,
            "moving-person": 254,
            "moving-motorcyclist": 255,
            "moving-on-rails": 256,
            "moving-bus": 257,
            "moving-truck": 258,
            "moving-other-vehicle": 259,
        }

        for k in name_label_mapping:
            name_label_mapping[k] = name_label_mapping[k.replace("moving-", "")]
        train_label_name_mapping = {
            0: "car",
            1: "bicycle",
            2: "motorcycle",
            3: "truck",
            4: "other-vehicle",
            5: "person",
            6: "bicyclist",
            7: "motorcyclist",
            8: "road",
            9: "parking",
            10: "sidewalk",
            11: "other-ground",
            12: "building",
            13: "fence",
            14: "vegetation",
            15: "trunk",
            16: "terrain",
            17: "pole",
            18: "traffic-sign",
        }

        label_map = np.zeros(260) + num_classes
        for i in range(num_classes):
            cls_name = train_label_name_mapping[i]
            label_map[name_label_mapping[cls_name]] = min(num_classes, i)
        return label_map.astype(np.int64)

    def inference(self):
        model = self.getModel()

        for point_cloud_name in self.input_point_clouds:
            if not point_cloud_name.endswith(".bin"):
                continue
            label_file_name = point_cloud_name.replace(".bin", ".label")
            vis_file_name = point_cloud_name.replace(".bin", ".png")
            gt_file_name = point_cloud_name.replace(".bin", "_GT.png")

            pc = np.fromfile(
                f"{self.args.velodyne_dir}/{point_cloud_name}", dtype=np.float32
            ).reshape(-1, 4)
            if os.path.exists(f"{self.args.velodyne_dir}/{label_file_name}"):
                label = np.fromfile(
                    f"{self.args.velodyne_dir}/{label_file_name}", dtype=np.int32
                )
            else:
                label = None
            print(f"label : {label}")

            feed_dict = self.process_point_cloud(pc, label)

            inputs = feed_dict["lidar"].to(self.device)
            outputs = model(inputs)
            predictions = outputs.argmax(1).cpu().numpy()
            predictions = predictions[feed_dict["inverse_map"]]

            if self.args.visualize_backend == "open3d":
                # visualize prediction
                bin_vis = BinVisualizer()
                bin_vis.run_visualize(
                    feed_dict["pc"], predictions.astype(np.int32), "Predictions"
                )
                if label is not None:
                    # TODO : save as open3d type
                    bin_vis = BinVisualizer()
                    bin_vis.run_visualize(
                        feed_dict["pc"], feed_dict["targets_mapped"], "Ground turth"
                    )
            else:
                raise NotImplementedError
            return feed_dict["pc"], predictions.astype(np.int32)
            """"
            "pc": out_pc,
            "lidar": lidar,
            "targets": labels,
            "targets_mapped": labels_,
            "inverse_map": inverse_map,
            """
