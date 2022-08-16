import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d

from scipy.spatial.transform import Rotation as R
from pyflann import *

import torch.optim as optim
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from torch.autograd import Variable
import matplotlib.pyplot as plt

# from chamfer_distance import ChamferDistance
from chamferdist import ChamferDistance
from test import add_noise


class calibTraining:
    def __init__(
        self,
        points,
        points_label,
        seg_img,
        camera_matrix,
        distortion,
        origin_img,
        epochs=280,
    ):

        # rot: tensor([1.5826, -2.1280, -0.1954], requires_grad=True)
        # trans: tensor([0.4717, -0.6444, -0.9595], requires_grad=True)
        # self.init_rotation = np.array([1.5826, -2.1280, -0.1954])
        # self.init_trans = np.array([0.4717, -0.6444, -0.9595])
        # TODO : finding origin R|T value
        self.init_rotation = np.array([63.0261, -89.0483, 26.9353]) * math.pi / 180.0
        self.init_trans = np.array([-4.069766e-03, -7.631618e-01, -2.717806e-01])
        print(f"init rot : {self.init_rotation}")
        self.gt_RPY = gt_RPY
        self.gt_translation = gt_translation
        self.origin_img = origin_img
        self.epochs = epochs
        self.test_mode = test_mode
        self.camera_matrix = camera_matrix

        self.gt_RPY
        self.gt_translation

        self.points = points
        self.points_label = points_label
        self.seg_img = seg_img

        # sorting seg_img according to ID
        # TODO : 시각화, MASK ID -> MAP화
        road_ind = np.where(self.seg_img == 7)
        self.road_seg_arr = np.concatenate(([road_ind[0]], [road_ind[1]]), axis=0)
        self.road_seg_arr = [self.road_seg_arr.transpose()]
        self.road_seg_arr = np.array(self.road_seg_arr)

        car_ind = np.where(self.seg_img == 26)
        self.car_seg_arr = np.concatenate(([car_ind[0]], [car_ind[1]]), axis=0)
        self.car_seg_arr = [self.car_seg_arr.transpose()]
        self.car_seg_arr = np.array(self.car_seg_arr)

        sidewalk_ind = np.where(self.seg_img == 8)
        self.sidewalk_seg_arr = np.concatenate(
            ([sidewalk_ind[0]], [sidewalk_ind[1]]), axis=0
        )
        self.sidewalk_seg_arr = [self.sidewalk_seg_arr.transpose()]
        self.sidewalk_seg_arr = np.array(self.sidewalk_seg_arr)

        self.height, self.width = self.seg_img.shape[:2]

        self.car_lidar_ind = 0
        self.sidewalk_lidar_ind = 10
        self.road_lidar_ind = 8
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.learning_rate = 7e-4
        self.wd = 1e-5

        self.debugProjection(self.init_rotation, self.init_trans, "before.jpg")

    def debugProjection(self, rotation, translation, name):

        RX = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, math.cos(rotation[0]), -math.sin(rotation[0])],
                [0.0, math.sin(rotation[0]), math.cos(rotation[0])],
            ]
        )
        RY = np.array(
            [
                ([math.cos(rotation[1]), 0.0, math.sin(rotation[1])]),
                ([0.0, 1.0, 0.0]),
                ([-math.sin(rotation[1]), 0.0, math.cos(rotation[1])]),
            ]
        )

        RZ = np.array(
            [
                ([math.cos(rotation[2]), -math.sin(rotation[2]), 0.0]),
                ([math.sin(rotation[2]), math.cos(rotation[2]), 0.0]),
                ([0.0, 0.0, 1.0]),
            ]
        )
        R = np.dot(RZ, RY)
        R = np.dot(R, RX)
        Tr = np.concatenate([R, translation.reshape(3, 1)], axis=1)
        ones = np.ones((self.points.shape[0], 1))
        points_homo = np.concatenate([self.points[:, :3], ones], axis=1)
        points_cam = np.dot(Tr, points_homo.T).T
        points_prj = np.dot(self.camera_matrix, points_cam.T).T
        points_prj /= points_prj[:, 2:3]
        eff_inds = (
            (points_prj[:, 0] < self.width)
            & (points_prj[:, 0] >= 0)
            & (points_prj[:, 1] < self.height)
            & (points_prj[:, 1] >= 0)
        )
        eff_inds_where = np.where(eff_inds)[0]
        eff_pcd_img = points_prj[eff_inds, :]

        prjection_img = self.origin_img.copy()
        cmap = plt.cm.get_cmap("hsv", 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
        for i in range(eff_pcd_img.shape[0]):
            pcd_cam_idx = eff_inds_where[i]
            pcd_cam_value = points_cam[pcd_cam_idx, :]
            depth = pcd_cam_value[2]
            if depth < -0 or depth > 150:
                continue
            color = cmap[int(abs(640.0 / depth)) % 256, :]
            cv2.circle(
                prjection_img,
                (int(np.round(eff_pcd_img[i, 0])), int(np.round(eff_pcd_img[i, 1]))),
                2,
                color=tuple(color),
                thickness=-1,
            )
        cv2.imwrite(name, prjection_img)

    def training_loop(self):

        dtype = torch.FloatTensor
        rotation = Variable(
            torch.Tensor(self.init_rotation).type(dtype), requires_grad=True
        )
        translation = Variable(
            torch.Tensor(self.init_trans).type(dtype), requires_grad=True
        )

        chamferDist = ChamferDistance()

        ones = np.ones((self.points.shape[0], 1))
        points = np.concatenate([self.points[:, :3], ones], axis=1)
        points = np.concatenate(
            [points, np.array([self.points_label]).transpose()], axis=1
        )
        optimizer = optim.AdamW(
            [rotation, translation], lr=self.learning_rate, weight_decay=self.wd
        )
        # .AdamW(model.parameters(),lr=0.00006,weight_decay=0.0001)
        max_rotation = []
        max_translation = []
        # self.epochs = 1
        max_dist = torch.zeros(1)
        pre_dist = torch.zeros(1)
        check_cnt = 0
        for t in range(self.epochs):

            tensor_0 = torch.zeros(1).squeeze()
            tensor_1 = torch.ones(1).squeeze()

            RX = torch.stack(
                [
                    torch.stack([tensor_1, tensor_0, tensor_0]),
                    torch.stack(
                        [tensor_0, torch.cos(rotation[0]), -torch.sin(rotation[0])]
                    ),
                    torch.stack(
                        [tensor_0, torch.sin(rotation[0]), torch.cos(rotation[0])]
                    ),
                ]
            ).reshape(3, 3)
            RY = torch.stack(
                [
                    torch.stack(
                        [torch.cos(rotation[1]), tensor_0, torch.sin(rotation[1])]
                    ),
                    torch.stack([tensor_0, tensor_1, tensor_0]),
                    torch.stack(
                        [-torch.sin(rotation[1]), tensor_0, torch.cos(rotation[1])]
                    ),
                ]
            ).reshape(3, 3)

            RZ = torch.stack(
                [
                    torch.stack(
                        [torch.cos(rotation[2]), -torch.sin(rotation[2]), tensor_0]
                    ),
                    torch.stack(
                        [torch.sin(rotation[2]), torch.cos(rotation[2]), tensor_0]
                    ),
                    torch.stack([tensor_0, tensor_0, tensor_1]),
                ]
            ).reshape(3, 3)
            R = torch.mm(RZ, RY)
            R = torch.mm(R, RX)
            Tr = torch.cat([R, translation.reshape(3, 1)], dim=1)
            ### road

            road_lidar_ind = np.where(points[:, 4] == self.road_lidar_ind)
            road_points = points[road_lidar_ind]
            normal_plane_ptr = torch.mm(
                Tr, torch.Tensor(road_points[:, :4].transpose())
            )
            normal_plane_ptr = normal_plane_ptr / normal_plane_ptr[2]
            img_plane_ptr = torch.mm(torch.Tensor(self.camera_matrix), normal_plane_ptr)

            eff_inds = (
                (img_plane_ptr[0, :] >= 0)
                & (img_plane_ptr[0, :] < self.width)
                & (img_plane_ptr[1, :] < self.height)
                & (img_plane_ptr[1, :] >= 0)
            )
            eff_points = img_plane_ptr[:, eff_inds]
            o3d.io.write_point_cloud("./log/road.pcd", eff_points)
            road_points_num = eff_points.shape[1]
            eff_points = eff_points[:2, :].reshape(-1, 2).unsqueeze(0)

            dist_road = chamferDist(
                eff_points,
                torch.Tensor(self.road_seg_arr.tolist()),
            )
            # o3d.io.write_point_cloud("./log/road.pcd", eff_points)
            ### car
            car_lidar_ind = np.where(points[:, 4] == self.car_lidar_ind)
            car_points = points[car_lidar_ind]
            normal_plane_ptr = torch.mm(Tr, torch.Tensor(car_points[:, :4].transpose()))
            normal_plane_ptr = normal_plane_ptr / normal_plane_ptr[2]
            img_plane_ptr = torch.mm(torch.Tensor(self.camera_matrix), normal_plane_ptr)

            eff_inds = (
                (img_plane_ptr[0, :] >= 0)
                & (img_plane_ptr[0, :] < self.width)
                & (img_plane_ptr[1, :] < self.height)
                & (img_plane_ptr[1, :] >= 0)
            )
            eff_points = img_plane_ptr[:, eff_inds]
            car_points_num = eff_points.shape[1]

            eff_points = eff_points[:2, :].reshape(-1, 2).unsqueeze(0)

            dist_car = chamferDist(
                eff_points,
                torch.Tensor(self.car_seg_arr.tolist()),
            )
            # TODO : 시각화
            ### sidewalk

            sidewalk_lidar_ind = np.where(points[:, 4] == self.sidewalk_lidar_ind)
            sidewalk_points = points[sidewalk_lidar_ind]
            normal_plane_ptr = torch.mm(
                Tr, torch.Tensor(sidewalk_points[:, :4].transpose())
            )
            normal_plane_ptr = normal_plane_ptr / normal_plane_ptr[2]
            img_plane_ptr = torch.mm(torch.Tensor(self.camera_matrix), normal_plane_ptr)

            eff_inds = (
                (img_plane_ptr[0, :] >= 0)
                & (img_plane_ptr[0, :] < self.width)
                & (img_plane_ptr[1, :] < self.height)
                & (img_plane_ptr[1, :] >= 0)
            )
            eff_points = img_plane_ptr[:, eff_inds]
            sidewalk_points_num = eff_points.shape[1]

            eff_points = eff_points[:2, :].reshape(-1, 2).unsqueeze(0)

            dist_sidewalk = chamferDist(
                eff_points,
                torch.Tensor(self.sidewalk_seg_arr.tolist()),
            )

            ## optimize
            optimizer.zero_grad()
            dist = (
                dist_road / torch.Tensor([road_points_num])
                + dist_car / torch.Tensor([car_points_num])
                + dist_sidewalk / torch.Tensor([sidewalk_points_num])
            )
            # dist = dist_road + dist_car + dist_sidewalk

            # dist = dist_car / torch.Tensor(
            #     [car_points_num]
            # ) + dist_sidewalk / torch.Tensor([sidewalk_points_num])
            if t == 1:
                max_dist = dist

            else:
                if dist > max_dist:
                    max_dist = dist
                    max_rotation = rotation
                    max_translation = translation

            if dist < pre_dist:
                check_cnt += 1
            else:
                check_cnt = 0
            if check_cnt > 6:
                break

            pre_dist = dist
            dist.backward()
            optimizer.step()
            print(f"iter : {t+1}")
            print(f"dist : {dist.detach().cpu().item()}")
            print(f"rot: {rotation}")
            print(f"trans : {translation}")
            self.debugProjection(
                rotation.detach().cpu().numpy(),
                translation.detach().cpu().numpy(),
                "result.jpg",
            )

        self.debugProjection(
            max_rotation.detach().cpu().numpy(),
            max_translation.detach().cpu().numpy(),
            "result.jpg",
        )
        np.savetxt("rot.txt", rotation.cpu().detach().numpy(), fmt="%f", delimiter=",")
        np.savetxt(
            "trans.txt", translation.cpu().detach().numpy(), fmt="%f", delimiter=","
        )

        return rotation.cpu().detach().numpy(), translation.cpu().detach().numpy()

    def test_mode(self, gt_RPY, gt_translation):
        error_sum_RPY = 0.0
        error_sum_translation = 0.0
        self.gt_RPY = gt_RPY
        self.gt_translation = gt_translation
        # gt_RPY = []
        # gt_translation = []
        for i in range(50):
            gt_RPY_add_noise = self.gt_RPY
            gt_translation_add_noise = self.gt_translation
            add_noise(gt_RPY_add_noise, gt_translation_add_noise)
            calib_RPY, calib_translation = calib_training.training_loop()
            error_RPY = np.mean(abs(calib_RPY - self.gt_RPY))
            error_translation = np.mean(abs(calib_translation - self.gt_translation))

        error_sum_RPY /= 50
        error_sum_translation /= 50
        return error_sum_RPY, error_sum_translation
        # print(error_sum)


# FIXME : 1. device error 확인
# TODO : GPU 사용여부 확인
# TODO : not detect 시 exception process and dist(loss) 무한
# TODO : model 로 짜기
