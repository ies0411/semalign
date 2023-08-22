import argparse
import warnings

warnings.filterwarnings("ignore")


import sdc_inference
import spvnas_inference
import yaml
import cv2
import numpy as np

import training
import random
import torch
import os
from test import add_noise


def main():
    # img segmentation 수행
    parser = argparse.ArgumentParser(description="sdc")
    parser.add_argument(
        "--demo-folder",
        type=str,
        default="./img_data/",
        help="path to the folder containing demo images",
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        default="./weight/cityscapes_best.pth",
        help="pre-trained checkpoint",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="network.deepv3.DeepWV3Plus",
        help="network architecture used for inference",
    )
    parser.add_argument(
        "--save-dir", type=str, default="./results", help="path to save your results"
    )

    args = parser.parse_args()

    sdc_net = sdc_inference.SDCnet(args)
    img_mask_data = sdc_net.inference()

    data_dir = args.demo_folder
    rgb_img = os.listdir(data_dir)
    img_dir = os.path.join(args.demo_folder, rgb_img[0])
    img = cv2.imread(img_dir, cv2.IMREAD_COLOR)

    parser = argparse.ArgumentParser(description="spv")

    parser.add_argument("--velodyne-dir", type=str, default="./lidar_data")
    parser.add_argument("--model", type=str, default="SemanticKITTI_val_SPVNAS@65GMACs")
    parser.add_argument(
        "--visualize_backend",
        type=str,
        default="open3d",
        help="visualization beckend, default=open3d",
    )

    args = parser.parse_args()
    spvnas_net = spvnas_inference.SPVNASnet(args)
    lidar_points, label_points = spvnas_net.inference()
    """
            "pc": out_pc,
            "lidar": lidar,
            "targets": labels,
            "targets_mapped": labels_,
            "inverse_map": inverse_map,
    """
    parser = argparse.ArgumentParser(description="calib")
    parser.add_argument("--intrinsic-dir", type=str, default="./configs/intrinsic.yaml")
    parser.add_argument("--gt-dir", type=str, default="./configs/gt.yaml")

    parser.add_argument("--test-mode", type=bool, default=False)
    args = parser.parse_args()

    with open(args.intrinsic_dir) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    intrinsic_param = np.array(conf["camera_matrix"], dtype="float32")
    distor_param = np.array(conf["dist_coeffs"], dtype="float32")

    print("initializing..")

    calib_training = training.calibTraining(
        lidar_points,
        label_points,
        img_mask_data,
        camera_matrix=intrinsic_param,
        distortion=distor_param,
        origin_img=img,
    )
    # print(img_mask_data.tolist()

    if args.test_mode is True:
        with open(args.gt_dir) as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)

        gt_RPY = np.array(conf["rotation"], dtype="float32")
        gt_translation = np.array(conf["translation"], dtype="float32")
        error_sum_RPY, error_sum_translation = calib_training.test_mode()
        print(f"test error_sum_RPY : {error_sum_RPY}")
        print(f"test error_sum_translation : {error_sum_translation}")

    else:
        calib_training.training_loop()
        print("finish..")


if __name__ == "__main__":
    main()
