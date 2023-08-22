import os
import sys
import time
import argparse
from PIL import Image
import numpy as np
import cv2

import torch
from torch.backends import cudnn
import torchvision.transforms as transforms


# sys.path.append("/home/eunsoo/dl/semantic-segmentation/")
sys.path.append(os.path.join(os.getcwd()))
sys.path.append(os.path.join(os.getcwd(), "../semantic-segmentation"))

import network
from optimizer import restore_snapshot
from datasets import cityscapes
from config import assert_and_infer_cfg


class SDCnet:
    def __init__(self, args):

        self.args = args
        assert_and_infer_cfg(args, train_mode=False)
        cudnn.benchmark = False
        torch.cuda.empty_cache()

    def getImage(self):
        data_dir = self.args.demo_folder
        images = os.listdir(data_dir)
        if len(images) == 0:
            print(
                "There are no images at directory %s. Check the data path." % (data_dir)
            )
        else:
            print("There are %d images to be processed." % (len(images)))
        images.sort()
        return images

    def getNN(self):
        self.args.dataset_cls = cityscapes
        print(self.args)
        net = network.get_net(self.args, criterion=None)
        net = torch.nn.DataParallel(net).cuda()
        print("Net built.")
        net, _ = restore_snapshot(
            net,
            optimizer=None,
            snapshot=self.args.snapshot,
            restore_optimizer_bool=False,
        )
        net.eval()
        print("Net restored.")
        return net

    def inference(self):
        images = self.getImage()
        net = self.getNN()
        masked_img_list = []
        img_list = []
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        img_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(*mean_std)]
        )
        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir)

        start_time = time.time()
        for img_id, img_name in enumerate(images):
            img_dir = os.path.join(self.args.demo_folder, img_name)
            img = Image.open(img_dir).convert("RGB")
            img_tensor = img_transform(img)

            # predict
            with torch.no_grad():
                pred = net(img_tensor.unsqueeze(0).cuda())
                print("%04d/%04d: Inference done." % (img_id + 1, len(images)))

            pred = pred.cpu().numpy().squeeze()
            pred = np.argmax(pred, axis=0)
            # print(pred)
            # print(pred.size)
            color_name = "color_mask_" + img_name
            overlap_name = "overlap_" + img_name
            pred_name = "pred_mask_" + img_name

            # save colorized predictions
            colorized = self.args.dataset_cls.colorize_mask(pred)
            colorized.save(os.path.join(self.args.save_dir, color_name))

            # save colorized predictions overlapped on original images
            overlap = cv2.addWeighted(
                np.array(img), 0.5, np.array(colorized.convert("RGB")), 0.5, 0
            )
            cv2.imwrite(
                os.path.join(self.args.save_dir, overlap_name), overlap[:, :, ::-1]
            )
            # cv2.imshow("test", overlap[:, :, ::-1])
            # cv2.waitKey(10)
            # save label-based predictions, e.g. for submission purpose
            label_out = np.zeros_like(pred)
            for label_id, train_id in self.args.dataset_cls.id_to_trainid.items():
                label_out[np.where(pred == train_id)] = label_id
                cv2.imwrite(os.path.join(self.args.save_dir, pred_name), label_out)
            masked_img_list.append(label_out)
            img_list.append(img)
        end_time = time.time()

        print("Results saved.")
        print(
            "Inference takes %4.2f seconds, which is %4.2f seconds per image, including saving results."
            % (end_time - start_time, (end_time - start_time) / len(images))
        )
        return label_out
        # return masked_img_list, img_list
