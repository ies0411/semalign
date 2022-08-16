from random import uniform
import math


def add_noise(gt_RPY=None, gt_translation=None):
    # add_noise_RPY = []
    # add_noise_translation = []
    if gt_RPY == None and gt_translation == None:
        print("no param")
        return -1
    if gt_RPY is not None:
        for idx in range(3):
            gt_RPY[idx] += round(uniform(-10.0, 10.0), 1)

    if gt_translation is not None:
        for idx in range(3):
            gt_translation[idx] += round(uniform(-1.0, 1.0), 1)


def eval_score():
    return 1


# Random float:  2.5 <= x <= 10.0
