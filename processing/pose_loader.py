from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from random import randint
import json

def FormatToList(pose):
    formated_keypoints = {}
    for k, v in pose["keypoints"].items():
        formated_keypoints[k] = {"value":v['value'].tolist(), "visibility":v['visibility']}
    
    formated_dict = {"video":pose["video"], "frame":pose["frame"], "keypoints":formated_keypoints}

    return formated_dict

def FormatToArray(pose):
    formated_keypoints = {}
    for k, v in pose["keypoints"].items():
        formated_keypoints[k] = {"value":np.array(v['value']), "visibility":v['visibility']}
    
    formated_dict = {"video":pose["video"], "frame":pose["frame"], "keypoints":formated_keypoints}

    return formated_dict

def SavePoses(poses, path = "data/poses.json"):
    with open(path, "w") as f:
        json.dump([FormatToList(pose) for pose in poses], f)

def load_poses(path):
    with open(path, "r") as f:
        poses = json.load(f)
    poses = [FormatToArray(pose) for pose in poses]
    return poses