from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from random import randint
import json
from io import BytesIO
from PIL import Image
import base64

SIMPLIFIED_KEYPOINTS = ["NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"]
SIMPLIFIED_CONNECTIONS = [(1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (1, 7), (2, 8), (7, 8), (7, 9), (8, 10), (9, 11), (10, 12)]

def GetPoseCenter(keypoints):
    left_hip = keypoints["LEFT_HIP"]["value"]
    right_hip = keypoints["RIGHT_HIP"]["value"]
    center = (left_hip + right_hip) / 2
    return center

def GetPoseSize(keypoints, torso_size_multiplier):
    left_hip = keypoints["LEFT_HIP"]["value"]
    right_hip = keypoints["RIGHT_HIP"]["value"]
    hips = (left_hip + right_hip) / 2

    left_shoulder = keypoints["LEFT_SHOULDER"]["value"]
    right_shoulder = keypoints["RIGHT_SHOULDER"]["value"]
    shoulders = (left_shoulder + right_shoulder) / 2

    torso_size = np.linalg.norm(shoulders - hips)

    center = GetPoseCenter(keypoints)
    max_dist = np.max(max([np.linalg.norm(center - value["value"]) for key, value in keypoints.items()]))

    return max(max_dist, torso_size_multiplier * torso_size)

def GetDistanceBetweenJoints(keypoints, joint1, joint2):
    joint1 = keypoints[joint1]["value"]
    joint2 = keypoints[joint2]["value"]
    return joint1 - joint2

def GetAverageOfJoints(keypoints, joint1, joint2):
    joint1 = keypoints[joint1]["value"]
    joint2 = keypoints[joint2]["value"]
    return (joint1 + joint2) * 0.5

def GetDistance(joint1, joint2):
    return joint1 - joint2

def GetAngleBetweenJoints(keypoints, joint1, joint2, joint3):
    v1 = GetDistanceBetweenJoints(keypoints, joint1, joint2)
    v2 = GetDistanceBetweenJoints(keypoints, joint3, joint2)
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def FlattenKeypoints(keypoints):
    embedding = [v["value"].tolist() for k,v in keypoints.items()]
    embedding = np.array(embedding).flatten()
    return embedding

def PoseDistanceEmbedding(keypoints):
    embedding = np.array([
        # Distance between the hips and the ankles
        GetDistance(
            GetAverageOfJoints(keypoints, "LEFT_HIP", "RIGHT_HIP"),
            GetAverageOfJoints(keypoints, "LEFT_ANKLE", "RIGHT_ANKLE")
        ),

        # Distance between the hips and the shoulders
        
        # GetDistance(
        #     GetAverageOfJoints(keypoints, "LEFT_HIP", "RIGHT_HIP"),
        #     GetAverageOfJoints(keypoints, "LEFT_SHOULDER", "RIGHT_SHOULDER")
        # ),
        

        # Distances between opposite joints
        GetDistanceBetweenJoints(keypoints, "LEFT_ELBOW", "RIGHT_ELBOW"),
        GetDistanceBetweenJoints(keypoints, "LEFT_WRIST", "RIGHT_WRIST"),
        GetDistanceBetweenJoints(keypoints, "LEFT_KNEE", "RIGHT_KNEE"),
        GetDistanceBetweenJoints(keypoints, "LEFT_ANKLE", "RIGHT_ANKLE"),

        # Distances between joints separated by a single joint
        GetDistanceBetweenJoints(keypoints, "LEFT_HIP", "LEFT_WRIST"),
        GetDistanceBetweenJoints(keypoints, "RIGHT_HIP", "RIGHT_WRIST"),
        GetDistanceBetweenJoints(keypoints, "LEFT_HIP", "LEFT_ANKLE"),
        GetDistanceBetweenJoints(keypoints, "RIGHT_HIP", "RIGHT_ANKLE"),

        # Distances between joints separated by two joints
        GetDistanceBetweenJoints(keypoints, "LEFT_ANKLE", "LEFT_WRIST"),
        GetDistanceBetweenJoints(keypoints, "RIGHT_ANKLE", "RIGHT_WRIST"),

        # Distances between joints cross body
        GetDistanceBetweenJoints(keypoints, "LEFT_SHOULDER", "RIGHT_ANKLE"),
        GetDistanceBetweenJoints(keypoints, "RIGHT_SHOULDER", "LEFT_ANKLE"),
        GetDistanceBetweenJoints(keypoints, "LEFT_WRIST", "RIGHT_ANKLE"),
        GetDistanceBetweenJoints(keypoints, "RIGHT_WRIST", "LEFT_ANKLE"),
    ]).flatten()

    # Angles between joints
    angles = np.array([
            GetAngleBetweenJoints(keypoints, "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
            GetAngleBetweenJoints(keypoints, "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),

            GetAngleBetweenJoints(keypoints, "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
            GetAngleBetweenJoints(keypoints, "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),

            GetAngleBetweenJoints(keypoints, "LEFT_HIP", "LEFT_SHOULDER", "LEFT_ELBOW"),
            GetAngleBetweenJoints(keypoints, "RIGHT_HIP", "RIGHT_SHOULDER", "RIGHT_ELBOW"),

            GetAngleBetweenJoints(keypoints, "LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"),
            GetAngleBetweenJoints(keypoints, "RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"),

            GetAngleBetweenJoints(keypoints, "LEFT_ELBOW", "LEFT_WRIST", "LEFT_INDEX"),
            GetAngleBetweenJoints(keypoints, "RIGHT_ELBOW", "RIGHT_WRIST", "RIGHT_INDEX"),

            GetAngleBetweenJoints(keypoints, "LEFT_KNEE", "LEFT_ANKLE", "LEFT_FOOT_INDEX"),
            GetAngleBetweenJoints(keypoints, "RIGHT_KNEE", "RIGHT_ANKLE", "RIGHT_FOOT_INDEX")
        ])

    embedding = np.concatenate((FlattenKeypoints(keypoints), embedding, angles))
    return embedding

def PoseAnglesEmbedding(keypoints):
    hips_center = GetAverageOfJoints(keypoints, "LEFT_HIP", "RIGHT_HIP")
    shoulders_center = GetAverageOfJoints(keypoints, "LEFT_SHOULDER", "RIGHT_SHOULDER")
    head_to_shoulders_d = keypoints["NOSE"]["value"] - shoulders_center
    hips_to_shoulders_d = hips_center - shoulders_center
    spine_angle = np.arccos(np.dot(head_to_shoulders_d, hips_to_shoulders_d) / (np.linalg.norm(head_to_shoulders_d) * np.linalg.norm(hips_to_shoulders_d)))

    angles = np.array([
            spine_angle,

            GetAngleBetweenJoints(keypoints, "LEFT_HIP", "LEFT_SHOULDER", "LEFT_ELBOW"), # left shoulder angle
            GetAngleBetweenJoints(keypoints, "RIGHT_HIP", "RIGHT_SHOULDER", "RIGHT_ELBOW"), # right shoulder angle

            GetAngleBetweenJoints(keypoints, "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"), # left elbow angle
            GetAngleBetweenJoints(keypoints, "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"), # right elbow angle

            GetAngleBetweenJoints(keypoints, "LEFT_ELBOW", "LEFT_WRIST", "LEFT_INDEX"), # left wrist angle
            GetAngleBetweenJoints(keypoints, "RIGHT_ELBOW", "RIGHT_WRIST", "RIGHT_INDEX"), # right wrist angle

            GetAngleBetweenJoints(keypoints, "LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"), # left hip angle
            GetAngleBetweenJoints(keypoints, "RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"), # right hip angle

            GetAngleBetweenJoints(keypoints, "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"), # left knee angle
            GetAngleBetweenJoints(keypoints, "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"), # right knee angle

            GetAngleBetweenJoints(keypoints, "LEFT_KNEE", "LEFT_ANKLE", "LEFT_FOOT_INDEX"), # left ankle angle
            GetAngleBetweenJoints(keypoints, "RIGHT_KNEE", "RIGHT_ANKLE", "RIGHT_FOOT_INDEX") # right ankle angle
        ])

    return angles


def ComputeEmbedding(keypoints, mode):
    if mode == "flatten":
        return FlattenKeypoints(keypoints)
    elif mode == "distances":
        return PoseDistanceEmbedding(keypoints)
    else:
        return np.NaN
    

def SimplifyKeypoints(pose):
    keypoints = pose["keypoints"]
    simplified_keypoints = {key: keypoints[key] for key in SIMPLIFIED_KEYPOINTS}
    return simplified_keypoints


def normalise_pose(pose, torso_size_multiplier=1.0):
    keypoints = pose["keypoints"]
    center = GetPoseCenter(keypoints)
    normalized_pose = {key: (value["value"] - center) for key, value in keypoints.items()}
    size = GetPoseSize(keypoints, torso_size_multiplier)
    normalized_pose = {key: value / size for key, value in normalized_pose.items()}

    pose["keypoints_normalized"] = {key: {"value": value / size} for key,value in normalized_pose.items()}
    pose["flat_embedding"] = ComputeEmbedding(pose["keypoints_normalized"], mode = "flatten")
    pose["distances_embedding"] = ComputeEmbedding(pose["keypoints_normalized"], mode = "distances")
    pose["angles_embedding"] = PoseAnglesEmbedding(pose["keypoints_normalized"])
    pose["keypoints_simplified"] = SimplifyKeypoints(pose)
    pose["center"] = center
    pose["size"] = size
    return pose

def embed_pose(pose):
    keypoints = pose["keypoints_simplified"]
    fig = plt.figure(figsize=(4,4))
    plt.scatter([value["value"][0] for key, value in keypoints.items()], [value["value"][1] for key, value in keypoints.items()], color = "black")
    for connection in SIMPLIFIED_CONNECTIONS:
        start = SIMPLIFIED_KEYPOINTS[connection[0]]
        end = SIMPLIFIED_KEYPOINTS[connection[1]]
        plt.plot([keypoints[start]["value"][0], keypoints[end]["value"][0]], [keypoints[start]["value"][1], keypoints[end]["value"][1]], color = "black")
    plt.axis("off")

    buffer = BytesIO()
    fig.savefig(buffer)
    buffer.seek(0)
    img = Image.open(buffer).resize((64,64), Image.Resampling.BICUBIC)
    img.save(buffer, format="png")
    for_encoding = buffer.getvalue()

    plt.close(fig)
    
    return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()

def create_annotation_image(pose, linewidth = 1, color = "black", transparent = True):
    keypoints = pose["keypoints_simplified"]
    fig = plt.figure(figsize=(4,4))
    plt.scatter([value["value"][0] for key, value in keypoints.items()], [value["value"][1] for key, value in keypoints.items()], color = color)
    for connection in SIMPLIFIED_CONNECTIONS:
        start = SIMPLIFIED_KEYPOINTS[connection[0]]
        end = SIMPLIFIED_KEYPOINTS[connection[1]]
        plt.plot([keypoints[start]["value"][0], keypoints[end]["value"][0]], [keypoints[start]["value"][1], keypoints[end]["value"][1]], color = color, linewidth = linewidth)
    plt.axis("off")
    plt.gca().patch.set_alpha(0)
    buffer = BytesIO()
    fig.savefig(buffer)
    buffer.seek(0)
    plt.close(fig)
    
    img = Image.open(buffer).resize((64,64), Image.Resampling.BICUBIC)

    if transparent:
        img = img.convert("RGBA")  
        datas = img.getdata()
        newData = []
        for item in datas:
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)
        img.putdata(newData)
        
    return img