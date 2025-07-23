import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
from ast import literal_eval

from processing.pose_loader import load_poses
from processing.pose_processing import normalise_pose

def load_rts(merge_metadata = False, scene_min_duration = 4):
    rts_features = pd.read_pickle("data/rts_features.pkl")
    rts_features.drop(columns=['face_features', 'landmark_features'], inplace=True)
    
    if merge_metadata:
        metadata = pd.read_hdf("data/rts_metadata.hdf5")
        metadata.rename({"mediaId":"umid"}, axis = 1, inplace = True)
        metadata_per_umid = metadata[["umid", "mediaFolderPath", "contentType", "collection"]].fillna("MISSING").groupby("umid").agg(list).reset_index()
        # Assign most common contentType and collection
        metadata_per_umid["contentType"] = metadata_per_umid.contentType.map(lambda x: max(set(x), key=x.count))
        metadata_per_umid["collection"] = metadata_per_umid.collection.map(lambda x: max(set(x), key=x.count))

        rts_features = rts_features.merge(metadata_per_umid, on = "umid", how = "left")
        
    rts_features = rts_features.explode(column="imagenet_features")
    rts_features.dropna(inplace=True)
    rts_features = rts_features[rts_features["scene_mean_duration"] > scene_min_duration]
    
    return rts_features
    
def load_pdl_poses():
    POSES_FOLDER = "data/lp_poses_every5/"
    poses_pdl = []
    poses_fp = os.listdir(POSES_FOLDER)
    for pose_fp in poses_fp:
        poses = load_poses(POSES_FOLDER + pose_fp)
        for pose in poses:
            pose["video"] = pose["video"].replace("D:/", "E:/")
        poses_pdl.extend(poses)

    poses_pdl = [normalise_pose(pose, torso_size_multiplier = 2.5) for pose in poses_pdl if pose["keypoints"] is not None]
    return poses_pdl
    
def load_ioc_poses():
    poses_ioc = pd.read_csv("data/poses_ioc.csv", converters={"embedding_33": literal_eval})
    return poses_ioc
    
def load_mjf(merge_metadata = False):
    genres = pd.read_csv("data/mjf_vectors_genre.csv")
    genres["media_id"] = genres.media_id.map(lambda x: int(x.split("-")[1]))
    genres.sort_values("media_id", inplace = True)

    instruments = pd.read_csv("data/mjf_vectors_instrument.csv")
    instruments["media_id"] = instruments.media_id.map(lambda x: int(x.split("-")[1]))
    instruments.sort_values("media_id", inplace = True)

    moods = pd.read_csv("data/mjf_vectors_mood.csv")
    moods["media_id"] = moods.media_id.map(lambda x: int(x.split("-")[1]))
    moods.sort_values("media_id", inplace = True)
    
    mjf_features = genres.copy()
    mjf_features["genres_f"] = genres.drop("media_id", axis = 1).agg(list, axis = 1).tolist()
    mjf_features["insts_f"] = instruments.drop("media_id", axis = 1).agg(list, axis = 1).tolist()
    mjf_features["moods_f"] = moods.drop("media_id", axis = 1).agg(list, axis = 1).tolist()

    mjf_features = mjf_features[["media_id", "genres_f", "insts_f", "moods_f"]]
    
    if merge_metadata:
        with open("data/mjf_metadata.json", "r") as f:
            mjf_metadata = json.load(f)
        # TODO: merge metadata
        
    return mjf_features

def load_datasets(datasets_to_load = ["rts", "pdl", "ioc", "mjf"], merge_metadata = False):
    datasets = {}
    if "rts" in datasets_to_load:
        rts_features = load_rts(merge_metadata = merge_metadata)
        datasets["rts"] = np.array(rts_features["imagenet_features"].tolist())
        
    if "pdl" in datasets_to_load:
        pdl_features = load_pdl_poses()
        datasets["pdl"] = np.array([list(pose["flat_embedding"]) for pose in pdl_features])
        
    if "ioc" in datasets_to_load:
        ioc_features = load_ioc_poses()
        datasets["ioc"] = np.array(ioc_features["embedding_33"].tolist())
        
    if "mjf" in datasets_to_load:
        mjf_features = load_mjf(merge_metadata = merge_metadata)
        datasets["mjf"] = np.array(mjf_features["genres_f"].tolist())
        
    if "mnist" in datasets_to_load:
        datasets["mnist"] = np.load('DATA/mnist_pca.npy', allow_pickle=True).reshape(70000, -1)
    
    print("Features loaded:")
    for name,data in datasets.items():
        print(f"{name}: {data.shape[0]} samples of dimension {data.shape[1]}")
    
    return datasets
    