import math
from pathlib import Path
import random
import pandas as pd
from lighter.utils.misc import apply_fns
from totalsegmentator.map_to_binary import class_map
import requests
import json


BODY_PART_IDS = {
    "organs_v1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    "vertebra_v1": [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41],
    "cardiac_v1": [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 104, 93],
    "muscles_v1": [82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103],
    "ribs_v1": [58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81],
    "merlin_v2": [6, 5, 4, 3, 2, 1, 22, 32, 31, 30, 29, 28, 27, 26, 25, 21, 20, 19, 18, 7],
    "v1": list(range(1, 105)),
    "v2": list(range(1, 118))
}


def get_ts_class_indices(group="v1"):
    assert group in BODY_PART_IDS.keys()
    class_indices = sorted([0] + BODY_PART_IDS[group])
    print(f"Number of classes: {len(class_indices)}")
    return class_indices


def get_ts_class_labels(class_indices, group="v1"):
    mapping_dict = class_map["total_v1" if "v1" in group else "total"]
    return ["background" if idx == 0 else mapping_dict[idx] for idx in class_indices]


def get_msd_datalist(data_dir, split, task="Task06_Lung"):
    assert task in ["Task06_Lung", "Task03_Liver", "Task07_Pancreas", "Task08_HepaticVessel"]
    assert split in ["train", "val", "test"]
    data_dir = Path(data_dir)
    task_id = task.split("_")[0]

    vista_json_url = f"https://raw.githubusercontent.com/Project-MONAI/VISTA/main/vista3d/data/jsons/{task_id}_5_folds.json"
    response = requests.get(vista_json_url)
    vista_json_data = response.json()

    match split:
        case "train":
            return [{"image": data_dir / task / item["image"], "label": data_dir  / task / item["label"], "id": item["image"].split(".")[0]} 
                            for item in vista_json_data["training"] if item["fold"] != 4]
        case "val":
            return [{"image": data_dir / task /item["image"], "label": data_dir  / task / item["label"], "id": item["image"].split(".")[0]} 
                            for item in vista_json_data["training"] if item["fold"] == 4]
        case "test":
            return [{"image": data_dir / task / item["image"], "label": data_dir / task / item["label"], "id": item["image"].split(".")[0]} 
                            for item in vista_json_data["testing"]]
    

def get_word_datalist(data_dir, split, samples=None):
    assert split in ["train", "val", "test"]
    data_dir = Path(data_dir)

    vista_json_url = "https://raw.githubusercontent.com/Project-MONAI/VISTA/main/vista3d/data/external/WORD.json"
    response = requests.get(vista_json_url)
    vista_json_data = response.json()

    match split:
        case "train":
            datalist = [{"image": data_dir / item["image"], "label": data_dir  / item["label"], "id": item["image"].split(".")[0]} 
                            for item in vista_json_data["training"]]
        case "val":
            datalist = [{"image": data_dir / item["image"], "label": data_dir  / item["label"], "id": item["image"].split(".")[0]} 
                            for item in vista_json_data["validation"]]
        case "test":
            datalist = [{"image": data_dir / item["image"], "label": data_dir /  item["label"], "id": item["image"].split(".")[0]} 
                            for item in vista_json_data["testing"]]
            
    if samples is not None:
        datalist = datalist[:samples]

    return datalist

def get_word_class_labels():
    vista_json_url = "https://raw.githubusercontent.com/Project-MONAI/VISTA/main/vista3d/data/external/WORD.json"
    response = requests.get(vista_json_url)
    vista_json_data = response.json()
    return vista_json_data["labels"].values()    

def get_ts_datalist(data_dir, percentage=100, filter_fn=[]):
    """
    Get the list of image and label paths for a given split.
    """
    data_dir = Path(data_dir)
    meta_df = pd.read_csv(data_dir / "meta.csv")
    meta_df = apply_fns(meta_df, filter_fn)

    ids = meta_df["image_id"].tolist()

    images = [data_dir / id / "ct.nii.gz" for id in ids]
    labels = [data_dir / id / "label.nii.gz" for id in ids]

    images = images[: math.ceil(len(images) * percentage / 100)]
    labels = labels[: math.ceil(len(labels) * percentage / 100)]

    print(f"Number of images: {len(images)}")

    return [
        {"image": image, "label": label, "id": id}
        for image, label, id in zip(images, labels, ids)
    ]


def get_lits_datalist(data_dir, split, test_size=0.4):
    image_paths = sorted(list(Path(data_dir).glob("volume*.nii")))
    label_paths = sorted(list(Path(data_dir).glob("segmentation*.nii")))

    assert len(image_paths) == len(label_paths)
    datalist = [{"image": image, "label": label} for image, label in zip(image_paths, label_paths)]

    if split == "train":
        return datalist[:int(len(datalist) * (1 - test_size))]

    if split == "val":
        return datalist[int(len(datalist) * (1 - test_size)):]


def get_radchest_datalist(data_dir, split):
    def get_label_values(df):
        verified_labels = ["nodule", "mass", "opacity", "consolidation", "atelectasis", "pleural_effusion", "pneumothorax", "pericardial_effusion", "cardiomegaly"]
        for label in verified_labels:
            df[label] = df.filter(like=label).any(axis=1)

        df = df.sort_values(by=["NoteAcc_DEID"])
        return df[verified_labels].values
    
    if split == "train":
        images_files = sorted([str(path) for path in Path(data_dir).glob("trn*.npz")])
        df = pd.read_csv(f"{data_dir}/imgtrain_Abnormality_and_Location_Labels.csv")
        labels = get_label_values(df)

    if split == "val":
        images_files = sorted([str(path) for path in Path(data_dir).glob("val*.npz")])
        df = pd.read_csv(f"{data_dir}/imgvalid_Abnormality_and_Location_Labels.csv")
        labels = get_label_values(df)

    if split == "test":
        images_files = sorted([str(path) for path in Path(data_dir).glob("tst*.npz")])
        df = pd.read_csv(f"{data_dir}/imgtest_Abnormality_and_Location_Labels.csv")
        labels = get_label_values(df)

    return [{"image": image, "label": label} for image, label in zip(images_files, labels)]


def _get_patient_dirs(data_dir, percentage, split, split_ratio=(0.8, 0.1, 0.1), seed=0):
    """Get the paths to the images and labels for the train, val, or test split.
    Assumes that the data_dir contains a directory for each patient.

    Args:
        data_dir (str): data directory
        percentage (int): percentage of the dataset to use. Used for train and val, test is always 100%.
        split (str): train, val, or test
        split_ratio (tuple, optional): Ratios or number of samples for train/val/test.
        For example (0.8, 0.1, 0.1) or (100, 10, 10) if you have 120 samples in total.
        Defaults to (0.8, 0.1, 0.1).
        seed (int, optional): seed for reproducibility. Defaults to 0.

    Returns:
        List[Dict[str, Path]]: list of dictionaries containing the image and label paths
    """
    if split not in ["train", "val", "test"]:
        raise ValueError("split must be one of 'train', 'val', or 'test'")

    if not (all(isinstance(x, int) for x in split_ratio) or all(isinstance(x, float) for x in split_ratio)):
        raise ValueError("split_ratio must be a tuple of integers or floats")

    # Get the patient directories
    patient_dirs = sorted([x for x in Path(data_dir).iterdir() if x.is_dir()])

    # Check if split_ratio is actually a ratio or a number of samples
    is_ratio = isinstance(split_ratio[0], float)

    # Check if split_ratio is a ratio and if it sums to 1.0
    if is_ratio and sum(split_ratio) != 1.0:
        raise ValueError("split_ratio must sum to 1.0 if it is a ratio")

    # Check if split_ratio is a number of samples and if it sums to the number of samples
    if not is_ratio and sum(split_ratio) != len(patient_dirs):
        raise ValueError("split_ratio must sum to the number of samples if it is a number of samples")

    # Shuffle the patient directories with a fixed seed for reproducibility
    random.seed(seed)
    random.shuffle(patient_dirs)

    # Get the train, val, and test split indices
    if is_ratio:
        train_split_idx = int(split_ratio[0] * len(patient_dirs))
        val_split_idx = int((split_ratio[0] + split_ratio[1]) * len(patient_dirs))
    else:
        train_split_idx = split_ratio[0]
        val_split_idx = split_ratio[0] + split_ratio[1]

    # Split the data into train, val, and test sets
    if split == "train":
        patient_dirs = patient_dirs[:train_split_idx]
    elif split == "val":
        patient_dirs = patient_dirs[train_split_idx:val_split_idx]
    else:
        patient_dirs = patient_dirs[val_split_idx:]

    if split != "test":
        # Get the number of samples to use. Test is always 100%.
        num_samples = int(percentage / 100 * len(patient_dirs))
        print(f"Using {percentage}% (n={num_samples}) of the {split} set with {len(patient_dirs)} images.")
        # Get the subset of patient directories
        patient_dirs = patient_dirs[:num_samples]

    return patient_dirs


def get_sinoct_datalist(data_dir, percentage, split, split_ratio=(0.8, 0.1, 0.1), seed=0):
    labels_df = pd.read_csv(Path(data_dir) / "labels.csv", index_col="patient_id")
    # Create "abnormal" column by using "label" column. Set to 0 (normal) if label is "1,0", and 1 (abnormal) if "0,1".
    labels_df["abnormal"] = labels_df["label"].apply(lambda x: int(x.split(",")[1]))
    # Drop all the other columns except for patient_id and abnormal
    labels_df = labels_df[["abnormal"]]

    datalist = []
    patient_dirs = _get_patient_dirs(data_dir, percentage, split, split_ratio, seed)
    for patient_dir in patient_dirs:
        datalist.append({
            "id": patient_dir.name,
            "image": patient_dir / "nrrd" / "scan.nrrd",
            "label": labels_df.loc[patient_dir.name, "abnormal"],
        })
    return datalist


def get_cq500_datalist(data_dir, percentage, split, split_ratio=(0.0, 0.0, 1.0), seed=0):
    labels_df = pd.read_csv(Path(data_dir) / "reads.csv", index_col="name")
    # Drop Category
    labels_df = labels_df.drop(columns=["Category"])
    # All the remaining columns, except for name, should be combined into a single one
    # called "abnormal", with 1 if any of the columns is 1, and 0 otherwise.
    labels_df["abnormal"] = labels_df.any(axis=1).astype(int)
    # Drop all the other columns except for name and abnormal
    labels_df = labels_df[["abnormal"]]

    datalist = []
    patient_dirs = _get_patient_dirs(data_dir, percentage, split, split_ratio, seed)
    for patient_dir in patient_dirs:
        # Get the number of the CT and rename it to CQ500-CT-<number>. They are labeled like that in reads.csv.
        patient_id = f"CQ500-CT-{patient_dir.name.split(' ')[0].replace('CQ500CT', '')}"
        datalist.append({
            "id": patient_id,
            # Get the path to the CT scan
            "image": list(patient_dir.rglob("scan.nrrd"))[0],
            "label": labels_df.loc[patient_id, "abnormal"],
        })
    return datalist

