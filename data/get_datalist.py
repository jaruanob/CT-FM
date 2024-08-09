from pathlib import Path
import pandas as pd
from lighter.utils.misc import apply_fns
from totalsegmentator.map_to_binary import class_map

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

    images = images[: int(len(images) * percentage / 100)]
    labels = labels[: int(len(labels) * percentage / 100)]

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
     
