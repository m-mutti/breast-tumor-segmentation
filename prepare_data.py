import os
import shutil

import kagglehub
from sklearn.model_selection import train_test_split


def download_busi_dataset() -> str:
    path = kagglehub.dataset_download("aryashah2k/breast-ultrasound-images-dataset")
    return path


def prepare_busi_dataset():
    import os

    DATA_DIR = os.path.join(download_busi_dataset(), "Dataset_BUSI_with_GT")
    classes = ["benign", "malignant"]  # ignore "normal" (no masks)

    image_paths = []
    mask_paths = []

    for cls in classes:
        cls_dir = os.path.join(DATA_DIR, cls)
        for fname in os.listdir(cls_dir):
            if "mask" not in fname.lower() and fname.endswith(".png"):
                mask_name = fname[:-4] + "_mask.png"
                img_path = os.path.join(cls_dir, fname)
                mask_path = os.path.join(cls_dir, mask_name)
                if os.path.exists(mask_path):
                    image_paths.append(img_path)
                    mask_paths.append(mask_path)

    # Split dataset
    trainval_image_paths, test_image_paths, trainval_mask_paths, test_mask_paths = (
        train_test_split(image_paths, mask_paths, test_size=0.2, random_state=42)
    )
    train_img_pths, val_img_pths, train_mask_pths, val_mask_pths = train_test_split(
        trainval_image_paths, trainval_mask_paths, test_size=0.2, random_state=42
    )

    return (
        train_img_pths,
        train_mask_pths,
        val_img_pths,
        val_mask_pths,
        test_image_paths,
        test_mask_paths,
    )


def save_split_data(base_dir="dataset_split"):
    splits = ["train", "val", "test"]
    for split in splits:
        os.makedirs(os.path.join(base_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, split, "masks"), exist_ok=True)

    (train_img, train_mask, val_img, val_mask, test_img, test_mask) = (
        prepare_busi_dataset()
    )
    data_map = {
        "train": (train_img, train_mask),
        "val": (val_img, val_mask),
        "test": (test_img, test_mask),
    }

    for split, (imgs, masks) in data_map.items():
        for img, mask in zip(imgs, masks):
            shutil.copy(
                img, os.path.join(base_dir, split, "images", os.path.basename(img))
            )
            shutil.copy(
                mask, os.path.join(base_dir, split, "masks", os.path.basename(mask))
            )


if __name__ == "__main__":
    save_split_data()
