import numpy as np
import torch
from torchvision import transforms
from datasets import load_dataset, Dataset, DatasetDict, Features, Image as DsImage
import PIL.Image as Image
from huggingface_hub import HfApi, login
import os
import tempfile


def reorganize_and_upload_dataset(
    original_dataset_name, new_dataset_name, auth_token=None
):
    if auth_token:
        login(token=auth_token)
    else:
        print(
            "No auth token provided. Make sure you're logged in with huggingface-cli login"
        )

    print(f"Loading dataset: {original_dataset_name}")
    original_dataset = load_dataset(original_dataset_name)

    split_name = (
        "train" if "train" in original_dataset else list(original_dataset.keys())[0]
    )
    dataset = original_dataset[split_name]

    transform = transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5
            ),
            transforms.RandomApply(
                [
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
                ],
                p=0.5,
            ),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
            ),
        ]
    )

    image_column = None
    for col in dataset.column_names:
        if col in ["image", "img", "pixel_values"]:
            image_column = col
            break

    if not image_column:
        example = dataset[0]
        for col, value in example.items():
            if isinstance(value, (Image.Image, np.ndarray)) or (
                hasattr(value, "shape") and len(value.shape) >= 2
            ):
                image_column = col
                break

    if not image_column:
        raise ValueError("Could not identify the image column in the dataset")

    print(f"Found image column: {image_column}")
    print(f"Creating perturbed versions of {len(dataset)} images...")

    with tempfile.TemporaryDirectory() as temp_dir:
        new_dataset = DatasetDict()

        for split in original_dataset:
            print(f"Processing split: {split}")

            good_data = {}
            bad_data = {}

            for key in original_dataset[split].features:
                if key != image_column:
                    good_data[key] = original_dataset[split][key]
                    bad_data[key] = original_dataset[split][key]

            good_images = []
            bad_images = []

            print(
                f"Processing {len(original_dataset[split])} images for split '{split}'"
            )

            for i, example in enumerate(original_dataset[split]):
                if i % 10 == 0:
                    print(f"  Processing image {i+1}/{len(original_dataset[split])}")

                orig_img = example[image_column]

                if not isinstance(orig_img, Image.Image):
                    if isinstance(orig_img, np.ndarray):
                        orig_img = Image.fromarray(orig_img)
                    else:
                        raise TypeError(f"Unsupported image type: {type(orig_img)}")

                good_images.append(orig_img)

                perturbed_img = transform(orig_img)
                bad_images.append(perturbed_img)

            good_data[image_column] = good_images
            bad_data[image_column] = bad_images

            good_features = {
                key: DsImage()
                if key == image_column
                else original_dataset[split].features[key]
                for key in good_data
            }

            bad_features = {
                key: DsImage()
                if key == image_column
                else original_dataset[split].features[key]
                for key in bad_data
            }

            good_dataset = Dataset.from_dict(
                good_data, features=Features(good_features)
            )
            bad_dataset = Dataset.from_dict(bad_data, features=Features(bad_features))

            good_dataset = good_dataset.add_column(
                "category", ["good"] * len(good_dataset)
            )
            bad_dataset = bad_dataset.add_column("category", ["bad"] * len(bad_dataset))

            split_dataset = Dataset.from_dict(
                {
                    **{
                        k: good_dataset[k] + bad_dataset[k]
                        for k in good_dataset.column_names
                        if k != "category"
                    },
                    "category": good_dataset["category"] + bad_dataset["category"],
                }
            )

            new_dataset[split] = split_dataset
            split_dataset.save_to_disk(os.path.join(temp_dir, split))

        # Push the dataset to the Hub
        print(f"Uploading reorganized dataset to {new_dataset_name}...")
        new_dataset.push_to_hub(new_dataset_name)
        print("Upload complete!")

    return new_dataset


if __name__ == "__main__":
    original_dataset = "GangGreenTemperTatum/lfw-sample"
    new_dataset_name = "GangGreenTemperTatum/lfw-sample-organized"

    reorganized_dataset = reorganize_and_upload_dataset(
        original_dataset_name=original_dataset,
        new_dataset_name=new_dataset_name,
    )
