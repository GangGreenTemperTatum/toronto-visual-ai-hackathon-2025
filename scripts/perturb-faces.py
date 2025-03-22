import numpy as np
import torch
from torchvision import transforms
from datasets import load_dataset, Dataset
import PIL.Image as Image


def perturb_huggingface_dataset(dataset_name, num_perturbations=1, output_path=None):
    original_dataset = load_dataset(dataset_name)

    split_name = (
        "train" if "train" in original_dataset else list(original_dataset.keys())[0]
    )
    dataset = original_dataset[split_name]

    # Define transformations using torchvision
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

    perturbed_metadata = []

    for i, example in enumerate(dataset):
        orig_img = example[image_column]

        # Make sure we have a PIL Image
        if not isinstance(orig_img, Image.Image):
            if isinstance(orig_img, np.ndarray):
                orig_img = Image.fromarray(orig_img)
            else:
                raise TypeError(f"Unsupported image type: {type(orig_img)}")

        for j in range(num_perturbations):
            # Apply transformations
            perturbed_img = transform(orig_img)

            # Create a new example with all the original metadata
            new_example = dict(example)
            new_example[image_column] = perturbed_img
            new_example["perturbation_id"] = j + 1
            new_example["original_id"] = i

            perturbed_metadata.append(new_example)

    # Create a new dataset with the perturbed images
    columns = list(perturbed_metadata[0].keys())
    perturbed_dataset = Dataset.from_dict(
        {col: [ex[col] for ex in perturbed_metadata] for col in columns}
    )

    if output_path:
        perturbed_dataset.save_to_disk(output_path)

    return perturbed_dataset


if __name__ == "__main__":
    dataset_name = "GangGreenTemperTatum/lfw-sample"

    perturbed_dataset = perturb_huggingface_dataset(
        dataset_name=dataset_name,
        num_perturbations=1,
        output_path="perturbed_lfw_sample",
    )
