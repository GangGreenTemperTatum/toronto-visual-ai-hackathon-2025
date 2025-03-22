from datasets import load_dataset

"""
Dataset Structure: GangGreenTemperTatum/lfw-sample-organized

This dataset contains facial images organized into two categories:
- "good": Original, unmodified facial images from the LFW (Labeled Faces in the Wild) dataset
- "bad": Perturbed versions of the same facial images with various transformations applied

The dataset is structured as follows:
1. Main split: "train" containing all images
2. Each example in the dataset has at least these fields:
   - "image": PIL Image object containing the facial image
   - "category": String label ("good" or "bad") indicating if the image is original or perturbed
   - Other metadata fields from the original LFW dataset (may include person name, etc.)

The dataset is stored in Parquet format on Hugging Face Hub, which provides efficient storage
and retrieval. When loaded with the datasets library, the Parquet files are automatically
converted back to usable Python objects (PIL Images, strings, etc.).

This dataset is designed for training models to distinguish between original and manipulated
facial images, which can be useful for deepfake detection and image authentication tasks.
"""

dataset = load_dataset("GangGreenTemperTatum/lfw-sample-organized")

example = dataset["train"][0]

image = example["image"]
category = example["category"]

import matplotlib.pyplot as plt

plt.imshow(image)
plt.title(f"Category: {category}")
plt.axis("off")
plt.show()
