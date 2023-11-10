import os
import cv2
import numpy as np
import imgaug.augmenters as iaa
from ultralytics import YOLO, SAM
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd


CELEBA_DATA_PATH = "./"
IMG_PATH = os.path.join(CELEBA_DATA_PATH, "/img_align_celeba")


def getImagePath(image_id):
    return os.path.join(IMG_PATH, image_id)


def load_image(source_image_path):
    image = cv2.imread(source_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_bounding_box(yolo_model, image):
    person_class_index = 0  # ultralytics class dictionary
    results = yolo_model.predict(
        image, conf=0.5, classes=[person_class_index], verbose=False
    )
    for result in results:
        if result:
            boxes = result.boxes
            bbox = boxes.xyxy.tolist()[0]

        else:
            height, width, _ = image.shape
            bbox = [0, 0, width, height]
    return bbox


def apply_segmentation(sam_model, image, bbox):
    sam_output = sam_model.predict(image, bboxes=bbox, verbose=False)
    mask_object = sam_output[0].masks
    mask_tensor = mask_object.data
    mask_np = mask_tensor.numpy()
    mask_2d = mask_np.squeeze(axis=0)
    return mask_2d


def apply_mask(image, mask_2d):
    masked_image = np.zeros_like(image)
    for c in range(image.shape[2]):  # Assuming image has 3 channels
        masked_image[..., c] = image[..., c] * mask_2d
    return masked_image


def crop_and_resize(masked_image, mask_2d, resize_dims):
    y_indices, x_indices = np.where(mask_2d)
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    cropped_image = masked_image[y_min : y_max + 1, x_min : x_max + 1]
    resized_image = cv2.resize(cropped_image, resize_dims)
    normalized_image = resized_image / 255.0  # Scale pixel values to [0, 1]
    return normalized_image


def augment_images(images, augmenter):
    augmented_images = []
    for img in images:
        # Check if image data type is uint8, if not convert to uint8
        if img.dtype != np.uint8:
            img_uint8 = (img * 255).astype(np.uint8)
            augmented = augmenter(image=img_uint8)
        else:
            augmented = augmenter(image=img)
        augmented_images.append(augmented)
    return augmented_images


def process_batch(batch, yolo_model, sam_model, resize_dims, augmenter, pbar=None):
    # batch here is batch_paths
    batch_results = []
    for source_image_path in batch:
        try:
            image = load_image(source_image_path)
            bbox = get_bounding_box(yolo_model, image)
            mask_2d = apply_segmentation(sam_model, image, bbox)
            masked_image = apply_mask(image, mask_2d)
            normalized_image = crop_and_resize(masked_image, mask_2d, resize_dims)
            batch_results.append(normalized_image)
        except Exception as e:
            print(f"Failed to process image {source_image_path}: {e}")
            if pbar is not None:
                pbar.update(1)  # Update the progress bar even if there's an error
            continue  # Skip the rest of the loop and proceed with the next image
        if pbar is not None:
            pbar.update(1)
    try:
        augmented_batch = augment_images(batch_results, augmenter)
        #         display_batch(augmented_batch)
        save_batch_images(batch_results, batch, CELEBA_DATA_PATH)
    except Exception as e:
        print(f"Failed to augment and save images: {e}")
    return augmented_batch


def display_batch(images, figsize=(8, 8), columns=5):
    plt.figure(figsize=figsize)
    for i, image in enumerate(images):
        plt.subplot(len(images) // columns + 1, columns, i + 1)
        plt.imshow(image)
        plt.text(5, 15, str(i), color="white", fontsize=12, weight="bold")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def process_partition(
    df_partition, yolo_model, sam_model, resize_dims, augmenter, batch_size=32
):
    num_images = len(df_partition)
    processed_images = []
    with tqdm(total=num_images, desc="Processing partition", unit="img") as pbar:
        for start_idx in range(0, num_images, batch_size):
            end_idx = min(start_idx + batch_size, num_images)
            # get image paths from the partitioned_df
            batch_paths = [
                getImagePath(df_partition.iloc[i]["image_id"])
                for i in range(start_idx, end_idx)
            ]
            processed_batch = process_batch(
                batch_paths, yolo_model, sam_model, resize_dims, augmenter, pbar
            )
            processed_images.extend(processed_batch)
    return processed_images


def save_batch_images(batch_images, batch_image_paths, base_output_dir):
    """
    Save a batch of images to the specified base output directory.

    :param batch_images: List of image data to be saved.
    :param batch_image_paths: List of source paths of images, used to extract the image IDs.
    :param base_output_dir: Base directory where the 'processed_img' folder will be created and images will be saved.
    """
    # Create the output directory if it does not exist
    output_dir = os.path.join(base_output_dir, "processed_img")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save each image in the batch
    for img, path in zip(batch_images, batch_image_paths):
        image_id = os.path.basename(path)
        output_image_path = os.path.join(output_dir, image_id)
        img_pil = Image.fromarray(
            (img * 255).astype("uint8")
        )  # Convert from [0,1] to [0,255] and to uint8
        img_pil.save(output_image_path, format="JPEG")
