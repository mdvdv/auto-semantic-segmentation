import os
import subprocess
import sys
import urllib.request
from io import BytesIO
from typing import Any

import cv2
import numpy as np
import requests
import supervision as sv
import torch
from groundingdino.util.inference import Model
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

ACCEPTED_IMAGE_FORMATS = ["PIL", "cv2", "numpy"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not torch.cuda.is_available():
    print("WARNING: CUDA not available. GroundingDINO will run very slowly.")


def combine_detections(detections_list, overwrite_class_ids):
    if len(detections_list) == 0:
        return sv.Detections.empty()

    if overwrite_class_ids is not None and len(overwrite_class_ids) != len(
        detections_list
    ):
        raise ValueError(
            "Length of overwrite_class_ids must match the length of detections_list."
        )

    xyxy = []
    mask = []
    confidence = []
    class_id = []
    tracker_id = []

    for idx, detection in enumerate(detections_list):
        xyxy.append(detection.xyxy)

        if detection.mask is not None:
            mask.append(detection.mask)

        if detection.confidence is not None:
            confidence.append(detection.confidence)

        if detection.class_id is not None:
            if overwrite_class_ids is not None:
                # Overwrite the class IDs for the current Detections object
                class_id.append(
                    np.full_like(
                        detection.class_id, overwrite_class_ids[idx], dtype=np.int64
                    )
                )
            else:
                class_id.append(detection.class_id)

        if detection.tracker_id is not None:
            tracker_id.append(detection.tracker_id)

    xyxy = np.vstack(xyxy)
    mask = np.vstack(mask) if mask else None
    confidence = np.hstack(confidence) if confidence else None
    class_id = np.hstack(class_id) if class_id else None
    tracker_id = np.hstack(tracker_id) if tracker_id else None

    return sv.Detections(
        xyxy=xyxy,
        mask=mask,
        confidence=confidence,
        class_id=class_id,
        tracker_id=tracker_id,
    )


def load_grounding_dino():
    AUTODISTILL_CACHE_DIR = os.path.expanduser("~/.cache/autodistill")

    GROUDNING_DINO_CACHE_DIR = os.path.join(AUTODISTILL_CACHE_DIR, "groundingdino")

    GROUNDING_DINO_CONFIG_PATH = os.path.join(
        GROUDNING_DINO_CACHE_DIR, "GroundingDINO_SwinT_OGC.py"
    )
    GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(
        GROUDNING_DINO_CACHE_DIR, "groundingdino_swint_ogc.pth"
    )

    try:
        print("trying to load grounding dino directly")
        grounding_dino_model = Model(
            model_config_path=GROUNDING_DINO_CONFIG_PATH,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
            device=DEVICE,
        )
        return grounding_dino_model
    except Exception:
        print("downloading dino model weights")
        if not os.path.exists(GROUDNING_DINO_CACHE_DIR):
            os.makedirs(GROUDNING_DINO_CACHE_DIR)

        if not os.path.exists(GROUNDING_DINO_CHECKPOINT_PATH):
            url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
            urllib.request.urlretrieve(url, GROUNDING_DINO_CHECKPOINT_PATH)

        if not os.path.exists(GROUNDING_DINO_CONFIG_PATH):
            url = "https://raw.githubusercontent.com/roboflow/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
            urllib.request.urlretrieve(url, GROUNDING_DINO_CONFIG_PATH)

        grounding_dino_model = Model(
            model_config_path=GROUNDING_DINO_CONFIG_PATH,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
            device=DEVICE,
        )

        return grounding_dino_model


def load_SAM():
    # Check if segment-anything library is already installed

    AUTODISTILL_CACHE_DIR = os.path.expanduser("~/.cache/autodistill")
    SAM_CACHE_DIR = os.path.join(AUTODISTILL_CACHE_DIR, "segment_anything")
    SAM_CHECKPOINT_PATH = os.path.join(SAM_CACHE_DIR, "sam_vit_h_4b8939.pth")

    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

    # Create the destination directory if it doesn't exist
    os.makedirs(os.path.dirname(SAM_CHECKPOINT_PATH), exist_ok=True)

    # Download the file if it doesn't exist
    if not os.path.isfile(SAM_CHECKPOINT_PATH):
        urllib.request.urlretrieve(url, SAM_CHECKPOINT_PATH)

    SAM_ENCODER_VERSION = "vit_h"

    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(
        device=DEVICE
    )
    sam_predictor = SamPredictor(sam)

    return sam_predictor


def load_SAM2():
    cur_dir = os.getcwd()

    AUTODISTILL_CACHE_DIR = os.path.expanduser("~/.cache/autodistill")
    SAM_CACHE_DIR = os.path.join(AUTODISTILL_CACHE_DIR, "segment_anything_2")
    SAM_CHECKPOINT_PATH = os.path.join(SAM_CACHE_DIR, "sam2_hiera_base_plus.pth")

    url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt"

    # Create the destination directory if it doesn't exist
    os.makedirs(os.path.dirname(SAM_CHECKPOINT_PATH), exist_ok=True)

    os.chdir(SAM_CACHE_DIR)

    if not os.path.isdir("~/.cache/autodistill/segment_anything_2/segment-anything-2"):
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/facebookresearch/segment-anything-2.git",
            ]
        )

        os.chdir("segment-anything-2")

        subprocess.run(["pip", "install", "-e", "."])

    sys.path.append("~/.cache/autodistill/segment_anything_2/segment-anything-2")

    # Download the file if it doesn't exist
    if not os.path.isfile(SAM_CHECKPOINT_PATH):
        urllib.request.urlretrieve(url, SAM_CHECKPOINT_PATH)

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    checkpoint = "~/.cache/autodistill/segment_anything_2/sam2_hiera_base_plus.pth"
    checkpoint = os.path.expanduser(checkpoint)
    model_cfg = "sam2_hiera_b+.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

    os.chdir(cur_dir)

    return predictor


def load_image(
    image: Any,
    return_format: str = "numpy",
) -> Any:
    """Load an image from a file path, URI, PIL Image, or NumPy array.

    Args:
        image (Any): The image to load.
        return_format (str): The format to return the image in.

    Returns:
        Any: The image in the specified format.
    """
    if return_format not in ACCEPTED_IMAGE_FORMATS:
        raise ValueError(f"return_format must be one of {ACCEPTED_IMAGE_FORMATS}.")

    if isinstance(image, Image.Image) and return_format == "PIL":
        return image
    elif isinstance(image, Image.Image) and return_format == "cv2":
        # Channels need to be reversed for cv2
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    elif isinstance(image, Image.Image) and return_format == "numpy":
        return np.array(image)

    if isinstance(image, np.ndarray) and return_format == "PIL":
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    elif isinstance(image, np.ndarray) and return_format == "cv2":
        return image
    elif isinstance(image, np.ndarray) and return_format == "numpy":
        return image

    if isinstance(image, str) and image.startswith("http"):
        if return_format == "PIL":
            response = requests.get(image)
            return Image.open(BytesIO(response.content))
        elif return_format == "cv2" or return_format == "numpy":
            response = requests.get(image)
            pil_image = Image.open(BytesIO(response.content))
            return np.array(pil_image)
    elif os.path.isfile(image):
        if return_format == "PIL":
            return Image.open(image)
        elif return_format == "cv2":
            # Channels need to be reversed for cv2
            return cv2.cvtColor(np.array(Image.open(image)), cv2.COLOR_RGB2BGR)
        elif return_format == "numpy":
            pil_image = Image.open(image)
            return np.array(pil_image)
    else:
        raise ValueError(f"{image} is not a valid file path or URI.")
