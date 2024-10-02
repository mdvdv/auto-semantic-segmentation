import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import supervision as sv
import torch

from autosemantic.detection.caption_ontology import CaptionOntology
from autosemantic.detection.detection_base_model import DetectionBaseModel
from autosemantic.helpers import (combine_detections, load_grounding_dino,
                                  load_image, load_SAM2)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SamPredictor = load_SAM2()

SUPPORTED_GROUNDING_MODELS = ["Grounding DINO"]


@dataclass
class GroundedSAM2(DetectionBaseModel):
    ontology: CaptionOntology
    box_threshold: float
    text_threshold: float

    def __init__(
        self,
        ontology: CaptionOntology,
        model="Grounding DINO",
        grounding_dino_box_threshold=0.35,
        grounding_dino_text_threshold=0.25,
    ):
        if model not in SUPPORTED_GROUNDING_MODELS:
            raise ValueError(
                f"Grounding model {model} is not supported. Supported models are {SUPPORTED_GROUNDING_MODELS}"
            )

        self.ontology = ontology
        if model == "Grounding DINO":
            self.grounding_dino_model = load_grounding_dino()
        self.sam_2_predictor = SamPredictor
        self.model = model
        self.grounding_dino_box_threshold = grounding_dino_box_threshold
        self.grounding_dino_text_threshold = grounding_dino_text_threshold

    def predict(self, input: Any) -> sv.Detections:
        image = load_image(input, return_format="cv2")

        if self.model == "Grounding DINO":
            # GroundingDINO predictions
            detections_list = []

            for i, description in enumerate(self.ontology.prompts()):
                # Detect objects
                detections = self.grounding_dino_model.predict_with_classes(
                    image=image,
                    classes=[description],
                    box_threshold=self.grounding_dino_box_threshold,
                    text_threshold=self.grounding_dino_text_threshold,
                )

                detections_list.append(detections)

            detections = combine_detections(
                detections_list, overwrite_class_ids=range(len(detections_list))
            )

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.sam_2_predictor.set_image(image)
            result_masks = []
            for box in detections.xyxy:
                masks, scores, _ = self.sam_2_predictor.predict(
                    box=box, multimask_output=False
                )
                index = np.argmax(scores)
                masks = masks.astype(bool)
                result_masks.append(masks[index])

        detections.mask = np.array(result_masks)

        return detections
