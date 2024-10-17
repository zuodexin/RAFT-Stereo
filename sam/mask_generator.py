from segment_anything import (
    sam_model_registry,
    SamPredictor,
    SamAutomaticMaskGenerator,
)
from segment_anything.modeling import Sam
from segment_anything.utils.amg import MaskData, generate_crop_boxes, rle_to_mask
import logging
import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore
import os.path as osp
from typing import Any, Dict, List, Optional, Tuple
import cv2
import torch.nn.functional as F
import ipdb

pretrained_weight_dict = {
    "vit_l": "sam_vit_l_0b3195.pth",  # 1250MB
    "vit_b": "sam_vit_b_01ec64.pth",  # 375MB
    "vit_h": "sam_vit_h_4b8939.pth",  # 2500MB
}


def load_sam(model_type, checkpoint_dir, device):
    logging.info(f"Loading SAM model from {checkpoint_dir}")
    sam = sam_model_registry[model_type](
        checkpoint=osp.join(checkpoint_dir, pretrained_weight_dict[model_type])
    )
    sam.to(device=device)
    return sam


def load_sam_mask_generator(model_type, checkpoint_dir, device):
    logging.info(f"Loading SAM model from {checkpoint_dir}")
    sam = sam_model_registry[model_type](
        checkpoint=osp.join(checkpoint_dir, pretrained_weight_dict[model_type])
    )
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam, output_mode="coco_rle")
    return mask_generator


class CustomSamAutomaticMaskGenerator(SamAutomaticMaskGenerator):
    def __init__(
        self,
        sam: Sam,
        min_mask_region_area: int = 0,
        points_per_batch: int = 64,
        stability_score_thresh: float = 0.95,
        box_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        segmentor_width_size=None,
    ):
        SamAutomaticMaskGenerator.__init__(
            self,
            sam,
            min_mask_region_area=min_mask_region_area,
            points_per_batch=points_per_batch,
            stability_score_thresh=stability_score_thresh,
            box_nms_thresh=box_nms_thresh,
            crop_overlap_ratio=crop_overlap_ratio,
        )
        self.segmentor_width_size = segmentor_width_size
        logging.info(f"Init CustomSamAutomaticMaskGenerator done!")

    def preprocess_resize(self, image: np.ndarray):
        orig_size = image.shape[:2]
        height_size = int(self.segmentor_width_size * orig_size[0] / orig_size[1])
        resized_image = cv2.resize(
            image.copy(), (self.segmentor_width_size, height_size)  # (width, height)
        )
        return resized_image

    def postprocess_resize(self, detections, orig_size):
        detections["masks"] = F.interpolate(
            detections["masks"].unsqueeze(1).float(),
            size=(orig_size[0], orig_size[1]),
            mode="bilinear",
            align_corners=False,
        )[:, 0, :, :]
        scale = orig_size[1] / self.segmentor_width_size
        detections["boxes"] = detections["boxes"].float() * scale
        detections["boxes"][:, [0, 2]] = torch.clamp(
            detections["boxes"][:, [0, 2]], 0, orig_size[1] - 1
        )
        detections["boxes"][:, [1, 3]] = torch.clamp(
            detections["boxes"][:, [1, 3]], 0, orig_size[0] - 1
        )
        return detections

    @torch.no_grad()
    def generate_masks(self, image: np.ndarray) -> List[Dict[str, Any]]:
        if self.segmentor_width_size is not None:
            orig_size = image.shape[:2]
            image = self.preprocess_resize(image)
        # Generate masks
        mask_data = self._generate_masks(image)

        # Filter small disconnected regions and holes in masks
        if self.min_mask_region_area > 0:
            mask_data = self.postprocess_small_regions(
                mask_data,
                self.min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )
        if self.segmentor_width_size is not None:
            mask_data = self.postprocess_resize(mask_data, orig_size)
        return mask_data

    def _generate_masks(self, image: np.ndarray) -> MaskData:
        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )

        # Iterate over image crops
        data = MaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image, crop_box, layer_idx, orig_size)
            data.cat(crop_data)

        # Remove duplicate masks between crops
        if len(crop_boxes) > 1:
            # Prefer masks from smaller crops
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                scores,
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)

        data["masks"] = [torch.from_numpy(rle_to_mask(rle)) for rle in data["rles"]]
        data["masks"] = torch.stack(data["masks"])
        return {"masks": data["masks"].to(data["boxes"].device), "boxes": data["boxes"]}

    def remove_small_detections(self, mask_data: MaskData, img_size: List) -> MaskData:
        # calculate area and number of pixels in each mask
        area = box_area(mask_data["boxes"]) / (img_size[0] * img_size[1])
        idx_selected = area >= self.mask_post_processing.min_box_size
        mask_data.filter(idx_selected)
        return mask_data


def visualize_sam_results(image, sam_output, vis_boxes=True):
    masks = sam_output["masks"]
    boxes = sam_output["boxes"]
    canvas = image.copy().astype(np.float32)
    colors = np.random.random((len(masks), 3)) * 255

    for i, mask in enumerate(masks):
        mask = mask.cpu().numpy()
        canvas[mask > 0] = canvas[mask > 0] * 0.65 + 0.35 * colors[i]

    # show boxes
    for i, box in enumerate(boxes):
        box = box.cpu().numpy()
        cv2.rectangle(
            canvas,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            colors[i],
            2,
        )

    return canvas.astype(np.uint8)


# python -m sam.mask_generator
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="vit_h", help="model type")
    parser.add_argument(
        "--checkpoint_dir", type=str, default="models/sam", help="checkpoint directory"
    )
    parser.add_argument("--device", type=str, default="cuda", help="device")
    args = parser.parse_args()

    sam = load_sam(args.model_type, args.checkpoint_dir, args.device)
    predictor = SamPredictor(sam)
    mask_generator = load_sam_mask_generator(
        args.model_type, args.checkpoint_dir, args.device
    )
    custom_mask_generator = CustomSamAutomaticMaskGenerator(
        sam,
        mask_generator.min_mask_region_area,
        mask_generator.points_per_batch,
        mask_generator.stability_score_thresh,
        mask_generator.box_nms_thresh,
        mask_generator.crop_overlap_ratio,
        segmentor_width_size=512,
    )

    image = cv2.imread("./RAFTStereo.png")
    output = custom_mask_generator.generate_masks(image)

    vis_sam = visualize_sam_results(image, output)
    cv2.imwrite("vis_sam.png", vis_sam)
    print("Done!")
