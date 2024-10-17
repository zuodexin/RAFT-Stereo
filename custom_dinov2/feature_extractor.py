import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import ipdb
import numpy as np

import sys

from custom_dinov2.pca import dinov2_pca


class CustomFeatureExtractor(torch.nn.Module):
    def __init__(self, device="cuda"):
        super(CustomFeatureExtractor, self).__init__()
        self.dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        self.dinov2.to(device)

        self.device = device

        self.num_patch = 40
        self.patch_size = 14

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.inv_normalize = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
                ),
                transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
            ]
        )

    def preprocess_resize(self, image):
        """
        Resize the input image by adjusting the long side to target_size while maintaining aspect ratio.
        Padding is added to make the image square (target_size x target_size).
        """
        # Get original size
        original_size = image.size  # (width, height)
        image = transforms.ToTensor()(image).to(self.device)
        image = self.normalize(image)

        # Define target size for the long side
        target_size = self.num_patch * self.patch_size

        # Calculate new size for resizing while maintaining aspect ratio
        ratio = target_size / max(original_size)
        new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))

        # Resize the image while maintaining aspect ratio
        image = F.interpolate(
            image.unsqueeze(0),
            size=new_size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        # Calculate padding to make the image square (224x224)
        delta_w = target_size - new_size[0]
        delta_h = target_size - new_size[1]
        padding = (
            delta_w // 2,
            delta_w - (delta_w // 2),
            delta_h // 2,
            delta_h - (delta_h // 2),
        )
        # Pad the image and convert to tensor
        image = F.pad(image, padding)  # Pad with black (0) by default

        return (
            image,
            padding,
            target_size,
            original_size,
        )  # Also return padding and original size for postprocessing

    def postprocess_resize(self, features, padding, target_size, original_size):
        """
        Reverse the preprocessing steps by removing padding and resizing the feature map back to the original size.
        """
        # Remove the padding from the feature map
        bs, fdim = features.shape[0], features.shape[1]
        # Crop the feature map to remove padding

        # new_grid = A@Grid_xy
        features = F.grid_sample(
            features,
            F.affine_grid(
                torch.tensor(
                    [
                        [
                            [
                                (target_size - padding[0] - padding[1]) / target_size,
                                0,
                                0,
                            ],
                            [
                                0,
                                (target_size - padding[2] - padding[3]) / target_size,
                                0,
                            ],
                        ]
                    ]
                ).to(self.device),
                (bs, fdim, original_size[1], original_size[0]),
            ),
        )

        return features.squeeze(0)

    @torch.no_grad()
    def extract_features(self, image):
        image, padding, target_size, original_size = self.preprocess_resize(image)
        features_dict = self.dinov2.forward_features(image)
        features = features_dict["x_norm_patchtokens"]
        features = features.permute(0, 2, 1).reshape(
            features.shape[0], -1, self.num_patch, self.num_patch
        )

        # features = F.interpolate(
        #     image,
        #     size=(self.num_patch, self.num_patch),
        #     mode="bilinear",
        #     align_corners=False,
        # )

        features = self.postprocess_resize(
            features, padding, target_size, original_size
        )
        return features


# python -m dinov2.feature_extractor
if __name__ == "__main__":
    dinov2 = CustomFeatureExtractor().cuda()

    image = Image.open("./RAFTStereo.png").convert("RGB")

    output = dinov2.extract_features(image)

    pca_features = dinov2_pca(output.unsqueeze(0).cpu()).squeeze(0)

    cv2.imwrite(
        "output.png",
        # (self.inv_normalize(features).permute(1, 2, 0).cpu().numpy() * 255).astype(
        #     np.uint8
        # ),
        (pca_features[:3, :, :].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8),
    )

    print("Done!")
