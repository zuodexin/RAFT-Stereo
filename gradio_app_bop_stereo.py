# This demo needs to be run from the repo folder.
# python demo/fake_gan/run.py
import json
import sys
from turtle import color, left

sys.path.append("core")

import ipdb
import fire
import random
import os
import open3d as o3d
import plotly.graph_objs as go
import torch.nn.functional as F

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
import gradio as gr
from PIL import Image
from functools import partial
import numpy as np
import torch
from easydict import EasyDict as edict
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from raft_stereo import RAFTStereo
from utils.utils import InputPadder

_TITLE = """RAFT Stereo for ROBI"""
DEVICE = "cuda"


custom_theme = gr.themes.Soft(primary_hue="blue").set(
    button_secondary_background_fill="*neutral_100",
    button_secondary_background_fill_hover="*neutral_200",
)
custom_css = """#disp_image {
    text-align: center; /* Horizontally center the content */
}"""


def preprocess(left_image, right_image, camera_cfg):

    left_image = np.array(left_image.convert("RGB")).astype(np.uint8)
    left_image = torch.from_numpy(left_image).permute(2, 0, 1).float()
    left_image = left_image[None].to(DEVICE)

    right_image = np.array(right_image.convert("RGB")).astype(np.uint8)
    right_image = torch.from_numpy(right_image).permute(2, 0, 1).float()
    right_image = right_image[None].to(DEVICE)

    # shift right image
    shift = camera_cfg.shift
    right_image = F.grid_sample(
        right_image,
        F.affine_grid(
            torch.tensor(
                [
                    [
                        [1, 0, shift / right_image.shape[3]],
                        [0, 1, 0],
                    ]
                ]
            ).to(DEVICE),
            right_image.shape,
        ),
    )

    return left_image, right_image


def apply_disparity(img, disp):  # gets a warped output
    batch_size, _, height, width = img.size()

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, width).repeat(batch_size, height, 1).to(img)
    y_base = (
        torch.linspace(0, 1, height)
        .repeat(batch_size, width, 1)
        .transpose(1, 2)
        .to(img)
    )

    # Apply shift in X direction
    x_shifts = (
        disp[:, 0, :, :] / width
    )  # Disparity is passed in NCHW format with 1 channel
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(
        img,
        2 * flow_field - 1,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    return output


def predict(model, camera_cfg, left_image, right_image, iters=32):
    image1, image2 = preprocess(left_image, right_image, camera_cfg)

    padder = InputPadder(image1.shape, divis_by=32)
    image1_pad, image2_pad = padder.pad(image1, image2)

    with torch.no_grad():
        _, flow_up = model(image1_pad, image2_pad, iters=iters, test_mode=True)
        overlap = 0.5 * apply_disparity(image2, flow_up) + image1 * 0.5
        overlap = overlap.squeeze(0).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        flow_up = padder.unpad(flow_up).cpu().numpy().squeeze()

    disparty = -flow_up

    depth = diparity_to_depth(disparty, camera_cfg)
    norm = mcolors.Normalize(vmin=disparty.min(), vmax=disparty.max())
    disp_show = (cm.jet(norm(disparty))[:, :, :3] * 255).astype(np.uint8)

    norm_depth = mcolors.Normalize(vmin=depth.min(), vmax=depth.clip(0, 2).max())
    depth_show = (cm.jet(norm_depth(depth.clip(0, 2)))[:, :, :3] * 255).astype(np.uint8)

    points = depth2pointcloud(depth, camera_cfg)
    points = points - points.mean(axis=0, keepdims=True)
    # random sample
    indices = np.random.choice(
        points.shape[0], min(500000, points.shape[0]), replace=False
    )
    # indices = np.arange(points.shape[0])
    points = points[indices]
    colors = image1.permute(0, 2, 3, 1).flatten(0, 2).cpu().numpy() / 255
    colors = colors[indices]

    # save to ply
    # point_path = "/tmp/pointcloud.ply"
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.colors = o3d.utility.Vector3dVector(
    #     image1.permute(0, 2, 3, 1).flatten(0, 2).cpu().numpy() / 255
    # )
    # o3d.io.write_point_cloud(point_path, pcd, write_ascii=True)
    trace = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=dict(
            size=2,
            color=colors,
        ),
    )
    layout = go.Layout(
        scene=dict(
            xaxis=dict(
                title="",
                showgrid=False,
                zeroline=False,
                showline=False,
                ticks="",
                showticklabels=False,
            ),
            yaxis=dict(
                title="",
                showgrid=False,
                zeroline=False,
                showline=False,
                ticks="",
                showticklabels=False,
            ),
            zaxis=dict(
                title="",
                showgrid=False,
                zeroline=False,
                showline=False,
                ticks="",
                showticklabels=False,
            ),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False,
    )

    fig = go.Figure(data=[trace], layout=layout)

    return (
        Image.fromarray(disp_show),
        Image.fromarray(overlap),
        Image.fromarray(depth_show),
        fig,
    )


def depth2pointcloud(depth, camera_cfg):
    H, W = depth.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x = x - camera_cfg.cx
    y = y - camera_cfg.cy
    z = depth
    x = x * z / camera_cfg.fx
    y = y * z / camera_cfg.fy
    return np.stack([x, y, z], axis=-1).flatten().reshape(-1, 3)


def diparity_to_depth(disp, camera_cfg):
    disp = disp.astype(np.float32)
    depth = camera_cfg.baseline_dis * camera_cfg.fx / disp
    return depth


def raft_stereo_init():
    args = edict(
        mixed_precision=False,
        # architecture
        hidden_dims=[128] * 3,
        corr_implementation="alt",
        shared_backbone=False,
        corr_levels=4,
        corr_radius=4,
        n_downsample=2,
        context_norm="batch",
        slow_fast_gru=False,
        n_gru_layers=3,
        restore_ckpt="models/raftstereo-middlebury.pth",
    )

    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    return model.eval()


def load_bop_examples(dataset_root, split, split_type=None, num_examples=5):

    scene_folder = f"{dataset_root}/{split}_{split_type}"
    left_example_fns = []
    right_example_fns = []

    for scene_id in os.listdir(scene_folder):
        im_id = os.listdir(f"{scene_folder}/{scene_id}/rgb")[0]
        left_example_fns.append(f"{scene_folder}/{scene_id}/rgb/{im_id}")
        right_example_fns.append(f"{scene_folder}/{scene_id}/rgb_r/{im_id}")

    camera = edict(json.load(open(f"{dataset_root}/camera_{split_type}.json")))

    camera_left = edict(
        json.load(open(f"{dataset_root}/camera_{split_type}_left.json"))
    )
    camera_right = edict(
        json.load(open(f"{dataset_root}/camera_{split_type}_right.json"))
    )
    shift = camera_right.cx - camera_left.cx
    camera.shift = shift

    return left_example_fns[:num_examples], right_example_fns[:num_examples], camera


def run_demo():
    left_example_fns, right_example_fns, camera_cfg = load_bop_examples(
        dataset_root="/home/dexin/projects/bin-picking/data/ROBI/bop/robi",
        split="test",
        split_type="ensenso",
        num_examples=30,
    )
    raft_stereo_model = raft_stereo_init()
    with gr.Blocks(title=_TITLE, theme=custom_theme, css=custom_css) as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("# " + _TITLE)

        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                left_image = gr.Image(
                    type="pil",
                    label="Left Image",
                    # interactive=False,
                    height=320,
                    image_mode="RGB",
                    elem_id="left_image",
                    visible=True,
                )
            with gr.Column(scale=1):
                right_image = gr.Image(
                    type="pil",
                    label="Right Image",
                    # interactive=False,
                    height=320,
                    image_mode="RGB",
                    elem_id="right_image",
                    visible=True,
                )
        with gr.Row():
            with gr.Column(scale=1):
                submit_btn = gr.Button("Submit", variant="primary", interactive=True)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Disparty")
                out_disp_image = gr.Image(
                    type="pil",
                    label="Disparty Image",
                    interactive=False,
                    height=320,
                    image_mode="RGB",
                    elem_id="output_image",
                    visible=True,
                )
            with gr.Column(scale=1):
                gr.Markdown("## Warped Right")
                out_mixup = gr.Image(
                    type="pil",
                    label="Warped Right",
                    interactive=False,
                    height=320,
                    image_mode="RGB",
                    elem_id="warped_right",
                    visible=True,
                )
            with gr.Column(scale=1):
                gr.Markdown("## Depth")
                out_depth_image = gr.Image(
                    type="pil",
                    label="Depth Image",
                    interactive=False,
                    height=320,
                    image_mode="RGB",
                    elem_id="output_image",
                    visible=True,
                )
            with gr.Column(scale=1):
                gr.Markdown("## Point Cloud")
                with gr.Column(scale=1.0):
                    pc_plot = gr.Plot(label="Inferred point cloud")

        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                gr.Examples(
                    examples=list(map(list, zip(left_example_fns, right_example_fns))),
                    inputs=[left_image, right_image],
                    cache_examples=False,
                    label="Examples (click one of the images below to start)",
                    examples_per_page=5,
                )

        submit_btn.click(
            fn=partial(predict, raft_stereo_model, camera_cfg),
            inputs=[left_image, right_image],
            outputs=[out_disp_image, out_mixup, out_depth_image, pc_plot],
            queue=True,
        )
        demo.queue().launch(share=False, max_threads=80)


if __name__ == "__main__":

    fire.Fire(run_demo)
