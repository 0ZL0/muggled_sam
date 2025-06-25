#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simplified video segmentation demo using SAMURAI tracking.

This script uses OpenCV for the UI so it does not rely on the more
complex UI helpers in the repository.  It allows selecting objects
with bounding boxes or foreground/background points and shows four
mask predictions for each frame.  The mask with the highest SAMURAI
score is automatically selected and tracked when tracking is enabled.
"""
import argparse
import os.path as osp
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import torch

from lib.make_sam import make_sam_from_state_dict
from lib.v2_sam.sam_v2_model import SAMV2Model
from lib.demo_helpers.samurai import SimpleSamurai
from lib.demo_helpers.video_data_storage import SAM2VideoObjectResults
from lib.demo_helpers.misc import make_device_config, get_default_device_string


@dataclass
class UIState:
    """Simple container for user interaction state."""

    mode: str = "idle"  # one of: idle, fg, bg, box
    fg_points: List[Tuple[int, int]] = None
    bg_points: List[Tuple[int, int]] = None
    boxes: List[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    drawing: bool = False
    start_point: Tuple[int, int] | None = None
    current_box: Tuple[Tuple[int, int], Tuple[int, int]] | None = None

    def __post_init__(self):
        self.fg_points = []
        self.bg_points = []
        self.boxes = []


# ---------------------------------------------------------------------------
# Utility functions


def norm_points(points, hw):
    h, w = hw
    return [(x / w, y / h) for (x, y) in points]


def norm_boxes(boxes, hw):
    h, w = hw
    ret = []
    for (pt1, pt2) in boxes:
        x1, y1 = pt1
        x2, y2 = pt2
        ret.append(((x1 / w, y1 / h), (x2 / w, y2 / h)))
    return ret


# ---------------------------------------------------------------------------
# SAMURAI helper


def samurai_video_step(
    samurai: SimpleSamurai,
    sammodel: SAMV2Model,
    encoded_img_list,
    prompt_memory_enc,
    prompt_object_ptrs,
    prev_memory_enc,
    prev_object_ptrs,
):
    """Run a single SAMURAI tracking step."""

    with torch.inference_mode():
        lowres, *hires = encoded_img_list
        memfused = sammodel.memory_fusion(
            lowres,
            prompt_memory_enc,
            prompt_object_ptrs,
            prev_memory_enc,
            prev_object_ptrs,
        )
        patch_grid_hw = memfused.shape[2:]
        grid_posenc = sammodel.coordinate_encoder.get_grid_position_encoding(
            patch_grid_hw
        )
        mask_preds, iou_preds, obj_ptrs, obj_score = sammodel.mask_decoder(
            [memfused, *hires],
            sammodel.prompt_encoder.create_video_no_prompt_encoding(),
            grid_posenc,
            mask_hint=None,
            blank_promptless_output=False,
        )
        best_idx, best_samurai_iou, _ = samurai.get_best_decoder_results(
            mask_preds, iou_preds
        )
        best_mask = mask_preds[:, [best_idx], ...]
        best_iou = iou_preds[:, [best_idx]]
        best_ptr = obj_ptrs[:, [best_idx]]
        is_ok = samurai._check_memory_ok(obj_score, best_iou, best_samurai_iou)
        memory_enc = sammodel.memory_encoder(lowres, best_mask, obj_score)

    return (
        obj_score,
        best_idx,
        mask_preds,
        iou_preds,
        memory_enc,
        best_ptr,
        is_ok,
        float(best_samurai_iou),
    )


# ---------------------------------------------------------------------------
# Mouse callback


def mouse_cb(event, x, y, flags, param):
    state: UIState = param
    if event == cv2.EVENT_LBUTTONDOWN:
        if state.mode == "box":
            state.drawing = True
            state.start_point = (x, y)
            state.current_box = (state.start_point, (x, y))
        elif state.mode == "fg":
            state.fg_points.append((x, y))
        elif state.mode == "bg":
            state.bg_points.append((x, y))
    elif event == cv2.EVENT_MOUSEMOVE and state.drawing:
        state.current_box = (state.start_point, (x, y))
    elif event == cv2.EVENT_LBUTTONUP and state.drawing:
        state.drawing = False
        state.boxes.append((state.start_point, (x, y)))
        state.current_box = None


# ---------------------------------------------------------------------------
# Main


def main():
    parser = argparse.ArgumentParser(description="Simple SAMURAI video demo")
    parser.add_argument("-i", "--video", default=0, help="Video path or camera index")
    parser.add_argument(
        "-m",
        "--model",
        default="sam2.1_hiera_tiny.pt",
        help="Path to SAMv2 model weights",
    )
    parser.add_argument(
        "-d",
        "--device",
        default=get_default_device_string(),
        help="Device to use (cpu/cuda)",
    )
    parser.add_argument("--base_size", type=int, default=1024)
    args = parser.parse_args()

    device_cfg = make_device_config(args.device, use_float32=False)

    print("Loading model", args.model)
    _, sammodel = make_sam_from_state_dict(args.model)
    assert isinstance(sammodel, SAMV2Model)
    sammodel.to(**device_cfg)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    state = UIState()
    cv2.namedWindow("Video")
    cv2.setMouseCallback("Video", mouse_cb, state)

    memory = SAM2VideoObjectResults.create()
    samurai = None
    tracking = False
    encoded_img_list = None

    mask_preds = None
    iou_preds = None
    obj_score = 0.0
    sam_score = 0.0
    best_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        h, w = frame.shape[:2]

        # Draw annotations
        for pt in state.fg_points:
            cv2.circle(display, pt, 4, (0, 255, 0), -1)
        for pt in state.bg_points:
            cv2.circle(display, pt, 4, (0, 0, 255), -1)
        for box in state.boxes:
            cv2.rectangle(display, box[0], box[1], (255, 255, 0), 2)
        if state.drawing and state.current_box is not None:
            cv2.rectangle(display, state.current_box[0], state.current_box[1], (255, 255, 0), 1)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("f"):
            state.mode = "fg"
        elif key == ord("g"):
            state.mode = "bg"
        elif key == ord("b"):
            state.mode = "box"
        elif key == ord("h"):
            state.mode = "idle"
        elif key == ord("c"):
            state.fg_points.clear()
            state.bg_points.clear()
            state.boxes.clear()
        elif key == ord("s"):
            # Run segmentation on current frame
            encoded_img_list, _, _ = sammodel.encode_image(
                frame, args.base_size, use_square_sizing=True
            )
            boxes_norm = norm_boxes(state.boxes, (h, w))
            fg_norm = norm_points(state.fg_points, (h, w))
            bg_norm = norm_points(state.bg_points, (h, w))
            encoded_prompts = sammodel.encode_prompts(boxes_norm, fg_norm, bg_norm)
            mask_preds, iou_preds = sammodel.generate_masks(
                encoded_img_list, encoded_prompts, blank_promptless_output=True
            )
            best_idx = sammodel.get_best_mask_index(iou_preds)
            obj_score = float(iou_preds[0, best_idx])
            samurai = SimpleSamurai(mask_preds[:, [best_idx]], fps)
            tracking = False
        elif key == ord("t"):
            if not tracking and mask_preds is not None:
                # Initialize tracking
                boxes_norm = norm_boxes(state.boxes, (h, w))
                fg_norm = norm_points(state.fg_points, (h, w))
                bg_norm = norm_points(state.bg_points, (h, w))
                (
                    init_mask,
                    mem_enc,
                    obj_ptr,
                ) = sammodel.initialize_video_masking(
                    encoded_img_list,
                    boxes_norm,
                    fg_norm,
                    bg_norm,
                    mask_index_select=best_idx,
                )
                memory.store_prompt_result(0, mem_enc, obj_ptr)
                samurai = SimpleSamurai(init_mask, fps)
                tracking = True
            else:
                tracking = False
                memory.prevframe_buffer.clear()

        if tracking and samurai is not None and encoded_img_list is not None:
            encoded_img_list, _, _ = sammodel.encode_image(
                frame, args.base_size, use_square_sizing=True
            )
            step_out = samurai_video_step(
                samurai,
                sammodel,
                encoded_img_list,
                memory.prompts_buffer.memory_history,
                memory.prompts_buffer.pointer_history,
                memory.prevframe_buffer.memory_history,
                memory.prevframe_buffer.pointer_history,
            )
            (
                obj_score_t,
                best_idx,
                mask_preds,
                iou_preds,
                mem_enc,
                obj_ptr,
                ok_mem,
                sam_score,
            ) = step_out
            obj_score = float(obj_score_t.squeeze())
            if ok_mem:
                memory.store_result(0, mem_enc, obj_ptr)

        if mask_preds is not None:
            mask_disp = []
            for i in range(4):
                m = mask_preds[0, i].float().cpu().numpy()
                m = (m > 0).astype(np.uint8) * 255
                mask_disp.append(cv2.resize(m, (128, 128)))
            top = np.hstack(mask_disp[:2])
            bottom = np.hstack(mask_disp[2:])
            mask_show = np.vstack([top, bottom])
            cv2.imshow("Masks", mask_show)
            best_mask = mask_preds[0, best_idx].float().cpu().numpy()
            best_mask = cv2.resize((best_mask > 0).astype(np.uint8) * 255, (w, h))
            display_mask = cv2.cvtColor(best_mask, cv2.COLOR_GRAY2BGR)
            display = cv2.addWeighted(display, 1.0, display_mask, 0.5, 0)
            cv2.putText(
                display,
                f"IoU: {obj_score:.2f}  Samurai: {sam_score:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        cv2.imshow("Video", display)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
