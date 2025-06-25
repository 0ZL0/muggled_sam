#!/usr/bin/env python3
"""Simple OpenCV UI for SAMURAI video tracking."""

import argparse
import cv2
import numpy as np
import torch
from typing import List, Tuple

from lib.make_sam import make_sam_from_state_dict
from lib.v2_sam.sam_v2_model import SAMV2Model
from lib.demo_helpers.samurai import SimpleSamurai
from lib.demo_helpers.video_data_storage import SAM2VideoObjectResults


# -----------------------------------------------------------------------------
# Utility functions


def create_hires_mask_uint8(mask_prediction: torch.Tensor, output_hw: Tuple[int, int]) -> np.ndarray:
    """Upscale mask prediction to ``output_hw`` and return as uint8."""
    mask_up = torch.nn.functional.interpolate(
        mask_prediction, size=output_hw, mode="bilinear", align_corners=False
    )
    return (mask_up[0, 0] > 0).byte().cpu().numpy()


def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.5) -> np.ndarray:
    """Overlay binary mask onto image with some transparency."""
    overlay = np.zeros_like(image)
    overlay[:] = color
    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    out = image.copy()
    out[mask3 > 0] = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)[mask3 > 0]
    return out


def samurai_step_video_masking(
    samurai: SimpleSamurai,
    sammodel: SAMV2Model,
    encoded_image_features_list: List[torch.Tensor],
    prompt_memory_encodings: List[torch.Tensor],
    prompt_object_pointers: List[torch.Tensor],
    previous_memory_encodings: List[torch.Tensor],
    previous_object_pointers: List[torch.Tensor],
):
    """Perform a single SAMURAI tracking step."""
    with torch.inference_mode():
        lowres_imgenc, *hires_imgenc = encoded_image_features_list
        memfused = sammodel.memory_fusion(
            lowres_imgenc,
            prompt_memory_encodings,
            prompt_object_pointers,
            previous_memory_encodings,
            previous_object_pointers,
        )
        patch_hw = memfused.shape[2:]
        grid_posenc = sammodel.coordinate_encoder.get_grid_position_encoding(patch_hw)
        mask_preds, iou_preds, obj_ptrs, obj_score = sammodel.mask_decoder(
            [memfused, *hires_imgenc],
            sammodel.prompt_encoder.create_video_no_prompt_encoding(),
            grid_posenc,
            mask_hint=None,
            blank_promptless_output=False,
        )
        best_idx, best_samurai_iou, _ = samurai.get_best_decoder_results(mask_preds, iou_preds)
        best_mask_pred = mask_preds[:, [best_idx], ...]
        best_iou_pred = iou_preds[:, [best_idx], ...]
        best_obj_ptr = obj_ptrs[:, [best_idx], ...]
        is_ok_mem = samurai._check_memory_ok(obj_score, best_iou_pred, best_samurai_iou)
        memory_encoding = sammodel.memory_encoder(lowres_imgenc, best_mask_pred, obj_score)
    return (
        obj_score,
        int(best_idx),
        float(best_iou_pred.squeeze()),
        float(best_samurai_iou),
        mask_preds,
        memory_encoding,
        best_obj_ptr,
        is_ok_mem,
    )


# -----------------------------------------------------------------------------
# Main


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple SAMURAI video demo")
    parser.add_argument("--video", default=None, help="Video path, otherwise webcam")
    parser.add_argument("--model", default="sam2.1_hiera_tiny.pt", help="Model weights")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--base_size", type=int, default=1024)
    args = parser.parse_args()

    print("Loading model...", flush=True)
    _, sammodel = make_sam_from_state_dict(args.model)
    assert isinstance(sammodel, SAMV2Model), "SAMv2 model required"
    sammodel.to(args.device)
    sammodel.eval()

    cap = cv2.VideoCapture(0 if args.video is None else args.video)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read video frame")

    frame_h, frame_w = frame.shape[:2]
    encoded_img, _, pre_hw = sammodel.encode_image(frame, max_side_length=args.base_size)

    fg_points: List[Tuple[float, float]] = []
    bg_points: List[Tuple[float, float]] = []
    boxes: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    text_prompt = ""

    drawing_box = False
    box_start = None
    current_tool = "point"  # or 'box'

    memory = SAM2VideoObjectResults.create()
    samurai = None
    tracking = False

    selected_mask_idx = 0
    mask_preds = None
    obj_score = None
    sam_score = None

    # mouse callback for prompt input
    def mouse_cb(event, x, y, flags, param):
        nonlocal drawing_box, box_start
        if event == cv2.EVENT_LBUTTONDOWN:
            if current_tool == "box":
                drawing_box = True
                box_start = (x, y)
            else:
                fg_points.append((x / frame_w, y / frame_h))
        elif event == cv2.EVENT_MOUSEMOVE and drawing_box:
            pass
        elif event == cv2.EVENT_LBUTTONUP and drawing_box:
            drawing_box = False
            end_pt = (x, y)
            boxes.append(((box_start[0] / frame_w, box_start[1] / frame_h), (end_pt[0] / frame_w, end_pt[1] / frame_h)))
        elif event == cv2.EVENT_RBUTTONDOWN and current_tool != "box":
            bg_points.append((x / frame_w, y / frame_h))

    cv2.namedWindow("Video")
    cv2.setMouseCallback("Video", mouse_cb)
    for mi in range(4):
        cv2.namedWindow(f"Mask{mi}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        encoded_img, _, pre_hw = sammodel.encode_image(frame, max_side_length=args.base_size)

        if tracking:
            res = samurai_step_video_masking(
                samurai,
                sammodel,
                encoded_img,
                memory.prompts_buffer.memory_history,
                memory.prompts_buffer.pointer_history,
                memory.prevframe_buffer.memory_history,
                memory.prevframe_buffer.pointer_history,
            )
            obj_score, selected_mask_idx, iou_val, sam_score, mask_preds, mem_enc, obj_ptr, ok_mem = res
            obj_score = float(obj_score.squeeze().cpu().numpy())
            if ok_mem:
                memory.store_result(frame_idx, mem_enc, obj_ptr)
        else:
            if fg_points or bg_points or boxes:
                enc_prompts = sammodel.encode_prompts(boxes, fg_points, bg_points)
                mask_preds, iou_preds = sammodel.generate_masks(encoded_img, enc_prompts, blank_promptless_output=True)
                selected_mask_idx = int(sammodel.get_best_mask_index(iou_preds))
                obj_score = float(iou_preds[0, selected_mask_idx].cpu()) * 100
                sam_score = None
            else:
                mask_preds = None
                obj_score = None
                sam_score = None

        display = frame.copy()
        if mask_preds is not None:
            mask = create_hires_mask_uint8(mask_preds[:, [selected_mask_idx]], pre_hw)
            display = overlay_mask(display, mask)
        y0 = 25
        if obj_score is not None:
            cv2.putText(display, f"Score: {obj_score:.2f}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y0 += 25
        if sam_score is not None:
            cv2.putText(display, f"SAMURAI: {sam_score:.2f}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if drawing_box and box_start is not None:
            cv2.rectangle(display, box_start, (x, y), (0, 255, 255), 1)

        cv2.imshow("Video", display)

        if mask_preds is not None:
            for mi in range(4):
                mp = create_hires_mask_uint8(mask_preds[:, [mi]], pre_hw)
                cv2.imshow(f"Mask{mi}", mp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("b"):
            current_tool = "box" if current_tool != "box" else "point"
        elif key == ord("c"):
            fg_points.clear()
            bg_points.clear()
            boxes.clear()
        elif key == ord("p") and not tracking and mask_preds is not None:
            text_prompt = input("Enter text prompt (optional): ")
        elif key == ord("s") and not tracking and mask_preds is not None:
            init_mask, mem_enc, obj_ptr = sammodel.initialize_video_masking(
                encoded_img,
                boxes,
                fg_points,
                bg_points,
                mask_index_select=selected_mask_idx,
            )
            memory.store_prompt_result(frame_idx, mem_enc, obj_ptr)
            samurai = SimpleSamurai(init_mask, video_framerate=30, smoothness=0.5)
            tracking = True
            fg_points.clear()
            bg_points.clear()
            boxes.clear()
        elif key == ord("t") and tracking:
            tracking = False
            memory.prevframe_buffer.clear()
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
