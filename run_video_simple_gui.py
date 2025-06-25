import argparse
import cv2
import numpy as np
import torch

from lib.make_sam import make_sam_from_state_dict
from lib.v2_sam.sam_v2_model import SAMV2Model
from lib.demo_helpers.samurai import SimpleSamurai
from lib.demo_helpers.video_data_storage import SAM2VideoObjectResults
from lib.demo_helpers.misc import make_device_config, get_default_device_string
from lib.demo_helpers.shared_ui_layout import make_hires_mask_uint8


def samurai_step_video_masking(
    samurai: SimpleSamurai,
    sammodel: SAMV2Model,
    encoded_image_features_list: list,
    prompt_memory_encodings,
    prompt_object_pointers,
    previous_memory_encodings,
    previous_object_pointers,
):
    """Run a single SAMURAI tracking step and return useful data."""
    with torch.inference_mode():
        lowres_imgenc, *hires_imgenc = encoded_image_features_list
        memfused = sammodel.memory_fusion(
            lowres_imgenc,
            prompt_memory_encodings,
            prompt_object_pointers,
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
        best_idx, samurai_score, _ = samurai.get_best_decoder_results(mask_preds, iou_preds)
        best_mask_pred = mask_preds[:, [best_idx], ...]
        best_iou_pred = iou_preds[:, [best_idx], ...]
        best_obj_ptr = obj_ptrs[:, [best_idx], ...]
        is_ok_mem = samurai._check_memory_ok(obj_score, best_iou_pred, samurai_score)
        memory_encoding = sammodel.memory_encoder(lowres_imgenc, best_mask_pred, obj_score)
    return (
        obj_score,
        best_idx,
        mask_preds,
        iou_preds,
        samurai_score,
        memory_encoding,
        best_obj_ptr,
        is_ok_mem,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Simple video UI for SAMURAI")
    parser.add_argument("-i", "--video", default=None, help="Path to video (default webcam)")
    parser.add_argument("-m", "--model", default="sam2.1_hiera_tiny.pt", help="Model weights")
    parser.add_argument("-d", "--device", default=get_default_device_string(), help="Device to use")
    parser.add_argument("-f32", "--use_float32", action="store_true", help="Use float32 weights")
    parser.add_argument("-b", "--base_size", default=1024, type=int, help="Base model size")
    parser.add_argument("--max_memories", default=6, type=int, help="Max memory history")
    parser.add_argument("--max_pointers", default=15, type=int, help="Max pointer history")
    parser.add_argument("--object_threshold", default=0.0, type=float, help="Tracking score threshold")
    return parser.parse_args()


def main():
    args = parse_args()
    device_cfg = make_device_config(args.device, args.use_float32)
    model_cfg, sammodel = make_sam_from_state_dict(args.model)
    assert isinstance(sammodel, SAMV2Model), "Only SAMv2 models supported"
    sammodel.to(**device_cfg)
    imgenc_cfg = {"max_side_length": args.base_size, "use_square_sizing": True}

    video_src = 0 if args.video is None else args.video
    cap = cv2.VideoCapture(video_src)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    cv2.namedWindow("Video")
    cv2.namedWindow("Masks")

    smooth_val = 50
    def _on_smooth(x):
        nonlocal smooth_val
        smooth_val = x
    cv2.createTrackbar("Smoothness", "Video", smooth_val, 100, _on_smooth)

    frame_h, frame_w = None, None
    mode = "fg"  # fg, bg, box
    fg_points, bg_points, boxes = [], [], []
    box_start = None
    current_box = None

    samurai = None
    memory = SAM2VideoObjectResults.create(args.max_memories, args.max_pointers)
    mask_preds, iou_preds, best_idx, samurai_score = None, None, 0, 0.0

    def mouse_cb(event, x, y, flags, param):
        nonlocal box_start, current_box, fg_points, bg_points, boxes
        if frame_w is None or frame_h is None:
            return
        if mode in {"fg", "bg"}:
            if event == cv2.EVENT_LBUTTONDOWN:
                pt = (x / frame_w, y / frame_h)
                if mode == "fg":
                    fg_points.append(pt)
                else:
                    bg_points.append(pt)
        elif mode == "box":
            if event == cv2.EVENT_LBUTTONDOWN:
                box_start = (x, y)
                current_box = (box_start, box_start)
            elif event == cv2.EVENT_MOUSEMOVE and box_start is not None:
                current_box = (box_start, (x, y))
            elif event == cv2.EVENT_LBUTTONUP and box_start is not None:
                x1, y1 = box_start
                x2, y2 = x, y
                tl = (min(x1, x2) / frame_w, min(y1, y2) / frame_h)
                br = (max(x1, x2) / frame_w, max(y1, y2) / frame_h)
                boxes.append((tl, br))
                box_start = None
                current_box = None

    cv2.setMouseCallback("Video", mouse_cb)

    frame_idx = 0
    tracking = False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_h, frame_w = frame.shape[:2]
        display = frame.copy()

        if tracking and samurai is not None:
            enc_img, _, _ = sammodel.encode_image(frame, **imgenc_cfg)
            (
                obj_score,
                best_idx,
                mask_preds,
                iou_preds,
                samurai_score,
                mem_enc,
                obj_ptr,
                ok_mem,
            ) = samurai_step_video_masking(
                samurai,
                sammodel,
                enc_img,
                memory.prompts_buffer.memory_history,
                memory.prompts_buffer.pointer_history,
                memory.prevframe_buffer.memory_history,
                memory.prevframe_buffer.pointer_history,
            )
            if obj_score.item() >= args.object_threshold and ok_mem:
                memory.store_result(frame_idx, mem_enc, obj_ptr)
            mask_uint8 = make_hires_mask_uint8(mask_preds[:, best_idx], frame.shape[:2])
            overlay = frame.copy()
            overlay[mask_uint8 > 0] = (0.5 * overlay[mask_uint8 > 0] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
            display = overlay
            cv2.putText(
                display,
                f"SAM {float(iou_preds[0,best_idx]):.2f}  SAMURAI {float(samurai_score):.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
        else:
            # draw prompts
            for pt in fg_points:
                cv2.circle(display, (int(pt[0]*frame_w), int(pt[1]*frame_h)), 3, (0,255,0), -1)
            for pt in bg_points:
                cv2.circle(display, (int(pt[0]*frame_w), int(pt[1]*frame_h)), 3, (0,0,255), -1)
            for tl, br in boxes:
                tl_px = (int(tl[0]*frame_w), int(tl[1]*frame_h))
                br_px = (int(br[0]*frame_w), int(br[1]*frame_h))
                cv2.rectangle(display, tl_px, br_px, (255,0,0), 2)
            if current_box is not None:
                cv2.rectangle(display, current_box[0], current_box[1], (255,0,0), 1)

        # show mask previews
        if mask_preds is not None:
            mask_imgs = []
            for i in range(4):
                m_uint8 = make_hires_mask_uint8(mask_preds[:, i], frame.shape[:2])
                m_bgr = cv2.cvtColor(m_uint8, cv2.COLOR_GRAY2BGR)
                if i == best_idx:
                    cv2.rectangle(m_bgr, (0,0), (m_bgr.shape[1]-1, m_bgr.shape[0]-1), (0,255,0), 2)
                mask_imgs.append(cv2.resize(m_bgr, (frame_w//4, frame_h//4)))
            masks_panel = np.hstack(mask_imgs)
            cv2.imshow("Masks", masks_panel)

        cv2.imshow("Video", display)
        key = cv2.waitKey(1) & 0xFF
        if key in {27, ord("q")}:  # quit
            break
        elif key == ord("1"):
            mode = "fg"
        elif key == ord("2"):
            mode = "bg"
        elif key == ord("3"):
            mode = "box"
        elif key == ord("c"):
            fg_points, bg_points, boxes = [], [], []
            mask_preds = None
            iou_preds = None
        elif key == ord("s"):
            if fg_points or bg_points or boxes:
                enc_img, _, _ = sammodel.encode_image(frame, **imgenc_cfg)
                encoded_prompts = sammodel.encode_prompts(boxes, fg_points, bg_points)
                mask_preds, iou_preds = sammodel.generate_masks(enc_img, encoded_prompts, blank_promptless_output=True)
                best_idx = sammodel.get_best_mask_index(iou_preds)
                best_mask, mem_enc, obj_ptr = sammodel.initialize_video_masking(
                    enc_img, boxes, fg_points, bg_points, mask_index_select=best_idx
                )
                samurai = SimpleSamurai(best_mask, video_framerate=fps, smoothness=smooth_val/100)
                memory = SAM2VideoObjectResults.create(args.max_memories, args.max_pointers)
                memory.store_prompt_result(frame_idx, mem_enc, obj_ptr)
                _, samurai_score, _ = samurai.get_best_decoder_results(mask_preds, iou_preds)
                tracking = False
        elif key == ord("t"):
            if samurai is not None:
                tracking = not tracking
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
