#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# ---------------------------------------------------------------------------------------------------------------------
# %% Imports

# This is a hack to make this script work from outside the root project folder (without requiring install)
try:
    import lib  # NOQA
except ModuleNotFoundError:
    import os
    import sys

    parent_folder = os.path.dirname(os.path.dirname(__file__))
    if "lib" in os.listdir(parent_folder):
        sys.path.insert(0, parent_folder)
    else:
        raise ImportError("Can't find path to lib folder!")

import argparse
import os.path as osp
from time import perf_counter

import torch
import cv2

from lib.make_sam import make_sam_from_state_dict
from lib.v2_sam.sam_v2_model import SAMV2Model

from lib.demo_helpers.ui.window import DisplayWindow, KEY
from lib.demo_helpers.ui.layout import HStack, VStack
from lib.demo_helpers.ui.buttons import ToggleButton, RadioConstraint
from lib.demo_helpers.ui.sliders import HSlider, HMultiSlider
from lib.demo_helpers.ui.static import StaticMessageBar
from lib.demo_helpers.ui.colormaps import HColormapsBar
from lib.demo_helpers.shared_ui_layout import PromptUIControl, PromptUI, ReusableBaseImage

from lib.demo_helpers.video_frame_select_ui import run_video_frame_select_ui
from lib.demo_helpers.contours import get_contours_from_mask

from lib.demo_helpers.history_keeper import HistoryKeeper
from lib.demo_helpers.loading import ask_for_path_if_missing, ask_for_model_path_if_missing, load_init_prompts
from lib.demo_helpers.misc import get_default_device_string, make_device_config, normalize_to_npuint8


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up script args

# Set argparse defaults
default_device = get_default_device_string()
default_image_path = None
default_model_path = None
default_prompts_path = None
default_display_size = 900
default_base_size = 1024

# Define script arguments
parser = argparse.ArgumentParser(description="Visualizes SAM mask data while allowing for altered window sizing")
parser.add_argument("-i", "--image_path", default=default_image_path, help="Path to input image")
parser.add_argument("-m", "--model_path", default=default_model_path, type=str, help="Path to SAM model weights")
parser.add_argument(
    "-p",
    "--prompts_path",
    default=default_prompts_path,
    type=str,
    help="Path to a json file containing initial prompts to use on start-up (see saved json results for formatting)",
)
parser.add_argument(
    "-s",
    "--display_size",
    default=default_display_size,
    type=int,
    help=f"Controls size of displayed results (default: {default_display_size})",
)
parser.add_argument(
    "-d",
    "--device",
    default=default_device,
    type=str,
    help=f"Device to use when running model, such as 'cpu' (default: {default_device})",
)
parser.add_argument(
    "-f32",
    "--use_float32",
    default=False,
    action="store_true",
    help="Use 32-bit floating point model weights. Note: this doubles VRAM usage",
)
parser.add_argument(
    "-ar",
    "--use_aspect_ratio",
    default=False,
    action="store_true",
    help="Process the image at it's original aspect ratio",
)
parser.add_argument(
    "-b",
    "--base_size_px",
    default=default_base_size,
    type=int,
    help=f"Override base model size (default {default_base_size})",
)
parser.add_argument(
    "-nq",
    "--no_quality_estimate",
    default=False,
    action="store_true",
    help="Hide mask quality estimates",
)
parser.add_argument(
    "--hide_info",
    default=False,
    action="store_true",
    help="Hide text info elements from UI",
)

# For convenience
args = parser.parse_args()
arg_image_path = args.image_path
arg_model_path = args.model_path
init_prompts_path = args.prompts_path
display_size_px = args.display_size
device_str = args.device
use_float32 = args.use_float32
use_square_sizing = not args.use_aspect_ratio
imgenc_base_size = args.base_size_px
show_iou_preds = not args.no_quality_estimate
show_info = not args.hide_info

# Set up device config
device_config_dict = make_device_config(device_str, use_float32)

# Create history to re-use selected inputs
root_path = osp.dirname(osp.dirname(__file__))
history = HistoryKeeper(root_path)
_, history_imgpath = history.read("image_path")
_, history_modelpath = history.read("model_path")

# Get pathing to resources, if not provided already
image_path = ask_for_path_if_missing(arg_image_path, "image", history_imgpath)
model_path = ask_for_model_path_if_missing(root_path, arg_model_path, history_modelpath)

# Store history for use on reload
history.store(image_path=image_path, model_path=model_path)


# ---------------------------------------------------------------------------------------------------------------------
# %% Load resources

# Get the model name, for reporting
model_name = osp.basename(model_path)

print("", "Loading model weights...", f"  @ {model_path}", sep="\n", flush=True)
model_config_dict, sammodel = make_sam_from_state_dict(model_path)
sammodel.to(**device_config_dict)
is_v2_model = isinstance(sammodel, SAMV2Model)

# Load image (or frame from video)
full_image_bgr = cv2.imread(image_path)
if full_image_bgr is None:
    ok_video, full_image_bgr = run_video_frame_select_ui(image_path)
    if not ok_video:
        print("", "Unable to load image!", f"  @ {image_path}", sep="\n", flush=True)
        raise FileNotFoundError(osp.basename(image_path))


# ---------------------------------------------------------------------------------------------------------------------
# %% Initial model run

# Run Model
print("", "Encoding image data...", sep="\n", flush=True)
t1 = perf_counter()
encoded_img, token_hw, preencode_img_hw = sammodel.encode_image(full_image_bgr, imgenc_base_size, use_square_sizing)
if torch.cuda.is_available():
    torch.cuda.synchronize()
t2 = perf_counter()
init_time_taken_ms = round(1000 * (t2 - t1), 1)
print(f"  -> Took {init_time_taken_ms} ms", flush=True)

# Run model without prompts as sanity check. Also gives initial result values
encoded_prompts = sammodel.encode_prompts([], [], [])
init_mask_preds, iou_preds = sammodel.generate_masks(encoded_img, encoded_prompts, blank_promptless_output=False)
mask_uint8 = normalize_to_npuint8(init_mask_preds[0, 0, :, :])

# Provide some feedback about how the model is running
model_device = device_config_dict["device"]
model_dtype = str(device_config_dict["dtype"]).split(".")[-1]
image_hw_str = f"{preencode_img_hw[0]} x {preencode_img_hw[1]}"
token_hw_str = f"{token_hw[0]} x {token_hw[1]}"
print(
    "",
    f"Config ({model_name}):",
    f"  Device: {model_device} ({model_dtype})",
    f"  Resolution HW: {image_hw_str}",
    f"  Tokens HW: {token_hw_str}",
    sep="\n",
    flush=True,
)


# ---------------------------------------------------------------------------------------------------------------------
# %% Set up the UI

# Set up shared UI elements & control logic
ui_elems = PromptUI(full_image_bgr, init_mask_preds)
uictrl = PromptUIControl(ui_elems)

# Set up slider controls
rescale_slider = HSlider("Input rescale", 1, 0, 2, step_size=0.01, marker_steps=10, bar_bg_color=(35, 55, 60))
sidelength_slider = HSlider(
    "Encoding Side Length",
    initial_value=imgenc_base_size,
    min_value=32,
    max_value=max(2 * imgenc_base_size, 1024),
    step_size=32,
    marker_steps=8,
    bar_bg_color=(35, 55, 60),
)
thresholds_slider = HMultiSlider(
    "Stability thresholds",
    initial_values=(0.0, 8.0),
    min_value=-20,
    max_value=20,
    step_size=0.25,
    marker_steps=20,
    fill_between_points=True,
).set_is_changed(False)

# Set up buttons for controlling which image gets displayed
show_orig_btn, show_pred_btn, show_binary_btn = ToggleButton.many(
    "Original", "Prediction", "Binary", text_scale=0.5, on_color=(55, 60, 80)
)
display_constraint = RadioConstraint(show_orig_btn, show_pred_btn, show_binary_btn)

# Set up message bars
device_dtype_str = f"{model_device}/{model_dtype}"
header_msgbar = StaticMessageBar(model_name, f"{token_hw_str} tokens", device_dtype_str, space_equally=True)
footer_msgbar = StaticMessageBar(
    "[tab] Cycle view",
    "[Arrow up/down] Switch masks",
    "[Right-click] Reset sliders",
    text_scale=0.35,
    space_equally=True,
)

# Create bar of colormaps for adjusting display style
cmap_bar = HColormapsBar(cv2.COLORMAP_MAGMA, cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_HOT, cv2.COLORMAP_BONE)

# Set up full display layout
disp_layout = VStack(
    header_msgbar if show_info else None,
    cmap_bar,
    ui_elems.layout,
    HStack(show_orig_btn, show_pred_btn, show_binary_btn),
    rescale_slider,
    sidelength_slider,
    thresholds_slider,
    footer_msgbar if show_info else None,
)

# Load initial prompts, if provided
have_init_prompts, init_prompts_dict = load_init_prompts(init_prompts_path)
if have_init_prompts:
    uictrl.load_initial_prompts(init_prompts_dict)


# ---------------------------------------------------------------------------------------------------------------------
# %% Main display

# Set up display
cv2.destroyAllWindows()
window = DisplayWindow("Display - q to quit", display_fps=60).attach_mouse_callbacks(disp_layout)
window.move(200, 50)

# Attach key controls
uictrl.attach_arrowkey_callbacks(window)
window.attach_keypress_callback("c", ui_elems.tools.clear.click)
window.attach_keypress_callback(KEY.TAB, display_constraint.next)

# For clarity, some additional keypress codes
KEY_ZOOM_OUT, KEY_ZOOM_IN = ord("-"), ord("=")

# Set up helper object for re-using an image scaled to match display sizing
img_bgr_cache = ReusableBaseImage(full_image_bgr)

# Set up copies of variables that need to exist before first run
init_preds_hw = init_mask_preds.shape[2:]
preview_preds = init_mask_preds
mask_preds = init_mask_preds
mask_contours_norm = None
rescaled_img_bgr = full_image_bgr.copy()

# *** Main display loop ***
try:
    while True:

        # Read prompt input data & selected mask
        need_prompt_encode, prompts = uictrl.read_prompts()

        # Read controls
        is_display_changed, _, selected_display_btn = display_constraint.read()
        is_mask_changed, mselect_idx, selected_mask_btn = ui_elems.masks_constraint.read()
        is_length_changed, max_side_length = sidelength_slider.read()
        is_rescale_changed, rescale_factor = rescale_slider.read()
        is_thresh_changed, (lower_mask_threshold, upper_mask_threshold) = thresholds_slider.read()

        # Switch to prediction view if user adjusts thresholds
        if is_thresh_changed:
            display_constraint.change_to(show_pred_btn)

        # Resize the base image used by encoder
        if is_rescale_changed:

            # Compute new input sizing
            new_h, new_w = [max(3, round(rescale_factor * side)) for side in preencode_img_hw]

            # Downscale then upscale back to original size, so we get scaling artifacts
            orig_h, orig_w = full_image_bgr.shape[0:2]
            rescaled_img_bgr = cv2.resize(full_image_bgr, dsize=(new_w, new_h))
            rescaled_img_bgr = cv2.resize(rescaled_img_bgr, dsize=(orig_w, orig_h))
            img_bgr_cache.set_new_image(rescaled_img_bgr)

        # Re-encode image if window sizing changes
        need_image_encode = is_length_changed or is_rescale_changed
        if need_image_encode:

            # Update window sizing & re-run image segmentation to get new (raw) mask outputs for display
            t1 = perf_counter()
            encoded_img, _, _ = sammodel.encode_image(rescaled_img_bgr, max_side_length, use_square_sizing)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t2 = perf_counter()

        # Update masking result
        need_mask_update = need_prompt_encode or need_image_encode
        if need_mask_update:
            encoded_prompts = sammodel.encode_prompts(*prompts)
            mask_preds, iou_preds = sammodel.generate_masks(encoded_img, encoded_prompts, blank_promptless_output=False)

            # Make scaled copy of predictions for preview when sizing changes
            # (if we don't do this, the UI will jitter due to layout sizing changes!)
            preview_preds = mask_preds
            if max_side_length != imgenc_base_size:
                preview_preds = torch.nn.functional.interpolate(mask_preds, size=init_preds_hw)

        # Update display of mask previews
        uictrl.update_mask_previews(preview_preds)
        if show_iou_preds:
            uictrl.draw_iou_predictions(iou_preds)

        # Scale selected mask to match display sizing
        display_hw = ui_elems.image.get_render_hw()
        mask_select = mask_preds[:, mselect_idx, :, :].unsqueeze(1)
        use_nearest_interpolation = selected_display_btn is show_pred_btn
        mask_scaled = torch.nn.functional.interpolate(
            mask_select,
            size=display_hw,
            mode="nearest-exact" if use_nearest_interpolation else "bilinear",
            align_corners=None if use_nearest_interpolation else False,
        ).squeeze(dim=(0, 1))

        # Handle display output based on UI state
        if selected_display_btn is show_pred_btn:

            # Apply lower/upper thresholds to raw mask data & normalize for display
            thresh_mask = torch.clamp(mask_scaled, lower_mask_threshold, upper_mask_threshold)
            thresh_mask_norm = (thresh_mask - lower_mask_threshold) / (upper_mask_threshold - lower_mask_threshold)
            thresh_mask_uint8 = (thresh_mask_norm * 255).byte().cpu().numpy()

            # Apply colormap + special knock-out of lowest/highest regions
            disp_img = cmap_bar.apply_colormap(thresh_mask_uint8)
            disp_img[thresh_mask_uint8 == 0] = (0, 0, 0)
            disp_img[thresh_mask_uint8 == 255] = (255, 255, 255)

        elif selected_display_btn is show_binary_btn:
            # Display simple binary result
            disp_img = ((mask_scaled > 0.0).byte() * 255).cpu().numpy()
            mask_contours_norm = None

        else:
            # Show original color image
            disp_img = img_bgr_cache.regenerate(display_hw)

        # Draw contours if needed
        mask_contours_norm = None
        if selected_display_btn is not show_binary_btn:
            mask_uint8 = ((mask_scaled > 0.0).byte() * 255).cpu().numpy()
            ok_contours, mask_contours_norm = get_contours_from_mask(mask_uint8, normalize=True)

        # Render final output
        ui_elems.olays.polygon.set_polygons(mask_contours_norm)
        ui_elems.image.set_image(disp_img)
        display_image = disp_layout.render(h=display_size_px)
        req_break, keypress = window.show(display_image)
        if req_break:
            break

        # Scale display size up when pressing +/- keys
        if keypress == KEY_ZOOM_IN:
            display_size_px = min(display_size_px + 50, 10000)
        if keypress == KEY_ZOOM_OUT:
            display_size_px = max(display_size_px - 50, 250)

        pass

except KeyboardInterrupt:
    print("", "Closed with Ctrl+C", sep="\n")

except Exception as err:
    raise err

finally:
    cv2.destroyAllWindows()
