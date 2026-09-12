"""
app.py — Streamlit demo for the Satellite SR+Seg pipeline.

Run with:
    streamlit run app.py

Features:
  • Upload .tif, .png, or .jpg
  • 3-column view: input | SR output | segmentation map
  • Colourized segmentation
  • PSNR display if ground truth is uploaded
  • Friendly message when no checkpoint exists
"""

import os
import io
import warnings
import numpy as np
import torch

try:
    import streamlit as st
    _ST = True
except ImportError:
    raise RuntimeError("Please install streamlit: pip install streamlit")

try:
    from PIL import Image
    _PIL = True
except ImportError:
    _PIL = False
    warnings.warn("PIL not available — image display may be limited.")

import config as cfg
from model import build_model
from metrics import psnr as calc_psnr
from inference import load_image, infer_image, SEG_COLORS


# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Satellite SR + Segmentation",
    layout="wide",
    page_icon="🛰️",
)

st.title("🛰️ Satellite Image Super-Resolution & Land-Cover Segmentation")
st.caption(
    "Upload a Sentinel-2 patch (.tif / .png / .jpg) to run the "
    "4× super-resolution and land-cover segmentation pipeline."
)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    ckpt_path = st.text_input("Checkpoint path", value=cfg.DEMO_CHECKPOINT)
    patch_size = st.slider("Patch size (px)", 64, 512, cfg.PATCH_SIZE, step=64)
    stride     = st.slider("Stride (px)",     32, 480, cfg.PATCH_STRIDE, step=32)
    upload_gt  = st.checkbox("Upload ground truth (for PSNR)")

# ─── Model loader ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(ckpt: str):
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        else torch.device("cpu")
    )
    model = build_model(cfg).to(device)
    if os.path.isfile(ckpt):
        state_dict = torch.load(ckpt, map_location=device)
        if "model_state" in state_dict:
            state_dict = state_dict["model_state"]
        model.load_state_dict(state_dict)
        model.eval()
        return model, device, True   # (model, device, checkpoint_loaded)
    else:
        model.eval()
        return model, device, False


# ─── Check for checkpoint ─────────────────────────────────────────────────────
model, device, ckpt_loaded = load_model(ckpt_path)
if not ckpt_loaded:
    st.warning(
        f"⚠️ No trained checkpoint found at `{ckpt_path}`.  "
        "The model will run with **random weights** — outputs will be meaningless.  \n"
        "Train the model first with:  \n"
        "```\npython train.py\n```"
    )


# ─── Colorize segmentation ────────────────────────────────────────────────────
def colorize_seg(seg: np.ndarray) -> np.ndarray:
    """(H, W) int → (H, W, 3) uint8 RGB."""
    rgb = SEG_COLORS[seg.clip(0, len(SEG_COLORS) - 1)]
    return rgb


# ─── To PIL ──────────────────────────────────────────────────────────────────
def to_pil_rgb(arr: np.ndarray) -> "Image":
    """Convert (C, H, W) float [0,1] → PIL RGB image."""
    vis = (arr[:3].clip(0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
    return Image.fromarray(vis)


# ─── Main upload area ────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload image", type=["tif", "tiff", "png", "jpg", "jpeg"]
)

gt_uploaded = None
if upload_gt:
    gt_uploaded = st.file_uploader(
        "Upload ground-truth (HR) image", type=["tif", "tiff", "png", "jpg", "jpeg"],
        key="gt_upload",
    )

if uploaded is not None:
    # Save to temp file so rasterio / PIL can read it
    import tempfile
    suffix = os.path.splitext(uploaded.name)[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.spinner("Running model …"):
        try:
            image = load_image(tmp_path, num_bands=cfg.NUM_BANDS)
            sr_image, seg_map = infer_image(
                model, image, device,
                patch_size=patch_size,
                stride=stride,
                scale=cfg.SCALE_FACTOR,
                num_classes=cfg.NUM_CLASSES,
                num_bands=cfg.NUM_BANDS,
            )
        except Exception as e:
            st.error(f"Inference failed: {e}")
            st.stop()

    # ── Display columns ───────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📥 Input (LR)")
        if _PIL:
            vis_lr = to_pil_rgb(image)
            # Resize for display if very large
            st.image(vis_lr, caption=f"Input {image.shape[1]}×{image.shape[2]}px",
                     use_container_width=True)
        else:
            st.warning("PIL not available for display.")

    with col2:
        st.subheader("🔍 SR Output (4×)")
        if _PIL:
            vis_sr = to_pil_rgb(sr_image)
            st.image(vis_sr, caption=f"SR {sr_image.shape[1]}×{sr_image.shape[2]}px",
                     use_container_width=True)

            # PSNR against GT if provided
            if gt_uploaded is not None:
                suffix_gt = os.path.splitext(gt_uploaded.name)[1]
                with tempfile.NamedTemporaryFile(suffix=suffix_gt, delete=False) as tmp_gt:
                    tmp_gt.write(gt_uploaded.read())
                    tmp_gt_path = tmp_gt.name
                gt_image = load_image(tmp_gt_path, num_bands=cfg.NUM_BANDS)
                import torch
                sr_t  = torch.from_numpy(sr_image).unsqueeze(0)
                gt_t  = torch.from_numpy(gt_image).unsqueeze(0)
                # Resize GT to match SR if needed
                if sr_t.shape != gt_t.shape:
                    gt_t = torch.nn.functional.interpolate(
                        gt_t, size=sr_t.shape[-2:], mode="bilinear", align_corners=False
                    )
                psnr_val = calc_psnr(sr_t, gt_t)
                st.metric("PSNR vs GT", f"{psnr_val:.2f} dB")

    with col3:
        st.subheader("🗺️ Segmentation Map")
        seg_rgb = colorize_seg(seg_map)
        if _PIL:
            st.image(Image.fromarray(seg_rgb),
                     caption=f"Segmentation {seg_map.shape[0]}×{seg_map.shape[1]}px",
                     use_container_width=True)

        # Legend
        st.markdown("**Legend:**")
        color_names = list(zip(cfg.CLASS_NAMES, SEG_COLORS))
        legend_html = " ".join(
            f'<span style="background:rgb({r},{g},{b});'
            f'padding:2px 8px;margin:2px;border-radius:4px;color:{"black" if r+g+b>400 else "white"}">'
            f'{name}</span>'
            for name, (r, g, b) in color_names
        )
        st.markdown(legend_html, unsafe_allow_html=True)

    # Download buttons
    st.divider()
    st.subheader("💾 Download Results")
    if _PIL:
        # SR image as PNG
        sr_buf = io.BytesIO()
        to_pil_rgb(sr_image).save(sr_buf, format="PNG")
        st.download_button("Download SR image (PNG)", sr_buf.getvalue(),
                           file_name="sr_output.png", mime="image/png")
        # Seg map as PNG
        seg_buf = io.BytesIO()
        Image.fromarray(seg_rgb).save(seg_buf, format="PNG")
        st.download_button("Download Seg map (PNG)", seg_buf.getvalue(),
                           file_name="seg_map.png", mime="image/png")

    # Clean up temp files
    try:
        os.unlink(tmp_path)
    except Exception:
        pass
else:
    st.info("👆 Upload an image to get started.")

# ─── Sidebar class info ───────────────────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.subheader("Class colours")
    for name, (r, g, b) in zip(cfg.CLASS_NAMES, SEG_COLORS):
        st.markdown(
            f'<span style="background:rgb({r},{g},{b});padding:2px 10px;'
            f'border-radius:4px;color:{"black" if r+g+b>400 else "white"}">'
            f"{name}</span>",
            unsafe_allow_html=True,
        )
    st.divider()
    st.subheader("Device")
    st.caption(str(device))
