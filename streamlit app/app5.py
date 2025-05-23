import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import tempfile
import numpy as np
from io import BytesIO
import fitz  # PyMuPDF
import pandas as pd
import base64
import pyarrow as pa
import pyarrow.parquet as pq

from utils.manage import ImageManager  # your custom image manager
from utils.annotation import read_txt, output_txt
from utils.__init__ import st_img_label  # custom image labeling component

# Load YOLO model
model = YOLO("best.pt")

st.set_page_config(layout="wide")
st.title("📄 Automated Weld Detection")

# Session state init
if "parts_data" not in st.session_state:
    st.session_state.parts_data = None
if "mode" not in st.session_state:
    st.session_state.mode = "overview"  # modes: overview, review
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0

# Upload PDF
uploaded_pdf = st.file_uploader("📄 Upload your PDF file", type=["pdf"])

# Helpers
def convert_pdf_to_image(uploaded_pdf, zoom=2):
    pdf_data = uploaded_pdf.read()
    doc = fitz.open(stream=pdf_data, filetype="pdf")
    page = doc[0]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")
    return np.array(img)

def image_with_border(image, caption="", border_color="black", border_width=2):
    # Convert PIL or NumPy image to base64 string
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()

    # HTML block with inline styles
    html = f"""
    <div style="border: {border_width}px solid {border_color}; padding: 1px; border-radius: 3px; margin-bottom: 2px;">
        <img src="data:image/png;base64,{img_b64}" style="width: 100%; height: 120px; object-fit: cover;" />
        <p style="text-align: center; margin: 1px 0;">{caption}</p>
    </div>
    """
    return html

def split_image_with_overlap(image_np, rows=4, cols=4, delta_x=50, delta_y=50):
    h, w, _ = image_np.shape
    cell_h, cell_w = h // rows, w // cols
    parts = []
    for i in range(rows):
        for j in range(cols):
            x1 = max(j * cell_w - delta_x, 0)
            y1 = max(i * cell_h - delta_y, 0)
            x2 = min((j + 1) * cell_w + delta_x, w)
            y2 = min((i + 1) * cell_h + delta_y, h)
            crop = image_np[y1:y2, x1:x2]
            parts.append(crop)
    return parts

def resize_image(image_pil, max_dim=672):
    width, height = image_pil.size
    ratio = min(max_dim/width, max_dim/height, 1.0)
    if ratio < 1.0:
        resized = image_pil.resize((int(width*ratio), int(height*ratio)))
    else:
        resized = image_pil
    return resized, ratio

def resize_before_annotation(img, max_width):
    w, h = img.size
    scaling_factor = min(1.0, max_width / w)
    new_w = int(w * scaling_factor)
    new_h = int(h * scaling_factor)
    resized_img = img.resize((new_w, new_h))
    return resized_img, scaling_factor

# Clear all session state manually
def clear_all_data():
    st.session_state.parts_data = None
    st.session_state.edited_parts = {}
    st.session_state.mode = "overview"
    st.session_state.current_review_idx = 0
    if "uploaded_pdf" in st.session_state:
        del st.session_state.uploaded_pdf



# Clear session data if no file uploaded
if uploaded_pdf is None:
    st.session_state.parts_data = None
    st.session_state.edited_parts = {}
    st.session_state.mode = "overview"
    st.session_state.current_review_idx = 0


# Main App Logic
if uploaded_pdf:
    st.success("✅ PDF uploaded.")

    if st.session_state.parts_data is None:
        full_np = convert_pdf_to_image(uploaded_pdf, zoom=2)
        parts = split_image_with_overlap(full_np)

        parts_data = []
        for idx, part_np in enumerate(parts):
            part_pil = Image.fromarray(part_np)

            resized_img, resize_ratio = resize_image(part_pil)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                resized_img.save(tmp.name)
                tmp_path = tmp.name

            # YOLO Detection
            results = model(tmp_path, conf=0.1, iou=0.2, imgsz=640)
            boxes = results[0].boxes

            rects = []
            for box in boxes:
                cls_id = int(box.cls[0])
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                rects.append({
                    "left": x1,
                    "top": y1,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "label": cls_id
                })

            parts_data.append({
                "original_image": part_pil,
                "resized_image": resized_img,
                "resize_ratio": resize_ratio,
                "rects": rects
            })

        st.session_state.parts_data = parts_data

    # Shortcut
    parts_data = st.session_state.parts_data

    # ----- Overview mode -----
    if st.session_state.mode == "overview":
        with st.expander("Click to see PDF Splits", expanded=False):
            st.header("🖼️ PDF Split Overview")
            cols = st.columns(4)
            for idx, part_info in enumerate(parts_data):
                with cols[idx % 4]:
                    html = image_with_border(part_info["resized_image"], caption=f"Part {idx + 1}", border_color="gray")
                    st.markdown(html, unsafe_allow_html=True)
                    #st.image(part_info["resized_image"], caption=f"Part {idx + 1}", use_container_width=True)

        st.markdown("---")
        if st.button("✅ Start Reviewing Parts"):
            st.session_state.mode = "review"
            st.session_state.current_idx = 0
            st.rerun()

    # ----- Review mode -----
    elif st.session_state.mode == "review":
        idx = st.session_state.current_idx
        part_info = parts_data[idx]

        st.header(f"🔎 Reviewing Part {idx + 1}/16")
        #st.image(part_info["resized_image"], caption="Image to Review", use_container_width=True)


        st.subheader("📝 Edit Bounding Boxes")

        # Resize before showing annotation
        smaller_img, scaling_factor = resize_before_annotation(part_info["resized_image"], max_width=900)

        # Resize rects to match resized image
        resized_rects_for_annotation = []
        for rect in part_info["rects"]:
            resized_rect = {
                "left": rect["left"] * scaling_factor,
                "top": rect["top"] * scaling_factor,
                "width": rect["width"] * scaling_factor,
                "height": rect["height"] * scaling_factor,
                "label": rect["label"]
            }
            resized_rects_for_annotation.append(resized_rect)

        # Check size before passing
        # w, h = part_info["resized_image"].size
        # st.write(f"🖼️ Current resized image size: {w} x {h} pixels")

        # w, h = smaller_img.size
        # st.write(f"🖼️ Current resized image for display: {w} x {h} pixels")

        

        # Editable bounding boxes
        edited_resized_rects = st_img_label(
            smaller_img,
            rects=resized_rects_for_annotation,
            box_color="red",
            key=f"editor_{idx}"
        )

        # 4. Reverse resize the edited rects before saving
        edited_rects_final = []
        for rect in edited_resized_rects:
            edited_rects_final.append({
                "left": rect["left"] / scaling_factor,
                "top": rect["top"] / scaling_factor,
                "width": rect["width"] / scaling_factor,
                "height": rect["height"] / scaling_factor,
                "label": rect["label"]
            })

        # 5. Save the final corrected rects
        st.session_state.parts_data[idx]["rects"] = edited_rects_final

        edited_rects = edited_resized_rects

        

        # --- New Section for Relabeling Boxes ---

        if edited_rects:
            st.markdown("---")
            with st.expander("Click to edit Weld labels", expanded=False):
                st.subheader("🏷️ Re-label Detected Welds")

                for i, rect in enumerate(edited_rects):
                    col1, col2 = st.columns([1, 2.5])

                    with col1:
                        # Show cropped preview
                        left = int(rect["left"])
                        top = int(rect["top"])
                        width = int(rect["width"])
                        height = int(rect["height"])

                        crop = part_info["resized_image"].crop((left, top, left + width, top + height))
                        #st.write(crop.size)

                        min_preview_width = 90
                        w, h = crop.size
                        if w < min_preview_width:
                            scaling_factor = min_preview_width / w
                        else:
                            scaling_factor = 1.0  # Keep original if already big enough
                        #scaling_factor = min(1.0, max_preview_width / w)
                        preview_resized = crop.resize((int(w * scaling_factor), int(h * scaling_factor)))

                        st.image(preview_resized, use_container_width=False)
                        # 🖼️ Add this to display size info
                        #st.caption(f"Size: {preview_resized.size[0]} × {preview_resized.size[1]} pixels")

                    with col2:
                        # Determine current label string
                        current_label_str = ""
                        if isinstance(rect["label"], int):
                            if rect["label"] in model.names:
                                current_label_str = model.names[rect["label"]]
                        elif isinstance(rect["label"], str):
                            current_label_str = rect["label"]

                        use_custom = st.checkbox(f"✏️ Custom Label for Box #{i+1}", key=f"use_custom_{idx}_{i}")

                        if use_custom:
                            custom_label = st.text_input(f"Custom label for Box #{i+1}", value=current_label_str, key=f"custom_label_{idx}_{i}")
                            rect["label"] = custom_label  # Save string label
                        else:
                            select_label = st.selectbox(f"Select Label for Box #{i+1}", list(model.names.values()), index=list(model.names.values()).index(current_label_str) if current_label_str in model.names.values() else 0, key=f"select_label_{idx}_{i}")
                            rect["label"] = list(model.names.values()).index(select_label)  # Save index label

        # Always sync updated rects
        st.session_state.parts_data[idx]["rects"] = edited_rects


        # Navigation Buttons
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            if st.button("⬅️ Previous Part"):
                if idx > 0:
                    st.session_state.current_idx -= 1
                    st.rerun()

        with col2:
            if st.button("➡️ Next Part"):
                if idx < len(parts_data) - 1:
                    st.session_state.current_idx += 1
                    st.rerun()

        with col3:
            if st.button("⬅️ Back to Overview"):
                st.session_state.mode = "overview"
                st.rerun()


    # 🎯 NEW! Add Jump to Specific Part
        #st.markdown("---")
        jump_idx = st.selectbox(
            "🔢 Jump to Specific Part",
            options=list(range(1, len(parts_data)+1)),
            format_func=lambda x: f"Part {x}",
            key="jump_to_part"
        )

        if st.button("🚀 Go to Selected Part"):
            st.session_state.current_idx = jump_idx - 1  # because idx is 0-indexed
            st.rerun()

    st.markdown("---")

    # --- Build Live Detection DataFrame ---




if st.session_state.parts_data is not None:
    all_detections = []

    for part_idx, part_info in enumerate(st.session_state.parts_data):
        rects = part_info["rects"]
        for rect in rects:
            label_value = rect.get("label", "")
            if isinstance(label_value, int) and label_value in model.names:
                label_str = model.names[label_value]
            elif isinstance(label_value, str):
                label_str = label_value
            else:
                label_str = "Unknown"

            all_detections.append({
                "Part": part_idx + 1,
                # "X": int(rect["left"]),
                # "Y": int(rect["top"]),
                # "Width": int(rect["width"]),
                # "Height": int(rect["height"]),
                "Label": label_str
            })

    # ✅ Always freshly create the DataFrame
    st.session_state.detection_df = pd.DataFrame(all_detections)


    st.sidebar.header("📋 Live Detections Summary")
    # Now use Streamlit dynamic editor!
    # st.session_state.detection_df = st.sidebar.data_editor(
    #     st.session_state.detection_df,
    #     use_container_width=True,
    #     num_rows="dynamic",
    #     key="live_detection_editor"
    # )

    if len(st.session_state.detection_df) == 0:
        st.sidebar.write("No Welds Detected in this Drawing!") 

    # 👇 After sidebar data editor
    #st.sidebar.markdown("---")
    st.sidebar.subheader(f"🔢 Total Welds Detected: **{len(st.session_state.detection_df)}**")

else:
    st.sidebar.info("📄 Upload a PDF to see detections.")

# --- Build Cropped Previews (live) ---


if st.session_state.parts_data is not None:
    all_crops = []

    for part_idx, part_info in enumerate(st.session_state.parts_data):
        img = part_info["resized_image"]
        rects = part_info["rects"]

        for rect in rects:
            # Get coordinates
            left = int(rect["left"])
            top = int(rect["top"])
            width = int(rect["width"])
            height = int(rect["height"])

            # Crop the region
            crop = img.crop((left, top, left + width, top + height))

            # Resize preview if too small
            min_preview_width = 90
            w, h = crop.size
            if w < min_preview_width:
                scaling_factor = min_preview_width / w
            else:
                scaling_factor = 1.0
            preview_resized = crop.resize((int(w * scaling_factor), int(h * scaling_factor)))

            # 🛠️ Get the real label string
            label_value = rect.get("label", "")
            if isinstance(label_value, int) and label_value in model.names:
                label_str = model.names[label_value]
            elif isinstance(label_value, str):
                label_str = label_value
            else:
                label_str = "Unknown"

            # Save
            all_crops.append({
                "part_idx": part_idx + 1,
                "preview_image": preview_resized,
                "label": label_str
            })

    # Save to session state
    st.session_state.cropped_previews = all_crops

    #st.sidebar.subheader(f"🖼️ Total Weld Crops: {len(st.session_state.cropped_previews)}")

   
    #for crop_info in st.session_state.cropped_previews:
        #st.sidebar.image(crop_info["preview_image"], caption=f"Part {crop_info['part_idx']} - {crop_info['label']}", use_container_width=False)



    preview_data = []

    for crop_info in st.session_state.cropped_previews:
        buffered = BytesIO()
        crop_info["preview_image"].save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()

        preview_data.append({
            "Preview": f"data:image/png;base64,{img_b64}",
            "Label": crop_info["label"],
            "Part": crop_info["part_idx"]
        })

    # Create DataFrame
    df_preview = pd.DataFrame(preview_data)

    # Display in Sidebar
    #st.sidebar.subheader("🖼️ Weld Crops Table")

    st.sidebar.data_editor(
        df_preview,
        column_config={
            "Preview": st.column_config.ImageColumn("Preview", width="small")
        },
        use_container_width=False,
        hide_index=True
    )



st.sidebar.button("🧹 Clear All Data", on_click=clear_all_data)

def convert_image_to_base64(pil_img):
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    return img_base64

if st.session_state.parts_data is not None and "cropped_previews" in st.session_state:
    # Step 1: Prepare Data
    download_rows = []
    for crop_info in st.session_state.cropped_previews:
        img_base64 = convert_image_to_base64(crop_info["preview_image"])
        download_rows.append({
            "Image_Base64": img_base64,
            "Label": crop_info["label"],
            
        })
    
    # Step 2: Create DataFrame
    df_download = pd.DataFrame(download_rows)

    # Step 3: Convert to CSV
    csv = df_download.to_csv(index=False)

    # Step 4: Offer Download
    st.sidebar.download_button(
        label="📥 Download Cropped Welds CSV",
        data=csv,
        file_name="cropped_welds.csv",
        mime="text/csv"
    )

    # Step 3: Convert to Parquet
    table = pa.Table.from_pandas(df_download)
    parquet_buffer = BytesIO()
    pq.write_table(table, parquet_buffer)

    # Step 4: Offer Parquet download
    st.sidebar.download_button(
        label="📥 Download Cropped Welds Parquet",
        data=parquet_buffer.getvalue(),
        file_name="cropped_welds.parquet",
        mime="application/octet-stream"
    )
