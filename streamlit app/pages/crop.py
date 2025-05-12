import streamlit as st
from PIL import Image
import numpy as np
import io
import fitz  # PyMuPDF
import cv2
from ultralytics import YOLO
from streamlit_cropperjs import st_cropperjs
import tempfile
import time
from io import BytesIO
import pandas as pd
from PIL import ImageDraw

st.set_page_config(page_title="Crop Mode", layout="wide")
st.title("✂️ Crop Mode: Detect Objects in Custom Region")

# Load YOLO model
model = YOLO(r"C:\Thesis\Dataprep\YOLO_classification\runs\detect\train\weights\best.pt")

# --- Session State Init ---
for key, default in {
    "pdf_file": None,
    "page_number": 1,
    "image_with_boxes": None,
    "bounds": None,
    "cropped_image": None,
    "Processed_image": None,
    "crop_button_clicked": False,
    "pil_image": None,
    "tmp_file_path": None,
    "pdf_document": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

if "detections_df" not in st.session_state:
    st.session_state.detections_df = pd.DataFrame(columns=["Page", "Class", "Confidence", "X1", "Y1", "X2", "Y2"])




# --- Sidebar Upload & Page ---
pdf_file = st.sidebar.file_uploader("📄 Upload a PDF file", type="pdf")
page_number = st.sidebar.number_input("Enter page number", min_value=1, format="%d", value=1)

# --- Process Upload ---
if pdf_file:
    if st.session_state.pdf_file != pdf_file:
        st.session_state.crop_button_clicked = False
        st.session_state.pil_image = None
    st.session_state.pdf_file = pdf_file
    st.session_state.file_name = pdf_file.name.rsplit('.', 1)[0]
    st.write(f"Zeichnungs-Nr.: `{st.session_state.file_name}`")

    # Save uploaded PDF to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(st.session_state.pdf_file.read())
        st.session_state.tmp_file_path = tmp_file.name

    try:
        doc = fitz.open(st.session_state.tmp_file_path)
        st.session_state.pdf_document = doc

        if page_number > len(doc):
            st.error(f"Invalid page number. PDF has only {len(doc)} pages.")
        else:
            st.session_state.page_number = page_number
            page = doc.load_page(page_number - 1)
            pix = page.get_pixmap()
            image_bytes = pix.tobytes()
            st.session_state.image_bytes = image_bytes

            # Show selected page
            st.markdown("### 🖼️ Selected Page")
            st.image(image_bytes, caption=f"Page {page_number}")

            # Trigger cropper
            if st.button("✂️ Select area to crop"):
                st.session_state.crop_button_clicked = True

            if st.session_state.crop_button_clicked:
                cropped_image = st_cropperjs(st.session_state.image_bytes, btn_text="Crop Image")

                if cropped_image:
                    st.session_state.cropped_image = cropped_image

                    with st.spinner("Loading the cropped image..."):
                        time.sleep(1)

                    st.success("✅ Image cropped successfully!")

                    try:
                        # Convert to PIL
                        pil_image = Image.open(BytesIO(cropped_image)).convert("RGB")
                        st.session_state.pil_image = pil_image

                        st.markdown("### ✂️ Cropped Image")
                        st.image(pil_image)

                        # Save temp if needed
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img_file:
                            pil_image.save(tmp_img_file.name)
                            with open(tmp_img_file.name, "rb") as img_file:
                                _ = img_file.read()  # you can store it or use for download

                    except Exception as e:
                        st.error(f"❌ Error loading cropped image: {e}")

    except Exception as e:
        st.error(f"❌ Failed to open PDF: {e}")
        st.session_state.pdf_file = None

else:
    st.info("📂 Upload a PDF file to begin.")

# --- YOLO Detection ---
if st.session_state.pil_image is not None:

    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.1, 0.01)
    imgsz = st.number_input("Image Size", 320, 1280, 640, step=32)
    #if st.button("🔍 Detect Objects in Cropped Region"):
    st.subheader("📸 Detection Results")

    

    crop_np = np.array(st.session_state.pil_image)
    img_bgr = cv2.cvtColor(crop_np, cv2.COLOR_RGB2BGR)

    result = model(img_bgr, conf=conf_threshold, iou=0.2, imgsz=imgsz)
    names = model.names
    boxes = result[0].boxes
    output_img = img_bgr.copy()

    if len(boxes) == 0:
        st.info("No objects detected.")
    else:
        results = []

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = names[cls_id]
            xyxy = box.xyxy[0].cpu().numpy().astype(int)

            # Draw box
            color = (0, 0, 255)
            cv2.rectangle(output_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
            label = f"{name}"
            cv2.putText(output_img, label, (xyxy[0]+5, xyxy[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Save to results
            results.append({
                "Page": st.session_state.page_number,
                "Class": name,
                "Confidence": round(conf, 2),
                "X1": xyxy[0],
                "Y1": xyxy[1],
                "X2": xyxy[2],
                "Y2": xyxy[3]
            })

        # Append to session-wide DataFrame
        if results:
            new_df = pd.DataFrame(results)
            st.session_state.detections_df = pd.concat([st.session_state.detections_df, new_df], ignore_index=True)

        # Convert and display
        output_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        st.image(output_rgb)

        if not st.session_state.detections_df.empty:
            st.markdown("### 📋 Cumulative Detection Table")
            st.dataframe(st.session_state.detections_df)

            csv = st.session_state.detections_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", csv, "detections.csv", "text/csv")

    if st.button("➕ Start New Crop"):
        st.session_state.crop_button_clicked = False
        st.session_state.pil_image = None
        st.rerun()

