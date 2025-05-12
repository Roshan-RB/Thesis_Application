import streamlit as st
from PIL import Image
import numpy as np
import io
import fitz  # PyMuPDF
from streamlit_cropperjs import st_cropperjs
import tempfile
import time
import requests
from io import BytesIO

st.set_page_config(page_title="Extract Text with Gemma", layout="wide")
st.title("📝 Extract Text from PDF Region using Ollama (Gemma 3)")

# --- Session State ---
for key, default in {
    "pdf_file": None,
    "page_number": 1,
    "image_bytes": None,
    "cropped_image": None,
    "pil_image": None,
    "tmp_file_path": None,
    "crop_button_clicked": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- Sidebar Upload ---
pdf_file = st.sidebar.file_uploader("📄 Upload a PDF file", type="pdf")
page_number = st.sidebar.number_input("Page number", min_value=1, format="%d", value=1)

# --- PDF Handling ---
if pdf_file:
    if st.session_state.pdf_file != pdf_file:
        st.session_state.crop_button_clicked = False
        st.session_state.pil_image = None
    st.session_state.pdf_file = pdf_file
    st.session_state.file_name = pdf_file.name.rsplit('.', 1)[0]
    st.write(f"Zeichnungs-Nr.: `{st.session_state.file_name}`")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(st.session_state.pdf_file.read())
        st.session_state.tmp_file_path = tmp_file.name

    try:
        doc = fitz.open(st.session_state.tmp_file_path)

        if page_number > len(doc):
            st.error(f"Invalid page number. PDF has only {len(doc)} pages.")
        else:
            st.session_state.page_number = page_number
            page = doc.load_page(page_number - 1)
            pix = page.get_pixmap()
            image_bytes = pix.tobytes()
            st.session_state.image_bytes = image_bytes

            # Show PDF page image
            st.markdown("### 🖼️ Selected Page")
            st.image(image_bytes, caption=f"Page {page_number}")

            if st.button("✂️ Select area to crop"):
                st.session_state.crop_button_clicked = True

            if st.session_state.crop_button_clicked:
                cropped_bytes = st_cropperjs(st.session_state.image_bytes, btn_text="Crop Image")

                if cropped_bytes:
                    with st.spinner("Processing cropped image..."):
                        time.sleep(1)

                    st.success("✅ Image cropped successfully!")
                    pil_image = Image.open(BytesIO(cropped_bytes)).convert("RGB")
                    st.session_state.pil_image = pil_image
                    st.image(pil_image, caption="🖼️ Cropped Region")

    except Exception as e:
        st.error(f"❌ Failed to open PDF: {e}")
        st.session_state.pdf_file = None

else:
    st.info("📂 Upload a PDF file to begin.")

# --- Gemma Extraction ---
if st.session_state.pil_image is not None:
    if st.button("🔍 Extract Text with Gemma"):
        with st.spinner("Sending image to Ollama..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                    st.session_state.pil_image.save(tmp_img.name)
                    with open(tmp_img.name, "rb") as f:
                        image_bytes = f.read()

                response = requests.post(
                    "http://localhost:11434/api/generate",  # Ollama endpoint
                    json={
                        "model": "gemma3:4b",
                        "prompt": (
                            "You are a highly accurate assistant trained to interpret technical engineering drawings. "
                            "Your task is to extract structured technical information from the **legend section** of such drawings. "
                            "You understand common terminology used in manufacturing and CAD drawings, and you are capable of interpreting both typed and handwritten text in varied layouts.\n\n"
                            "Please analyze the image of the legend section and extract the following fields if present and legible:\n"
                            "- Drawing Number\n"
                            "- Title or Part Name\n"
                            "- Revision\n"
                            "- Scale\n"
                            "- Weight\n"
                            "- Material\n"
                            "- Author or Drawn By\n"
                            "- Date Drawn\n"
                            "- Company Name\n"
                            "- Customer\n\n"
                            "If any of these fields are missing or illegible, omit them. Format your response cleanly using markdown or structured bullet points."
                        ),
                        "image": image_bytes.hex()
                    }
                )

                if response.ok:
                    st.markdown("### 📄 Extracted Text")
                    full_response = ""
                    for line in response.iter_lines():
                        if line:
                            data = line.decode("utf-8")
                            try:
                                json_data = eval(data) if data.startswith("{") else {}
                                full_response += json_data.get("response", "")
                            except Exception as e:
                                st.warning(f"Could not parse a response chunk: {e}")
                    st.markdown(full_response)
                else:
                    st.error("❌ Failed to get a response from Ollama. Make sure Ollama is running and the model supports image input.")

            except Exception as e:
                st.error(f"🚨 Error sending request to Ollama: {e}")
