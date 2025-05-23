# YOLOv8 + Streamlit Weld Detection App

This project is a web application built with **Streamlit** and **YOLOv8** (Ultralytics) to automate the detection of weld symbols in engineering drawings. Manual inspection of weld symbols can be time-consuming and prone to errors, especially in large or complex drawings. This tool aims to improve efficiency and accuracy by providing an automated solution.

The application follows this workflow:
*   Users upload a PDF document (typically an engineering drawing).
*   The application converts the first page of the PDF to an image.
*   This image is then split into smaller, possibly overlapping, parts to handle large drawings and improve detection accuracy for smaller symbols.
*   The YOLOv8 model processes each part to detect weld symbols.
*   Users can then review these detections, edit bounding boxes, and correct or assign labels.
*   Finally, the annotated data can be exported.

## Features

**1. PDF Upload and Processing:**
*   Upload engineering drawings in PDF format.
*   Automatic conversion of the PDF's first page into a high-resolution image for detailed analysis.

**2. Intelligent Image Segmentation:**
*   Large drawings are automatically split into smaller, overlapping segments. This technique enhances the detection of small weld symbols and allows for efficient processing of high-resolution imagery.

**3. YOLOv8-Powered Weld Symbol Detection:**
*   Leverages a pre-trained YOLOv8 model (`best.pt`) to accurately identify weld symbols.

**4. Interactive Annotation and Review Interface:**
*   Review detected weld symbols on each segmented part of the drawing.
*   Visually inspect and validate the model's predictions.
*   Precisely edit bounding boxes by adjusting their size and position.
*   Relabel detected symbols by selecting from the model's predefined classes or assigning custom text labels.

**5. Comprehensive Detection Overview and Navigation:**
*   View a live, filterable summary table of all detected weld symbols across all processed image segments.
*   Easily navigate to specific parts of the drawing for targeted review.
*   Examine cropped image previews of each detected weld symbol directly in the sidebar for quick verification.

**6. Data Export for Further Use:**
*   Export the detected weld symbol data, including cropped images (Base64 encoded) and their corresponding labels, in CSV or Parquet format. This allows for easy integration with other data analysis tools or workflows.

## How to Run

Follow these steps to set up and run the application locally:

1.  **Clone the Repository:**
    ```bash
    git clone https://your-repository-url/your-project.git
    cd your-project-directory-name
    ```
    (Replace `https://your-repository-url/your-project.git` with the actual repository URL and `your-project-directory-name` with the actual directory name after cloning).

2.  **Install Dependencies:**
    The required Python packages are listed in `streamlit app/requirements.txt`. Install them using pip:
    ```bash
    pip install -r "streamlit app/requirements.txt"
    ```

3.  **Run the Application:**
    Launch the Streamlit application using the following command:
    ```bash
    streamlit run "streamlit app/app5.py"
    ```

4.  **Access the Application:**
    Once the application is running, Streamlit will typically provide a local URL in your terminal (e.g., `http://localhost:8501`). Open this URL in your web browser to use the application.

## Project Structure

The project is organized as follows:

*   `README.md`: This file, providing an overview and guide for the project.
*   `streamlit app/`: The main directory containing the Streamlit web application.
    *   `app5.py`: The core script that runs the Streamlit application, handling the main interface and workflow.
    *   `best.pt`: The pre-trained YOLOv8 model file used for detecting weld symbols.
    *   `requirements.txt`: A text file listing all Python dependencies required to run the application.
    *   `pages/`: This directory contains additional Streamlit pages that provide specific functionalities. For example:
        *   `crop.py`: (Example: A page for image cropping or displaying cropped sections).
        *   `gemma.py`: (Example: A page for integration with other models or specific data processing tasks).
    *   `utils/`: A directory for utility scripts that support the main application.
        *   `manage.py`: (Handles image management tasks, such as loading, processing, or organizing images).
        *   `annotation.py`: (Responsible for functions related to creating, reading, or writing annotation data).
        *   `frontend/`: (Contains custom Streamlit components, CSS styles, or JavaScript snippets to enhance the user interface).

## Dependencies

This project relies on several key Python libraries and technologies:

*   **Streamlit:** Used to build the interactive web application interface.
*   **Ultralytics YOLO (yolov8):** Powers the core weld symbol detection functionality.
*   **Pillow (PIL):** Essential for various image manipulation tasks, such as opening, resizing, and saving images.
*   **PyMuPDF (fitz):** Enables the extraction of images from PDF documents.
*   **NumPy:** Used for efficient numerical computations, particularly when handling image data arrays.
*   **Pandas:** Facilitates data organization and manipulation, for example, when creating summary tables of detected symbols.
*   **PyArrow:** Used for reading and writing data in the Apache Parquet format, enabling efficient data storage and retrieval.

For a complete list of all Python dependencies and their specific versions, please refer to the `streamlit app/requirements.txt` file. You can install all required packages by running:

```bash
pip install -r "streamlit app/requirements.txt"
```

## Model

The core of the weld symbol detection capability in this application is provided by a YOLOv8 model.

*   **Model Type:** The application utilizes a YOLOv8 (You Only Look Once version 8) object detection model, a state-of-the-art architecture developed by Ultralytics.
*   **Model File:** The pre-trained weights for the model are stored in a file named `best.pt`, which is expected to be located within the `streamlit app/` directory.
*   **Detection Task:** This specific YOLOv8 model is trained and configured to identify and locate various weld symbols commonly found in engineering drawings and technical diagrams.
*   **Availability:** The `best.pt` model file is included directly in the `streamlit app/` directory of this project, ensuring it is readily available when the application runs.

