
import streamlit as st
import fitz
import pytesseract
import joblib

from PIL import Image
from pathlib import Path

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# Load trained model
model = joblib.load("models/document_classifier.pkl")

st.set_page_config(
    page_title="Document Classifier",
    page_icon="📄",
    layout="centered"
)

st.title("📄 PDF Document Classification")
st.write(
    "Upload a PDF to extract its text using OCR "
    "and predict its document category."
)

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
)

if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()
    extracted_text = ""

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
            for page in pdf:
                page_text = page.get_text().strip()

                if page_text:
                    extracted_text += page_text + "\n"
                else:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    image = Image.frombytes(
                        "RGB",
                        [pix.width, pix.height],
                        pix.samples
                    )
                    extracted_text += (
                        pytesseract.image_to_string(image, lang="eng")
                        + "\n"
                    )

        if extracted_text.strip():
            st.subheader("Extracted Text")
            st.text_area(
                "OCR / PDF text",
                extracted_text,
                height=220
            )

            prediction = model.predict([extracted_text])[0]

            st.subheader("Predicted Document Type")
            st.success(prediction)

            st.warning(
                "Prototype prediction: labels were generated "
                "automatically from OCR keywords. Verify the result "
                "before using it for real documents."
            )
        else:
            st.error("No text could be extracted from this PDF.")

    except Exception as error:
        st.error(f"Could not process this PDF: {error}")
