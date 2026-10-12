# 📄 PDF Document Classification Using OCR and Machine Learning

## 📌 Project Overview
This project is a Python-based web application that extracts text from PDF documents and predicts their document category using Optical Character Recognition (OCR) and Machine Learning.

## ✨ Features
- Upload PDF documents through a web interface.
- Extract selectable text from PDFs.
- Use OCR to read text from scanned PDF pages.
- Predict the document category using a trained machine learning model.
- Display extracted text and the predicted category.

## 🛠️ Technologies Used
- Python
- Streamlit
- PyMuPDF
- Tesseract OCR
- Pytesseract
- Scikit-learn
- TF-IDF
- Logistic Regression

## 📂 Project Structure
- `app.py` — Streamlit web application
- `extract_text.py` — OCR text extraction
- `train_model.py` — Model training
- `models/` — Saved machine learning model
- `dataset/` — Dataset used for training
- `requirements.txt` — Python dependencies

## ⚙️ Installation and Setup

### 1. Install Python
Install Python on your computer.

### 2. Install Tesseract OCR
Install Tesseract OCR and ensure its executable path is configured correctly in the Python code.

### 3. Install Dependencies
Run this command in the project terminal:

```bash
pip install -r requirements.txt
```

### 4. Run the Application
Run:

```bash
python -m streamlit run app.py
```

Then open the local URL shown in the terminal.

## ⚠️ Limitations
This is a prototype. Training labels were generated using OCR keyword-based rules, so the predictions and evaluation results may not be reliable for all document types. The dataset and model require further validation, especially for underrepresented categories.

## 🚀 Future Improvements
- Use a properly labelled dataset with sufficient examples for each category.
- Improve OCR accuracy for low-quality scanned documents.
- Evaluate the model on a separate, representative test dataset.
- Improve classification accuracy and handling of uncertain predictions.

## 👨‍💻 Project Purpose
Developed as a learning project to explore OCR, text processing, machine learning, and web application development.
