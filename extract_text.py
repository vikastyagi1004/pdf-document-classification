
from pathlib import Path
import pandas as pd
import pytesseract
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Dataset image folder
IMAGE_FOLDER = Path(
    r"C:\Users\Lenovo\Downloads\archive\fullDataset\images"
)

# Output folder inside your project
OUTPUT_FOLDER = Path("dataset")
OUTPUT_FOLDER.mkdir(exist_ok=True)

records = []

# Supported image formats
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

if not IMAGE_FOLDER.exists():
    raise FileNotFoundError(
        f"Image folder not found: {IMAGE_FOLDER}"
    )

images = sorted(
    p for p in IMAGE_FOLDER.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
)

if not images:
    raise ValueError("No supported images found in the folder.")

# First test with 10 images
for image_path in images :
    try:
        with Image.open(image_path) as img:
            text = pytesseract.image_to_string(
                img.convert("RGB"),
                lang="eng"
            )

        records.append({
            "filename": image_path.name,
            "text": text.strip(),
            "label": "Receipt"
        })

        print(f"Processed: {image_path.name}")

    except Exception as error:
        print(f"Skipped {image_path.name}: {error}")

df = pd.DataFrame(records)
output_file = OUTPUT_FOLDER / "receipt_ocr.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("\nOCR test completed!")
print("Images found:", len(images))
print("Images processed:", len(records))
print("CSV saved at:", output_file.resolve())
print(df[["filename", "label"]].head())
