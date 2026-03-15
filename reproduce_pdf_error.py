import pdfplumber
import sys

pdf_path = r"C:\Users\iaian\Downloads\2403.14403v2.pdf"

try:
    print(f"Trying to read {pdf_path} with pdfplumber...")
    with pdfplumber.open(pdf_path) as pdf:
        first_page = pdf.pages[0]
        text = first_page.extract_text()
        print("Successfully extracted text from first page:")
        print(text[:1000] if text else "No text found")
except Exception as e:
    print(f"Error with pdfplumber: {e}")

from pypdf import PdfReader

try:
    print("\nTrying to read with pypdf...")
    reader = PdfReader(pdf_path)
    page = reader.pages[0]
    text = page.extract_text()
    print("Successfully extracted text content:")
    print(text[:1000] if text else "No text found")
except Exception as e:
    print(f"Error with pypdf: {e}")
