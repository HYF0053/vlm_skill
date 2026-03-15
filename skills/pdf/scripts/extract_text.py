import sys
import pdfplumber
import os

def extract_text(pdf_path, output_txt_path=None):
    """
    Extract text from PDF using pdfplumber with UTF-8 encoding.
    """
    text_content = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text_content.append(f"--- Page {i+1} ---\n{page_text}")
        
        full_text = "\n\n".join(text_content)
        
        if output_txt_path:
            with open(output_txt_path, "w", encoding="utf-8") as f:
                f.write(full_text)
            print(f"Successfully extracted text to {output_txt_path}")
        else:
            # Print to stdout
            print(full_text)
            
    except Exception as e:
        print(f"Error extracting text: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # Ensure stdout handles UTF-8 on Windows
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        
    if len(sys.argv) < 2:
        print("Usage: python extract_text.py <input.pdf> [output.txt]")
        sys.exit(1)
        
    pdf_input = sys.argv[1]
    txt_output = sys.argv[2] if len(sys.argv) > 2 else None
    
    extract_text(pdf_input, txt_output)
