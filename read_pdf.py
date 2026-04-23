import fitz
import sys

def main():
    pdf_path = sys.argv[1]
    out_path = sys.argv[2]
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

if __name__ == "__main__":
    main()
