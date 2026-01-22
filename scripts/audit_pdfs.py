from pathlib import Path
import csv
import pdfplumber
import shutil

PDF_DIR = Path(r"C:\Users\ssar\Desktop\RAG_notes\project\data_raw")
BAD_DIR = Path(r"C:\Users\ssar\Desktop\RAG_notes\project\data_raw_bad")
REPORT = Path(r"C:\Users\ssar\Desktop\RAG_notes\project\eval\pdf_audit.csv")

PAGES_TO_CHECK = 10

MIN_CHARS_THRESHOLD = 200

def extract_chars_quick(pdf_path: Path) -> int:
    total = 0
    with pdfplumber.open(pdf_path) as pdf:
        n = min(PAGES_TO_CHECK, len(pdf.pages))
        for i in range(n):
            txt = pdf.pages[i].extract_text() or ""
            total += len(txt)
    return total

def main():
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    BAD_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    print("PDFs found:", len(pdfs))

    rows = []
    bad = []

    for p in pdfs:
        try:
            chars = extract_chars_quick(p)
        except Exception as e:
            chars = -1
            rows.append({"pdf": p.name, "chars_checked": chars, "status": "ERROR", "error": str(e)})
            bad.append(p)
            continue

        status = "OK" if chars >= MIN_CHARS_THRESHOLD else "NEEDS_OCR"
        rows.append({"pdf": p.name, "chars_checked": chars, "status": status, "error": ""})

        if status != "OK":
            bad.append(p)

    # write report
    with open(REPORT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pdf", "chars_checked", "status", "error"])
        w.writeheader()
        w.writerows(rows)

    print(f"✅ Wrote report: {REPORT}")
    print(f"Bad PDFs: {len(bad)}")

    # move bad pdfs to BAD_DIR (safer than delete)
    for p in bad:
        dest = BAD_DIR / p.name
        if not dest.exists():
            shutil.move(str(p), str(dest))

    print(f"✅ Moved {len(bad)} PDFs to: {BAD_DIR}")

if __name__ == "__main__":
    main()
