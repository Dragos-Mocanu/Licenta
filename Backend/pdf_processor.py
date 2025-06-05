import fitz

class PDFProcessor:
    @staticmethod
    def extract_text(raw: bytes) -> str:
        pages = []
        with fitz.open(stream=raw, filetype="pdf") as doc:
            for page in doc:
                pages.append(page.get_text())
        return "".join(pages)
