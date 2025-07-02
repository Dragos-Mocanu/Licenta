import fitz

class PDFProcessor:
    @staticmethod
    def extract_text(raw: bytes) -> str:
        # Extracts and returns plain text from a PDF given as raw bytes.
        pages = []

        # Open the PDF from the byte stream
        with fitz.open(stream=raw, filetype="pdf") as doc:
            # Iterate through all pages and extract text
            for page in doc:
                pages.append(page.get_text())

        # Join text from all pages into a single string
        return "".join(pages)