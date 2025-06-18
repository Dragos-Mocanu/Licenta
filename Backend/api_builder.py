from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pdf_processor import PDFProcessor
from text_analyzer import TextAnalyzer

class APIBuilder:
    def __init__(self) -> None:
        self.analyzer = TextAnalyzer()
        self.app = FastAPI()
        self._cfg()
    def _cfg(self) -> None:
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self._routes()
    def _routes(self) -> None:
        @self.app.post("/api/analyze")
        async def analyze(
            text: str | None = Form(None),
            file: UploadFile | None = File(None),
        ):
            if file:
                raw = await file.read()
                if file.filename.lower().endswith(".pdf") or file.content_type == "application/pdf":
                    txt = PDFProcessor.extract_text(raw)
                else:
                    txt = raw.decode("utf-8", errors="ignore")
            elif text:
                txt = text
            else:
                return {"error": "No text, PDF or TXT provided."}
            return self.analyzer.analyze(txt)