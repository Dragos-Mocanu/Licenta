from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pdf_processor import PDFProcessor
from text_analyzer import TextAnalyzer

# Class responsible for setting up the FastAPI application and defining routes
class APIBuilder:
    def __init__(self) -> None:
        # Initialize the text analyzer instance
        self.analyzer = TextAnalyzer()
        # Create FastAPI app
        self.app = FastAPI()
        # Configure middleware and routes
        self._cfg()

    # Configure CORS and register routes
    def _cfg(self) -> None:
        # Enable CORS for all origins, headers, and methods (for frontend communication)
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        # Set up API routes
        self._routes()

    # Define application routes
    def _routes(self) -> None:
        # Define a POST endpoint to analyze either raw text or a file
        @self.app.post("/api/analyze")
        async def analyze(
            text: str | None = Form(None),           # Optional text input from form
            file: UploadFile | None = File(None),    # Optional uploaded file (PDF or TXT)
        ):
            # If a file is provided
            if file:
                raw = await file.read()
                # Check if the file is a PDF
                if file.filename.lower().endswith(".pdf") or file.content_type == "application/pdf":
                    txt = PDFProcessor.extract_text(raw)  # Extract text using PDF processor
                else:
                    txt = raw.decode("utf-8", errors="ignore")  # Decode as plain text
            elif text:
                # Use direct text input
                txt = text
            else:
                # No input provided
                return {"error": "No text, PDF or TXT provided."}

            # Perform text analysis and return the results
            return self.analyzer.analyze(txt)