from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
from typing import List
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from shared import ServiceLogger, create_logging_middleware


logger = ServiceLogger('document-service')

load_dotenv()

app = FastAPI(title="Document Processing Service")

# Add logging middleware
app.middleware("http")(create_logging_middleware(logger))

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VECTOR_SERVICE_URL = os.getenv("VECTOR_SERVICE_URL", "http://vector-service:8004")

@app.post("/process")
async def process_documents(files: List[UploadFile] = File(...)):
    try:
        all_texts = []
        
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                content = await file.read()
                tmp.write(content)
                tmp.flush()
                
                loader = PyPDFLoader(tmp.name)
                documents = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                texts = text_splitter.split_documents(documents)
                all_texts.extend(texts)
                
            os.unlink(tmp.name)
        
        # Send to vector service
        documents = [
            {
                "content": text.page_content,
                "metadata": text.metadata
            }
            for text in all_texts
        ]
        
        response = requests.post(
            f"{VECTOR_SERVICE_URL}/upsert",
            json={"documents": documents}
        )
        
        if response.status_code == 200:
            return {"success": True, "message": "Documents processed successfully"}
        else:
            return {"success": False, "error": "Failed to store vectors"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 
