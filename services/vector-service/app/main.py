from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from app.services.vector_service import VectorService
from app.dependencies import get_vector_service
from dotenv import load_dotenv
from shared import ServiceLogger, create_logging_middleware
import json

# Load environment variables
load_dotenv()

logger = ServiceLogger("vector_service")


# API Models
class Document(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    query: str
    k: int = Field(
        default=3,
        gt=0,
        description="Number of results to return (must be greater than 0)",
    )
    filter_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class UpsertRequest(BaseModel):
    documents: List[Document] = Field(..., min_items=1)


class DeleteRequest(BaseModel):
    ids: List[str] = Field(..., min_items=1)


class RestoreRequest(BaseModel):
    backup_path: str = Field(..., min_length=1)


# Initialize FastAPI app
app = FastAPI(
    title="Vector Database Service",
    description="API for managing and querying vector embeddings with metadata filtering",
    version="1.0.0",
)


# Add logging middleware
app.middleware("http")(create_logging_middleware(logger))


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upsert", response_model=Dict[str, Any])
async def upsert_documents(
    request: UpsertRequest,
    background_tasks: BackgroundTasks,
    service: VectorService = Depends(get_vector_service),
):
    """Add or update documents in the vector store"""
    try:
        doc_ids = await service.add_documents(
            [doc.model_dump() for doc in request.documents]
        )
        background_tasks.add_task(service.create_backup)
        return {"success": True, "doc_ids": doc_ids}
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=Dict[str, List[Dict[str, Any]]])
async def query_documents(
    request: QueryRequest, service: VectorService = Depends(get_vector_service)
):
    """Query similar documents"""
    try:
        results = await service.search(
            query=request.query, limit=request.k, filters=request.filter_metadata
        )
        # Convert SearchResult objects to dictionaries
        documents = [
            {
                "content": result.content,
                "metadata": result.metadata,
                "score": result.score,
            }
            for result in results
        ]
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents", response_model=Dict[str, Any])
async def delete_documents(
    request: DeleteRequest, service: VectorService = Depends(get_vector_service)
):
    """Delete documents by their IDs"""
    try:
        success = await service.delete_documents(request.ids)
        if not success:
            raise HTTPException(status_code=404, detail="No documents were deleted")
        return {"success": True, "deleted_ids": request.ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/backup", response_model=Dict[str, Any])
async def create_backup(service: VectorService = Depends(get_vector_service)):
    """Manually trigger a backup"""
    try:
        backup_path = await service.create_backup()
        return {"success": True, "backup_path": backup_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/restore", response_model=Dict[str, Any])
async def restore_from_backup(
    request: RestoreRequest, service: VectorService = Depends(get_vector_service)
):
    """Restore the vector store from a backup"""
    try:
        await service.restore_from_backup(request.backup_path)
        return {"success": True, "message": "Restore completed successfully"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=Dict[str, Any])
async def health_check(service: VectorService = Depends(get_vector_service)):
    """Health check endpoint"""
    return {"status": "healthy", "vector_store_initialized": service.is_healthy()}
