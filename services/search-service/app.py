import os
import traceback

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from serpapi import GoogleSearch

from shared import ServiceLogger, create_logging_middleware

load_dotenv()

logger = ServiceLogger("search-service")

app = FastAPI(title="Search Service")

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add logging middleware
app.middleware("http")(create_logging_middleware(logger))


class SearchRequest(BaseModel):
    query: str


@app.post("/search")
async def search(request: SearchRequest):
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="SERPAPI_API_KEY not found in environment variables",
            )

        search = GoogleSearch({"q": request.query, "api_key": api_key, "num": 3})

        results = search.get_dict()

        # Extract and format relevant information
        formatted_results = []
        if "organic_results" in results:
            for result in results["organic_results"][:3]:
                formatted_results.append(
                    f"Title: {result.get('title', '')}\n"
                    f"Snippet: {result.get('snippet', '')}\n"
                    f"URL: {result.get('link', '')}\n"
                )

        return {"results": "\n\n".join(formatted_results)}

    except Exception as e:
        # Log the string representation of the error for JSON compatibility
        logger.critical(
            "Error in search service", error=str(e), traceback=traceback.format_exc()
        )
        # Re-raise the original exception type if it's HTTPException, otherwise wrap
        if isinstance(e, HTTPException):
            raise
        else:
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
