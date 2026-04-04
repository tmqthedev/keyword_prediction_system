from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import logging
from pathlib import Path

# Cấu hình logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

# Đọc và xử lý dữ liệu CSV
try:
    data_path = Path(__file__).resolve().parent / 'data.csv'
    df = pd.read_csv(data_path, encoding='utf-8', skipinitialspace=True)
    expected_columns = {'query', 'keyword'}
    if not expected_columns.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {expected_columns}")
    logger.info(f"Loaded data shape: {df.shape}")
    logger.info(f"Sample data:\n{df.head()}")
except Exception as e:
    logger.error(f"Error loading CSV: {e}")
    df = pd.DataFrame(columns=['query', 'keyword'])

@app.post("/suggest", responses={400: {"description": "Bad request"}, 500: {"description": "Internal server error"}})
async def suggest_keywords(request: QueryRequest):
    query = request.query.lower().strip()
    logger.debug(f"Received query: {query}")

    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty")

    try:
        # Tìm từ khóa liên quan
        related_keywords = df[df['query'].str.lower().str.contains(query, na=False, regex=False)]['keyword'].tolist()
        related_queries = df[df['keyword'].str.lower().str.contains(query, na=False, regex=False)]['query'].tolist()

        # Kết hợp và làm sạch kết quả
        suggestions = sorted({s.strip() for s in related_keywords + related_queries if s.strip().lower() != query})
        
        logger.debug(f"Found suggestions: {suggestions}")
        
        return {
            "query": query,
            "suggested_keywords": suggestions
        }
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "data_loaded": len(df) > 0}