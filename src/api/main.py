from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import sys
from typing import Dict, Any, Optional, List

# Add the project directory to the path so we can import other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import project-specific modules
from src.models.bert_model import BERTSentimentModel
from src.preprocessing.text_cleaner import TextCleaner

# Define the request model
class SentimentRequest(BaseModel):
    text: str
    model_type: Optional[str] = "bert"  # "bert", "rnn", or "traditional"
    detailed: Optional[bool] = False

# Define the response model
class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    analysis: Dict[str, float]
    processed_text: Optional[str] = None

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="A machine learning API for real-time sentiment analysis of text",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize models (lazy loading)
models = {}
text_cleaner = TextCleaner()

def get_model(model_type: str):
    """Lazy load the requested model."""
    if model_type not in models:
        if model_type == "bert":
            models[model_type] = BERTSentimentModel()
        # Add other model types as needed
    return models[model_type]

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Sentiment Analysis API",
        "docs": "/docs",
        "health": "/health",
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    try:
        # Clean and preprocess the text
        cleaned_text = text_cleaner.clean(request.text)
        
        # Get the appropriate model
        model = get_model(request.model_type)
        
        # Analyze the sentiment
        sentiment_result = model.predict(cleaned_text)
        
        # Format the response
        response = {
            "sentiment": sentiment_result["label"],
            "confidence": sentiment_result["confidence"],
            "analysis": {
                "positive_score": sentiment_result["scores"]["positive"],
                "negative_score": sentiment_result["scores"]["negative"],
                "neutral_score": sentiment_result["scores"]["neutral"],
            }
        }
        
        # Include processed text if detailed results requested
        if request.detailed:
            response["processed_text"] = cleaned_text
            
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiment: {str(e)}")

@app.post("/batch-analyze")
async def batch_analyze(requests: List[SentimentRequest]):
    results = []
    for request in requests:
        try:
            result = await analyze_sentiment(request)
            results.append({"text": request.text, "result": result})
        except Exception as e:
            results.append({"text": request.text, "error": str(e)})
    return results

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 