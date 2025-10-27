import os
import time
import shutil
import torch
import uvicorn
import tempfile
import asyncio
import logging
from typing import Optional, List
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request, Depends, Header, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import soundfile as sf
import subprocess
from dotenv import load_dotenv
from whisperx_asr import load_whisperx_models, unload_whisperx_models, process_audio_with_whisperx, get_model_status
import gc

load_dotenv()

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ContextFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'request_id'):
            record.request_id = 'no-request'
        return True

logging.getLogger().addFilter(ContextFilter())

SUPPORTED_LANGUAGES = {
    "auto": "Auto-detect",
    "en": "English", 
    "ar": "Arabic",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
}

API_KEY = os.getenv("API_KEY", "")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 100))

class TranscriptionRequest(BaseModel):
    language: Optional[str] = Field(default="auto", description="Language code for transcription")

class TranscriptionResponse(BaseModel):
    transcription: str
    language: Optional[str] = None
    processing_time: float
    segments: Optional[List[dict]] = None
    word_segments: Optional[List[dict]] = None
    segment_count: int

def verify_api_key(x_api_key: Optional[str] = Header(None)):    
    # if not API_KEY:
    #     logger.warning("No API key configured")
    #     return
    # if x_api_key != API_KEY:
    #     raise HTTPException(
    #         status_code=status.HTTP_401_UNAUTHORIZED,
    #         detail="Invalid or missing API Key"
    #     )
    # disabled for now
    pass

# Function to clear model cache
def clear_model_cache():
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    if os.path.exists(cache_dir):
        try:
            for item in os.listdir(cache_dir):
                if item not in ['currently_downloading']:
                    item_path = os.path.join(cache_dir, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    global whisperx_model, whisperx_align_model, whisperx_metadata
    
    # Check if we're in startup mode
    is_startup = os.getenv("STARTUP_MODE", "false").lower() == "true"
    
    if is_startup:
        logger.info("Running in STARTUP_MODE - skipping model loading")
        # Set empty placeholders for models
        whisperx_model = None
        whisperx_align_model = None
        whisperx_metadata = None
        yield
        return
    
    try:
        load_whisperx_models()
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("Shutting down...")
    unload_whisperx_models()

app = FastAPI(
    title="WhisperX API Server",
    description="Production-ready speech-to-text API using WhisperX with alignment",
    version="2.0.0",
    lifespan=lifespan
)

# Serve static files
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom logging middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    import uuid
    request_id = str(uuid.uuid4())
    logger_with_id = logger.getChild(request_id)
    logger_with_id.addFilter(ContextFilter())
    request.state.logger = logger_with_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# API endpoints
@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("static/index.html", media_type="text/html")

@app.get("/health")
async def health_check(x_api_key: str = Depends(verify_api_key)):
    # Simple check that always returns OK during initial startup
    is_startup = os.getenv("STARTUP_MODE", "false").lower() == "true"
    
    if is_startup:
        return {
            "status": "starting",
            "model_loading": True
        }
    
    model_status = get_model_status()
    return {
        "status": "healthy",
        **model_status
    }

@app.get("/languages")
async def get_supported_languages(x_api_key: str = Depends(verify_api_key)):
    return {"supported_languages": SUPPORTED_LANGUAGES}

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    request: Request,
    file: UploadFile = File(..., description="Audio file to transcribe"),
    language: str = Query(default="auto", description="Language code for transcription"),
    batch_size: int = Query(default=16, description="Batch size for processing"),
    x_api_key: str = Depends(verify_api_key)
):
    logger = request.state.logger
    start_time = time.time()
    
    # Input validation
    if language not in SUPPORTED_LANGUAGES and language != "auto":
        raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")
    
    if batch_size < 1 or batch_size > 32:
        raise HTTPException(status_code=400, detail="Batch size must be between 1 and 32")
    
    # File validation
    allowed_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm'}
    file_ext = os.path.splitext(file.filename.lower())[1] if file.filename else ''
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
        )
    
    uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)
    temp_path = None
    converted_path = None
    
    try:
        # Save uploaded file
        file_extension = file_ext if file_ext else '.wav'
        safe_filename = file.filename if file.filename else f'upload_{int(time.time())}{file_extension}'
        safe_filename = os.path.basename(safe_filename)
        temp_path = os.path.join(uploads_dir, safe_filename)
        
        with open(temp_path, 'wb') as temp_file:
            while chunk := await file.read(1024 * 1024):
                temp_file.write(chunk)
        
        file_size = os.path.getsize(temp_path)
        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_FILE_SIZE_MB}MB")
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty file provided")
        
        # Convert .webm to .wav if needed
        audio_path = temp_path
        if file_ext == '.webm':
            converted_path = temp_path + '.wav'
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-i', temp_path,
                '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', converted_path
            ]
            try:
                result = subprocess.run(
                    ffmpeg_cmd, 
                    check=True, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE
                )
                logger.info("FFmpeg conversion completed")
                audio_path = converted_path
            except Exception as e:
                logger.error(f"FFmpeg conversion failed: {e}")
                raise HTTPException(status_code=500, detail="Failed to convert webm to wav")
        
        # Get audio duration
        with sf.SoundFile(audio_path) as audio:
            duration = len(audio) / audio.samplerate
        
        logger.info(f"Processing audio: {file.filename} ({file_size/1024/1024:.2f} MB, {duration:.2f}s)")
        
        # Process with WhisperX
        whisperx_result = process_audio_with_whisperx(
            audio_path=audio_path,
            language=language,
            batch_size=batch_size
        )
        
        result = whisperx_result["result"]
        detected_language = whisperx_result["language"]
        
        # Extract segments and word-level timestamps
        segments = []
        word_segments = []
        full_transcription = ""
        
        for segment in result.get("segments", []):
            segment_data = {
                "text": segment.get("text", "").strip(),
                "start": segment.get("start"),
                "end": segment.get("end")
            }
            
            segments.append(segment_data)
            full_transcription += segment_data["text"] + " "
            
            # Extract word-level timestamps if available
            if "words" in segment:
                for word in segment["words"]:
                    word_data = {
                        "word": word.get("word", ""),
                        "start": word.get("start"),
                        "end": word.get("end"),
                        "confidence": word.get("score")
                    }
                    word_segments.append(word_data)
        
        full_transcription = full_transcription.strip()
        processing_time = time.time() - start_time
        
        response_data = {
            "transcription": full_transcription,
            "language": detected_language,
            "processing_time": round(processing_time, 2),
            "segments": segments,
            "word_segments": word_segments if word_segments else None,
            "segment_count": len(segments)
        }
        
        logger.info(
            f"Transcription completed in {processing_time:.2f}s, "
            f"{len(segments)} segments"
        )
        
        return TranscriptionResponse(**response_data)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during transcription")
    finally:
        # Clean up temp files
        for path in [temp_path, converted_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {path}: {e}")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger = getattr(request.state, 'logger', logger)
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "An internal server error occurred"})

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    args = parser.parse_args()
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    workers = int(os.getenv("WORKERS", max(1, os.cpu_count() // 2)))
    
    logger.info(f"Starting WhisperX API server on {host}:{port} with {workers} workers")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=workers,
        reload=args.reload,
        log_level="info",
        access_log=True,
        timeout_keep_alive=30,
        limit_concurrency=100,
        limit_max_requests=1000
    )