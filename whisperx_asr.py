import os
import time
import torch
import logging
import whisperx

logger = logging.getLogger(__name__)

# Global variables for WhisperX
whisperx_model = None
whisperx_align_model = None
whisperx_metadata = None

MODEL_SIZE = os.getenv("MODEL_SIZE", "large-v2")
ENABLE_ALIGNMENT = os.getenv("ENABLE_ALIGNMENT", "true").lower() == "true"

def clear_model_cache():
    """Clear Hugging Face model cache to save space"""
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    if os.path.exists(cache_dir):
        try:
            for item in os.listdir(cache_dir):
                if item not in ['currently_downloading']:
                    item_path = os.path.join(cache_dir, item)
                    if os.path.isdir(item_path):
                        import shutil
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")

def load_whisperx_models():
    """Load WhisperX models and return them"""
    global whisperx_model, whisperx_align_model, whisperx_metadata
    
    start_time = time.time()
    try:
        # Load models with specific device configs
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() else "int8"
        
        logger.info(f"Loading WhisperX model {MODEL_SIZE} on {device}...")
        
        # save models in root
        model_dir = "/app/models"
        os.makedirs(model_dir, exist_ok=True)
        
        whisperx_model = whisperx.load_model(
            MODEL_SIZE, 
            device,
            compute_type=compute_type,
            download_root=model_dir 
        )
        
        # alignment model
        if ENABLE_ALIGNMENT:
            logger.info("Loading alignment model...")
            whisperx_align_model, whisperx_metadata = whisperx.load_align_model(
                language_code="en", 
                device=device
            )
            
        # Clear cache after loading models to save space
        clear_model_cache()
        
        logger.info(f"Models loaded in {time.time() - start_time:.2f} seconds")
        return True
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

def unload_whisperx_models():
    """Unload models and clean up memory"""
    global whisperx_model, whisperx_align_model, whisperx_metadata
    
    logger.info("Unloading WhisperX models...")
    if whisperx_model:
        del whisperx_model
    if whisperx_align_model:
        del whisperx_align_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    whisperx_model = None
    whisperx_align_model = None
    whisperx_metadata = None

def process_audio_with_whisperx(
    audio_path: str, 
    language: str = "auto",
    batch_size: int = 16
) -> dict:
    """Process audio with WhisperX including alignment"""
    
    if not whisperx_model:
        raise RuntimeError("WhisperX model not loaded")
    
    try:
        # Load and transcribe audio
        logger.info("Loading audio...")
        audio = whisperx.load_audio(audio_path)
        
        logger.info("Starting transcription...")
        transcribe_start = time.time()
        
        # Transcribe with WhisperX
        result = whisperx_model.transcribe(
            audio, 
            batch_size=batch_size,
            language=language if language != "auto" else None
        )
        
        transcribe_time = time.time() - transcribe_start
        logger.info(f"Transcription completed in {transcribe_time:.2f}s")
        
        detected_language = result.get("language", "unknown")
        
        # Align whisper output if enabled
        aligned_result = result
        if ENABLE_ALIGNMENT and result["segments"]:
            try:
                logger.info(f"Loading alignment model for language: {detected_language}")
                align_start = time.time()
                
                # Load alignment model for detected language
                model_a, metadata = whisperx.load_align_model(
                    language_code=detected_language, 
                    device=whisperx_model.device
                )
                
                # Align whisper output
                aligned_result = whisperx.align(
                    result["segments"], 
                    model_a, 
                    metadata, 
                    audio, 
                    whisperx_model.device, 
                    return_char_alignments=False
                )
                
                align_time = time.time() - align_start
                logger.info(f"Alignment completed in {align_time:.2f}s")
                
                # Clean up alignment model
                del model_a
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.warning(f"Alignment failed: {e}, using original result")
                aligned_result = result
        
        return {
            "result": aligned_result,
            "language": detected_language
        }
        
    except Exception as e:
        logger.error(f"Error in WhisperX processing: {e}")
        raise

def get_model_status():
    """Get current model loading status"""
    return {
        "model_loaded": whisperx_model is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_memory": torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else None,
        "model_size": MODEL_SIZE,
        "alignment_enabled": ENABLE_ALIGNMENT
    }