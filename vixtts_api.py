import os
import string
from unidecode import unidecode
import tempfile
from datetime import datetime
from typing import Optional

import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from underthesea import sent_tokenize
from vinorm import TTSnorm

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Khởi tạo FastAPI app
app = FastAPI(
    title="viXTTS API",
    description="API for Vietnamese Text-to-Speech using viXTTS model",
    version="1.0.0"
)

# Các biến toàn cục
XTTS_MODEL = None
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
REFERENCE_AUDIO = os.path.join(SCRIPT_DIR, "assets", "vixtts_sample_female.wav")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model request
class TTSRequest(BaseModel):
    text: str
    language: str = "vi"
    normalize_text: bool = True

def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_model():
    global XTTS_MODEL
    clear_gpu_cache()
    os.makedirs(MODEL_DIR, exist_ok=True)

    required_files = ["model.pth", "config.json", "vocab.json", "speakers_xtts.pth"]
    files_in_dir = os.listdir(MODEL_DIR)
    
    if not all(file in files_in_dir for file in required_files):
        raise HTTPException(
            status_code=500,
            detail="Model files not found. Please ensure all required files are in the model directory."
        )

    xtts_config = os.path.join(MODEL_DIR, "config.json")
    config = XttsConfig()
    config.load_json(xtts_config)
    XTTS_MODEL = Xtts.init_from_config(config)
    XTTS_MODEL.load_checkpoint(config, checkpoint_dir=MODEL_DIR, use_deepspeed=False)
    
    if torch.cuda.is_available():
        XTTS_MODEL.cuda()

def normalize_vietnamese_text(text):
    text = (
        TTSnorm(text, unknown=False, lower=False, rule=True)
        .replace("..", ".")
        .replace("!.", "!")
        .replace("?.", "?")
        .replace(" .", ".")
        .replace(" ,", ",")
        .replace('"', "")
        .replace("'", "")
        .replace("AI", "Ây Ai")
        .replace("A.I", "Ây Ai")
    )
    return text

def calculate_keep_len(text, lang):
    if lang in ["ja", "zh-cn"]:
        return -1

    word_count = len(text.split())
    num_punct = text.count(".") + text.count("!") + text.count("?") + text.count(",")

    if word_count < 5:
        return 15000 * word_count + 2000 * num_punct
    elif word_count < 10:
        return 13000 * word_count + 2000 * num_punct
    return -1

def get_file_name(text, max_char=50):
    filename = text[:max_char]
    filename = filename.lower()
    filename = filename.replace(" ", "_")
    filename = filename.translate(str.maketrans("", "", string.punctuation.replace("_", "")))
    filename = unidecode(filename)
    current_datetime = datetime.now().strftime("%m%d%H%M%S")
    filename = f"{current_datetime}_{filename}"
    return filename

@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/tts", response_class=FileResponse)
async def text_to_speech(request: TTSRequest):
    """
    Convert text to speech using viXTTS model
    """
    if XTTS_MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Chuẩn hóa văn bản nếu cần
    if request.normalize_text and request.language == "vi":
        text = normalize_vietnamese_text(request.text)
    else:
        text = request.text

    # Tách câu
    if request.language in ["ja", "zh-cn"]:
        sentences = text.split("。")
    else:
        sentences = sent_tokenize(text)

    # Lấy conditioning latents từ file âm thanh mẫu
    gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
        audio_path=REFERENCE_AUDIO,
        gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
        max_ref_length=XTTS_MODEL.config.max_ref_len,
        sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
    )

    # Tạo âm thanh cho từng câu
    wav_chunks = []
    for sentence in sentences:
        if sentence.strip() == "":
            continue
            
        wav_chunk = XTTS_MODEL.inference(
            text=sentence,
            language=request.language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=0.3,
            length_penalty=1.0,
            repetition_penalty=10.0,
            top_k=30,
            top_p=0.85,
            enable_text_splitting=True,
        )

        keep_len = calculate_keep_len(sentence, request.language)
        wav_chunk["wav"] = wav_chunk["wav"][:keep_len]
        wav_chunks.append(torch.tensor(wav_chunk["wav"]))

    # Ghép các đoạn âm thanh lại
    out_wav = torch.cat(wav_chunks, dim=0).unsqueeze(0)
    
    # Tạo tên file output
    output_filename = f"{get_file_name(text)}.wav"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # Lưu file âm thanh
    torchaudio.save(output_path, out_wav, 24000)
    
    # Trả về file âm thanh
    return FileResponse(
        path=output_path,
        media_type="audio/wav",
        filename=output_filename
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)