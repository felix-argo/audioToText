import logging
import os
from datetime import datetime

import librosa
import torch
import whisper

from app.utils import increase_volume, bandpass_filter, convert_to_wav, normalize_audio, \
    save_audio, audiosegment_to_numpy, numpy_to_audiosegment

# Existing logging setup...
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Глобальная переменная для модели Whisper
whisper_model = None

def load_whisper_model(model_name="turbo"):
    """
    Загружает модель Whisper и сохраняет её в глобальной переменной.
    """
    global whisper_model
    if whisper_model is None:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Загрузка модели Whisper ({model_name}) на {device}...")
            whisper_model = whisper.load_model(model_name, device=device)
            logger.info("Модель Whisper успешно загружена.")
        except Exception as e:
            logger.error(f"Не удалось загрузить модель Whisper: {e}")
            raise

def process_and_transcribe(audio_path, output_dir, save_intermediates=True):
    """
    Предобрабатывает и транскрибирует аудио файл, сохраняя все промежуточные файлы в одной директории.
    """
    try:
        load_whisper_model()
        wav_path = convert_to_wav(audio_path)

        # Создание уникальной директории для обработки
        unique_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        processing_dir = os.path.join(output_dir, f"processing_{unique_id}")
        os.makedirs(processing_dir, exist_ok=True)
        logger.info(f"Создана директория для обработки: {processing_dir}")

        # Загрузка и нормализация аудио
        audio_data, sample_rate = librosa.load(wav_path, sr=None, mono=True)
        logger.info(f"Загружено аудио: {wav_path} с частотой дискретизации {sample_rate} Hz")
        audio_data = increase_volume(audio_data, sample_rate)
        save_audio(audio_data, processing_dir, "increase_volume")

        audio_data = audiosegment_to_numpy(audio_data)
        audio_data = bandpass_filter(audio_data, sample_rate)
        audio_data = numpy_to_audiosegment(audio_data, sample_rate)
        save_audio(audio_data, processing_dir, "bandpass_filter")

        audio_np = audiosegment_to_numpy(audio_data)
        audio_np = normalize_audio(audio_np)
        audio_data = numpy_to_audiosegment(audio_np, sample_rate)
        save_audio(audio_data, processing_dir, "normalize_audio")

        normalize_audio_path = os.path.join(processing_dir, "audio_normalize_audio.wav")
        # Транскрипция с помощью Whisper
        result = whisper_model.transcribe(
            normalize_audio_path,
            language="ru",
            beam_size=7,
            best_of=5,
            temperature=0,
            condition_on_previous_text=True
        )
        transcription_text = result["text"].strip()
        logger.info("Транскрипция успешно выполнена.")

        # Сохранение транскрипции
        transcription_filename = os.path.splitext(os.path.basename(audio_path))[0] + ".txt"
        transcription_path = os.path.join(processing_dir, transcription_filename)
        with open(transcription_path, 'w', encoding='utf-8') as f:
            f.write(transcription_text)
        logger.info(f"Транскрипция сохранена по пути {transcription_path}")

        return transcription_path

    except Exception as e:
        logger.error(f"Ошибка в процессе транскрипции: {e}", exc_info=True)
        raise
