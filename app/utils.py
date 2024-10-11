# Define processing functions
# Existing logging setup...
import logging
import os
import shutil
import tempfile
from datetime import datetime, timedelta

import noisereduce as nr
import numpy as np
from pydub import AudioSegment, effects
from scipy.signal import butter, lfilter, sosfilt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_audio(audio_data, processing_dir, fileName, sample_rate = 16000):
    logger.info(f"Сохранение файла {fileName}")
    step_filename = f"audio_{fileName}.wav"
    logger.info(f"Название файла {step_filename}")
    step_path = os.path.join(processing_dir, step_filename)
    audio_data.export(step_path, format="wav")
    logger.info(f"Файл сохранён: {step_path}")

def audiosegment_to_numpy(audio_segment):
    samples = np.array(audio_segment.get_array_of_samples())
    return samples.astype(np.float32) / 32768.0  # Приведение к диапазону [-1, 1]

def numpy_to_audiosegment(audio_np, sample_rate=16000):
    audio_int16 = np.int16(audio_np * 32767)
    return AudioSegment(
        audio_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    )

def increase_volume(audio_data, sample_rate = 16000, db_increase=10):
    try:
        logger.info(f"Увеличение громкости на {db_increase} dB.")
        max_val = np.max(np.abs(audio_data))
        audio_int16 = np.int16(audio_data / max_val * 32767) if max_val != 0 else np.int16(audio_data)
        current_audio = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )
        logger.info("Увеличение громкости собвственно")
        return current_audio.apply_gain(db_increase)
    except Exception as e:
        logger.error(f"Ошибка при увеличении громкости: {e}", exc_info=True)
        return audio_data

def bandpass_filter(audio, sample_rate=16000, lowcut=400, highcut=3999  , order=2):
    """
      Применяет полосовой фильтр к аудио данным для усиления частот речи.
      """
    try:
        logger.info("Применение полосового фильтра к аудио.")
        nyquist = 0.5 * sample_rate

        if lowcut >= highcut:
            raise ValueError("lowcut должен быть меньше highcut")
        if highcut >= nyquist:
            raise ValueError(f"highcut должен быть меньше Найквистовой частоты ({nyquist} Гц)")
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        # Приведение коэффициентов к float32
        b = b.astype(np.float32)
        a = a.astype(np.float32)

        # Убедитесь, что audio тоже float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        y = lfilter(b, a, audio)
        logger.info("Полосовой фильтр применен.")
        return y
    except Exception as e:
        logger.error(f"Ошибка при применении полосового фильтра: {e}", exc_info=True)
        return audio


def normalize_audio(audio):
    """
    Нормализует аудио данные для консистентного уровня громкости.
    """
    try:
        peak = np.max(np.abs(audio))
        if peak == 0:
            return audio
        normalized_audio = audio / peak
        logger.info("Аудио успешно нормализовано.")
        logger.info(f"Размер audio_np после нормализации: {audio.shape}")
        return normalized_audio
    except Exception as e:
        logger.error(f"Ошибка при нормализации аудио: {e}", exc_info=True)
        return audio

def convert_to_wav(file_path):
    """
    Конвертирует аудио файл в WAV формат при необходимости.
    """
    try:
        if not file_path.lower().endswith('.wav'):
            logger.info("Конвертация файла в формат WAV.")
            sound = AudioSegment.from_file(file_path)
            wav_path = os.path.splitext(file_path)[0] + '.wav'
            sound.export(wav_path, format="wav")
            logger.info(f"Файл сконвертирован в WAV: {wav_path}")
            return wav_path
        else:
            logger.info("Файл уже в формате WAV.")
            return file_path
    except Exception as e:
        logger.error(f"Ошибка при конвертации в WAV: {e}", exc_info=True)
        raise

def evaluate_variant_audio(audio_segment):
    """
    Оценивает качество аудио сегмента по уровню громкости и динамики.
    Возвращает метрику, по которой можно сравнить варианты.
    """
    loudness = audio_segment.dBFS
    dynamic_range = effects.dynamics_compression(audio_segment).dBFS - loudness
    return loudness + dynamic_range  # Пример метрики
