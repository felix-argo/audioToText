# app/transcribe.py

import logging
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import whisper
import torch
import librosa
from whisper import transcribe

from .utils import (
    save_audio,
    split_audio,
    increase_volume,
    bandpass_filter,
    normalize_audio,
    convert_to_wav
)

# Настройка логирования
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_whisper_model(model_name="large-v3-turbo"):
    """
    Загружает модель Whisper на доступное устройство (GPU или CPU).

    :param model_name: Название модели Whisper.
    :return: Загруженная модель Whisper.
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Загрузка модели Whisper ({model_name}) на {device}...")
        model = whisper.load_model(model_name, device=device)
        logger.info("Модель Whisper успешно загружена.")
        return model
    except Exception as e:
        logger.error(f"Не удалось загрузить модель Whisper: {e}", exc_info=True)
        raise


def transcribe_chunk(chunk_path):
    """
    Транскрибирует один аудиочанк и сохраняет результат как текстовый файл.

    :param chunk_path: Путь к аудиочанку.
    :return: Путь к файлу с транскрипцией.
    :raises Exception: При возникновении ошибок.
    """
    try:
        model = load_whisper_model()
        result = transcribe(
            model,
            chunk_path,
            initial_prompt = "Совещание, обсуждение",
            language= "ru"
        )
        transcription_text = result["text"]
        # Сохранение транскрипции
        transcription_filename = os.path.splitext(os.path.basename(chunk_path))[0] + ".txt"
        transcription_path = os.path.join(os.path.dirname(chunk_path), transcription_filename)
        with open(transcription_path, 'w', encoding='utf-8') as f:
            f.write(transcription_text)
        logger.info(f"Транскрипция сохранена по пути: {transcription_path}")
        return transcription_path
    except Exception as e:
        logger.error(f"Ошибка при транскрипции чанка {chunk_path}: {e}", exc_info=True)
        raise


def process_and_transcribe(audio_path, output_dir, save_intermediates=True):
    """
    Полный процесс обработки и транскрипции аудио файла.

    :param audio_path: Путь к исходному аудио файлу.
    :param output_dir: Директория для сохранения результатов.
    :param save_intermediates: Сохранять ли промежуточные файлы предобработки.
    :param chunk_length_ms: Длина чанков в миллисекундах.
    :return: Путь к полному файлу транскрипции.
    :raises Exception: При возникновении ошибок.
    """
    try:
        # Конвертация в WAV с нужными параметрами
        wav_path = convert_to_wav(audio_path)
        if os.path.exists(wav_path):
            logger.info(f"Преобразовано в WAV: {wav_path}")
        else:
            logger.error(f"Не удалось преобразовать файл в WAV: {wav_path}")
            raise FileNotFoundError(f"Файл WAV не найден: {wav_path}")

        # Создание уникальной директории для обработки
        unique_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
        processing_dir = os.path.join(output_dir, f"processing_{unique_id}")
        os.makedirs(processing_dir, exist_ok=True)
        logger.info(f"Создана директория для обработки: {processing_dir}")

        # Шаг 1: Загрузка аудио
        audio_data, sample_rate = librosa.load(wav_path, sr=16000, mono=True)
        logger.info(f"Загружено аудио: {wav_path} с частотой дискретизации {sample_rate} Hz")

        if save_intermediates:
            # Шаг 2: Увеличение громкости
            audio_data_increased = increase_volume(audio_data, db_increase=8)
            increased_path = os.path.join(processing_dir, "increase_volume.wav")
            save_audio(audio_data_increased, processing_dir, "increase_volume", sample_rate=sample_rate)
            logger.info(f"Увеличенная громкость сохранена: {increased_path}")

            # Шаг 3: Применение полосового фильтра
            audio_data_filtered = bandpass_filter(audio_data_increased, sample_rate=sample_rate)
            filtered_path = os.path.join(processing_dir, "bandpass_filter.wav")
            save_audio(audio_data_filtered, processing_dir, "bandpass_filter", sample_rate=sample_rate)
            logger.info(f"Полосовой фильтр применен и сохранен: {filtered_path}")

            # Шаг 4: Нормализация аудио
            audio_data_normalized = normalize_audio(audio_data_filtered)
            normalized_path = os.path.join(processing_dir, "normalize_audio.wav")
            save_audio(audio_data_normalized, processing_dir, "normalize_audio", sample_rate=sample_rate)
            logger.info(f"Аудио нормализовано и сохранено: {normalized_path}")

            # Используем нормализованное аудио для дальнейшей обработки
            final_audio_path = normalized_path
        else:
            final_audio_path = wav_path

        # Шаг 5: Разбивка аудио на чанки
        chunks = split_audio(final_audio_path)
        logger.info(f"Всего создано {len(chunks)} чанков для транскрипции.")

        if not chunks:
            logger.error("Нет доступных аудиочанков для транскрипции.")
            raise ValueError("Нет доступных аудиочанков для транскрипции.")

        # Шаг 6: Транскрипция чанков с использованием ProcessPoolExecutor
        transcriptions = []
        max_workers = min(1, len(chunks))  # Ограничение числа процессов
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {executor.submit(transcribe_chunk, chunk): chunk for chunk in chunks}
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    transcription = future.result()
                    transcriptions.append((chunks.index(chunk), transcription))
                except Exception as exc:
                    logger.error(f"Чанк {chunk} вызвал исключение: {exc}")

        # Сортировка транскрипций по порядку чанков
        transcriptions.sort(key=lambda x: x[0])
        full_transcription = "\n".join([text for idx, text in transcriptions])

        # Шаг 7: Постобработка транскрипции (если необходимо)
        # Например, исправление ошибок, использование LanguageTool и т.д.

        # Шаг 8: Сохранение полной транскрипции
        transcription_filename = os.path.splitext(os.path.basename(audio_path))[0] + "_full.txt"
        transcription_path = os.path.join(processing_dir, transcription_filename)
        with open(transcription_path, 'w', encoding='utf-8') as f:
            f.write(full_transcription)
        logger.info(f"Полная транскрипция сохранена по пути: {transcription_path}")

        return transcription_path

    except Exception as e:
        logger.error(f"Ошибка в процессе транскрипции: {e}", exc_info=True)
        raise  # Проброс исключения вверх


# Функция для тестирования транскрипции отдельного файла
def transcribe_single_audio(audio_path, output_dir):
    try:
        transcription_path = process_and_transcribe(audio_path, output_dir)
        if transcription_path:
            logger.info(f"Транскрипция успешно завершена и сохранена по пути: {transcription_path}")
            return transcription_path
    except Exception as e:
        logger.error(f"Транскрипция не была завершена из-за ошибок: {e}", exc_info=True)
    return None


# Пример использования
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Транскрипция аудио файлов с использованием Whisper.")
    parser.add_argument("audio_path", type=str, help="Путь к исходному аудио файлу.")
    parser.add_argument("output_dir", type=str, help="Директория для сохранения результатов транскрипции.")
    parser.add_argument("--no-intermediates", action="store_true", help="Не сохранять промежуточные файлы предобработки.")
    parser.add_argument("--chunk_length_min", type=int, default=5, help="Длина чанков в минутах. По умолчанию 5 минут.")

    args = parser.parse_args()

    audio_path = args.audio_path
    output_dir = args.output_dir
    save_intermediates = not args.no_intermediates

    transcribe_single_audio(audio_path, output_dir)
