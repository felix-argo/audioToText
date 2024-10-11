# app/utils.py

import logging
import os
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter
from pydub import effects
import librosa  # Убедитесь, что импортировали librosa

logger = logging.getLogger(__name__)


def save_audio(audio_data, processing_dir, file_name, sample_rate=16000):
    """
    Сохраняет NumPy-массив аудио как WAV-файл с использованием библиотеки soundfile.

    :param audio_data: NumPy-массив аудио данных.
    :param processing_dir: Директория для сохранения файла.
    :param file_name: Название файла без расширения.
    :param sample_rate: Частота дискретизации.
    """
    logger.info(f"Сохранение файла {file_name}.wav")
    step_filename = f"{file_name}.wav"  # Убираем префикс 'audio_'
    step_path = os.path.join(processing_dir, step_filename)
    try:
        sf.write(step_path, audio_data, sample_rate)
        logger.info(f"Файл сохранён: {step_path}")
    except Exception as e:
        logger.error(f"Не удалось сохранить файл {step_path}: {e}", exc_info=True)


def split_audio(audio_path, chunk_length_ms=30 * 60 * 1000):
    """
    Разбивает аудио на чанки заданной длины.

    :param audio_path: Путь к исходному аудио файлу.
    :param chunk_length_ms: Длина чанка в миллисекундах. По умолчанию 5 минут.
    :return: Список путей к чанкам.
    """
    try:
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        duration_ms = int(len(audio) / sr * 1000)
        logger.info(f"Длительность исходного аудио: {duration_ms} ms")
        chunks = []
        for i in range(0, duration_ms, chunk_length_ms):
            start_sample = int(i / 1000 * sr)
            end_sample = int((i + chunk_length_ms) / 1000 * sr)
            chunk_audio = audio[start_sample:end_sample]
            chunk_duration = len(chunk_audio) / sr
            logger.debug(f"Чанк {i // chunk_length_ms}: начало {start_sample}, конец {end_sample}, длительность {chunk_duration:.2f} секунд")
            if chunk_duration < 5:  # Минимальная длина: 5 секунд
                logger.warning(f"Чанк слишком короткий ({chunk_duration:.2f} секунд): {i // chunk_length_ms}. Удаление.")
                continue
            chunk_filename = f"chunk_{i // chunk_length_ms}.wav"
            chunk_path = os.path.join(os.path.dirname(audio_path), chunk_filename)
            save_audio(chunk_audio, os.path.dirname(audio_path), chunk_filename.replace('.wav', ''), sample_rate=sr)
            chunks.append(chunk_path)
            logger.info(f"Создан чанк: {chunk_path} с длиной {chunk_duration:.2f} секунд")
        logger.info(f"Всего создано {len(chunks)} чанков.")
        return chunks
    except Exception as e:
        logger.error(f"Не удалось разбить аудио {audio_path} на чанки: {e}", exc_info=True)
        return []


def increase_volume(audio_data, db_increase=7):
    """
    Увеличивает громкость аудио на заданное количество децибел.

    :param audio_data: NumPy-массив аудио данных.
    :param db_increase: Количество децибел для увеличения громкости.
    :return: NumPy-массив с увеличенной громкостью.
    """
    try:
        logger.info(f"Увеличение громкости на {db_increase} dB.")
        peak = np.max(np.abs(audio_data))
        if peak == 0:
            logger.warning("Максимальное значение аудио равно 0. Пропуск увеличения громкости.")
            return audio_data
        gain = 10 ** (db_increase / 20)
        audio_increased = audio_data * gain
        audio_increased = np.clip(audio_increased, -1.0, 1.0)  # Ограничение диапазона
        logger.info("Усиление громкости успешно применено.")
        return audio_increased
    except Exception as e:
        logger.error(f"Ошибка при увеличении громкости: {e}", exc_info=True)
        return audio_data


def bandpass_filter(audio, sample_rate=16000, lowcut=400, highcut=3999, order=2):
    """
    Применяет полосовой фильтр к аудио данным для усиления частот речи.

    :param audio: NumPy-массив аудио данных.
    :param sample_rate: Частота дискретизации.
    :param lowcut: Нижняя граница полосы пропускания.
    :param highcut: Верхняя граница полосы пропускания.
    :param order: Порядок фильтра.
    :return: Отфильтрованное аудио.
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
        y = lfilter(b, a, audio)
        logger.info("Полосовой фильтр успешно применен.")
        return y
    except Exception as e:
        logger.error(f"Ошибка при применении полосового фильтра: {e}", exc_info=True)
        return audio


def normalize_audio(audio):
    """
    Нормализует аудио данные для консистентного уровня громкости.

    :param audio: NumPy-массив аудио данных.
    :return: Нормализованное аудио.
    """
    try:
        peak = np.max(np.abs(audio))
        if peak == 0:
            logger.warning("Максимальное значение аудио равно 0. Пропуск нормализации.")
            return audio
        normalized_audio = audio / peak
        logger.info("Аудио успешно нормализовано.")
        logger.debug(f"Размер audio_np после нормализации: {audio.shape}")
        logger.debug(f"Минимальное значение: {normalized_audio.min()}, Максимальное значение: {normalized_audio.max()}")
        return normalized_audio
    except Exception as e:
        logger.error(f"Ошибка при нормализации аудио: {e}", exc_info=True)
        return audio


def convert_to_wav(file_path):
    """
    Конвертирует аудио файл в WAV формат с монофоническим звуком и частотой 16000 Гц при необходимости.

    :param file_path: Путь к исходному аудио файлу.
    :return: Путь к WAV файлу.
    """
    try:
        if not file_path.lower().endswith('.wav'):
            logger.info("Конвертация файла в формат WAV.")
            audio, sr = librosa.load(file_path, sr=16000, mono=True)
            wav_path = os.path.splitext(file_path)[0] + '.wav'
            sf.write(wav_path, audio, sr)
            logger.info(f"Файл сконвертирован в WAV: {wav_path}")
            return wav_path
        else:
            logger.info("Файл не в формате WAV.")
            # Проверка и приведение к моно и 16000 Гц
            audio, sr = librosa.load(file_path, sr=16000, mono=True)
            sf.write(file_path, audio, sr)
            logger.info(f"Файл приведён к моно и 16000 Гц: {file_path}")
            return file_path
    except Exception as e:
        logger.error(f"Ошибка при конвертации в WAV: {e}", exc_info=True)
        raise


def evaluate_variant_audio(audio_segment):
    """
    Оценивает качество аудио сегмента по уровню громкости и динамики.
    Возвращает метрику, по которой можно сравнить варианты.

    :param audio_segment: AudioSegment объект.
    :return: Метрика качества аудио.
    """
    loudness = audio_segment.dBFS
    dynamic_range = effects.dynamics_compression(audio_segment).dBFS - loudness
    return loudness + dynamic_range  # Пример метрики
