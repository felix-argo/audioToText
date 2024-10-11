# server.py
import os
import tempfile
import uuid

from flask import Flask, request, jsonify
from waitress import serve

# Создание экземпляра Flask-приложения **перед** импортом обработчика
app = Flask(__name__)

from app.transcribe import process_and_transcribe  # Импорт после создания app


@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({'error': 'Нет файла в запросе'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400
    if file:
        try:
            # Сохранение загруженного файла во временную директорию
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                file.save(temp_audio.name)
                logger.info(f"Файл сохранен во временной директории: {temp_audio.name}")

                # Определение уникальной директории для сохранения результатов
                # Получение пути к директории текущего файла
                project_dir = os.path.dirname(os.path.abspath(__file__))
                # Формирование пути для новой папки в проекте
                output_dir = os.path.join(project_dir, "transcription")
                # Создание директории, если она не существует
                os.makedirs(output_dir, exist_ok=True)

                print(f"Папка создана по пути: {output_dir}")

                # Транскрибирование аудио
                transcription_path = process_and_transcribe(temp_audio.name, output_dir)

                # Чтение содержимого транскрипции
                with open(transcription_path, 'r', encoding='utf-8') as f:
                    transcription_text = f.read()
                return jsonify({'transcription': transcription_text, 'output_dir': output_dir}), 200
        except Exception as e:
            logger.error(f"Ошибка при транскрипции: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500


# Настройка логирования
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=5000, threads=16)  # Установите количество потоков в зависимости от CPU
