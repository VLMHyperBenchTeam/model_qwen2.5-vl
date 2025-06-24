import os
import subprocess

from model_interface.model_factory import ModelFactory


if __name__ == "__main__":
    # Инициализируем модель одной строкой
    model = ModelFactory.initialize_qwen_model(model_name="Qwen2.5-VL-7B-Instruct")

    # отвечаем на вопрос о по одной картинке
    image_path = "example_docs/old_tins/1.jpg"
    question = """
Подано изображение свидетельства о постановке на учет физического лица.
Пожалуйста, извлеките информацию и представьте её в виде структурированного JSON-объекта с указанными полями.

Поля для извлечения:
- "country": Страна (например, "Российская Федерация")
- "type": "СВИДЕТЕЛЬСТВО О ПОСТАНОВКЕ НА УЧЕТ ФИЗИЧЕСКОГО ЛИЦА В НАЛОГОВОМ ОРГАНЕ"
- "inn": "ИНН (12 цифр)"
- "fio": Полное имя владельца (Фамилия Имя Отчество)
- "date_of_birth": Дата рождения (в формате DD.MM.YYYY)
- "place_of_birth": Место рождения
- "gender": Пол ("Муж." или "Жен.")
- "issue_date": Дата выдачи (в формате DD.MM.YYYY)
- "issue_place": Место выдачи

JSON-структура:
{
  "country": "",
  "type": "",
  "inn": "",
  "fio": "",
  "date_of_birth": "",
  "place_of_birth": "",
  "gender": "",
  "issue_date": "",
  "issue_place": ""
}
"""
    model_answer = model.predict_on_image(image=image_path, prompt=question)
    print(model_answer)
    
    subprocess.run(["nvidia-smi"])

    print(model_answer)
