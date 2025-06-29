import subprocess
from typing import Any
from model_qwen2_5_vl import initialize_qwen_model

if __name__ == "__main__":
    # Инициализируем модель одной строкой
    model: Any = initialize_qwen_model(model_name="Qwen2.5-VL-7B-Instruct")

    # отвечаем на вопрос о по одной картинке
    image_path = "example_docs/passport/9.jpg"
    question = """
Подано изображение паспорта Российской Федерации.
Пожалуйста, извлеките информацию и представьте её в виде структурированного JSON-объекта с указанными полями.

Поля для извлечения:
- "country": Страна (например, "Российская Федерация")
- "type": "ПАСПОРТ",
- "passport_series": "серия паспорта (две цифры через пробел)"
- "passport_number": "номер паспорта (шесть цифр)"
- "fio": Полное имя владельца (Фамилия Имя Отчество)
- "date_of_birth": Дата рождения (в формате DD.MM.YYYY)
- "place_of_birth": Место рождения
- "gender": Пол ("Муж." или "Жен.")
- "issue_date": Дата выдачи (в формате DD.MM.YYYY)
- "issue_place": Место выдачи
- "issue_code": Код подразделения

JSON-структура:
{
  "country": "",
  "type": "",
  "passport_series": "",
  "passport_number": "",
  "fio": "",
  "date_of_birth": "",
  "place_of_birth": "",
  "gender": "",
  "issue_date": "",
  "issue_place": "",
  "issue_code": ""
}
"""
    model_answer = model.predict_on_image(image=image_path, prompt=question)
    print(model_answer)
    
    subprocess.run(["nvidia-smi"])

    print(model_answer)
