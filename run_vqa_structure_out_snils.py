import subprocess

from model_interface.model_factory import ModelFactory

if __name__ == "__main__":
    # Инициализируем модель одной строкой
    model = ModelFactory.initialize_qwen_model(model_name="Qwen2.5-VL-7B-Instruct")

    # отвечаем на вопрос о по одной картинке
    image_path = "example_docs/snils/8.jpg"
    question = """
Подано изображение паспорта Российской Федерации.
Пожалуйста, извлеките информацию и представьте её в виде структурированного JSON-объекта с указанными полями.

Поля для извлечения:
- "country": Страна (например, "Российская Федерация")
- "type": "СТРАХОВОЕ СВИДЕТЕЛЬСТВО ОБЯЗАТЕЛЬНОГО ПЕНСИОННОГО СТРАХОВАНИЯ",
- "snils_number": "номер документа (в формате XXX-XXX-XXX XX)"
- "fio": Полное имя владельца (Фамилия Имя Отчество)
- "date_of_birth": Дата рождения (в формате DD.MM.YYYY)
- "place_of_birth": Место рождения
- "gender": Пол ("Муж." или "Жен.")
- "registration_date": Дата регистрации (в формате DD.MM.YYYY)

JSON-структура:
{
  "country": "",
  "type": "",
  "snils_number": "",
  "fio": "",
  "date_of_birth": "",
  "place_of_birth": "",
  "gender": "",
  "registration_date": ""
}
"""
    model_answer = model.predict_on_image(image=image_path, prompt=question)
    print(model_answer)
    
    subprocess.run(["nvidia-smi"])

    print(model_answer)
