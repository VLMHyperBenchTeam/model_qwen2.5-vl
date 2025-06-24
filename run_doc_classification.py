import os
import subprocess

from model_interface.model_factory import ModelFactory


if __name__ == "__main__":
    # Инициализируем модель одной строкой
    model = ModelFactory.initialize_qwen_model(model_name="Qwen2.5-VL-7B-Instruct")

    # отвечаем на вопрос о по нескольким картинкам сразу
    image_path1 = "example_docs/classification/3.jpg"
    image_path2 = "example_docs/classification/5.jpg"
    image_path3 = "example_docs/classification/2.jpg"
    image_path4 = "example_docs/classification/1.jpg"
    image_path5 = "example_docs/classification/7.png"
    image_path6 = "example_docs/classification/6.jpg"
    
    images=[image_path1, image_path2, image_path3, image_path4, image_path5, image_path6]
    
    prompt = (f"""Количество поданных страниц документов - {len(images)}.
                Задача: Определите тип каждого документа на предоставленных изображениях и выведите их в виде последовательности цифр, где каждая цифра соответствует определенному типу документа. Ответ должен содержать только порядок цифр, без дополнительного текста.
                Типы документов:
                1 - old_tins: свидетельство о постановке на учет физического лица (документ желтого цвета).
                2 - new_tins: свидетельство о постановке на учет физического лица
                3 - interest_free_loan_agreement: договор беспроцентном займа
                4 - snils: страховое свидетельство обязательного пенсионного страхования (документ зеленого цвета).
                5 - invoice: счет фактура
                6 - passport: паспорт России
                Пример ответа: 2,4,5,1,3
                Пожалуйста, предоставьте ответ в указанном формате.""")
    
    print(prompt)
    
    model_answer = model.predict_on_images(
        images=images, prompt=prompt
    )
    
    subprocess.run(["nvidia-smi"])

    print(model_answer)
