import os

from model_utils import Qwen2VL_model

if __name__ == "__main__":
    
    cache_directory = "model_cache"
    
    # Сохраняем модели Qwen2-VL в примонтированную папку
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_directory = os.path.join(script_dir, cache_directory)
    
    # Инициализация модели
    model_name= "Qwen2-VL-2B-Instruct-GPTQ-Int4"

    image_path = "example_docs/schet_na_oplatu.png"
    question = "Пожалуйста собери следующую информацию с документа:покупатель,\r\nИНН покупателя,\r\nКПП покупателя,\r\nтелефон покупателя,\r\nпоставщик,\r\nИНН поставщика,\r\nБИК поставщика,\r\nКор. счет поставщика,\r\nР/с поставщика,\r\nдата документа,\r\nномер счета документа,\r\nПеречисли каждый купленный товар  (наименование, количество, цена за штуку)\r\nкакую сумму нужно заплатить за все товары\r\nв какой валюте платим,\r\nвзнос НДС.\r\nВерни ответ в виде json файла с полями и ответами на них."
    
    # создаем модель
    model = Qwen2VL_model(cache_directory, model_name)
    
    # отвечаем на вопрос о по одной картинке
    model_answer = model.predict_on_image(image=image_path, question=question)
    
    # отвечаем на вопрос о по нескольким картинкам сразу (пока не реализован)
    # model_answer = model.predict_on_images(images=[image_path1, image_path2], question=question)
    
    print(model_answer)