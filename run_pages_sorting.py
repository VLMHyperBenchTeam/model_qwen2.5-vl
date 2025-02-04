import os
import subprocess

from model_interface.model_factory import ModelFactory


if __name__ == "__main__":

    cache_directory = "model_cache"

    # Сохраняем модели Qwen2-VL в примонтированную папку
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_directory = os.path.join(script_dir, cache_directory)

    # Имена моделей и семейство моделей
    model_name_1 = "Qwen2.5-VL-7B-Instruct"
    model_family = "Qwen2.5-VL"

    # Инфо о том где взять класс для семейства моделей
    package = "model_qwen2_5_vl"
    module = "models"
    model_class = "Qwen2_5_VLModel"
    model_class_path = f"{package}.{module}:{model_class}"

    # Регистрация модели в фабрике
    ModelFactory.register_model(model_family, model_class_path)

    # создаем модель
    model_init_params = {
        "model_name": model_name_1,
        "system_prompt": "",
        "cache_dir": "model_cache",
    }

    model = ModelFactory.get_model(model_family, model_init_params)

    # отвечаем на вопрос о по нескольким картинкам сразу
    image_path1 = "example_docs/sorting/1.jpg"
    image_path2 = "example_docs/sorting/2.jpg"
    image_path3 = "example_docs/sorting/3.jpg"
    image_path4 = "example_docs/sorting/4.jpg"
    
    images=[image_path1, image_path2, image_path3, image_path4]
    
    # question = (f"""Перед вами {len(images)} изображений страниц одного типа документа, которые находятся в хаотичном порядке.
    #             Ваша задача — определить правильную последовательность страниц.
    #             Внимательно проанализируйте содержимое каждой страницы, включая текст, нумерацию, логическую структуру и визуальные элементы, чтобы восстановить правильный порядок.
    #             В ответе укажите только порядок страниц в виде цифр через запятую, например: 1,2,3,4.""")
    # question = (f"""Перед вами {len(images)} изображений страниц одного типа документа, которые находятся в хаотичном порядке.
    #             Анализируя содержимое предоставленных страниц документа, определите логический порядок страниц и выведите их в виде цифр через запятую.
    #             Страницы содержат различные разделы договора о беспроцентном займе, включая условия займа, порядок передачи и возврата суммы займа, ответственность сторон, форс-мажорные обстоятельства, разрешение споров, изменения и досрочное расторжение договора, а также заключительные положения.
    #             Пожалуйста, проанализируйте текст на каждой странице и укажите правильный порядок страниц.""")
    question = (f"""Перед вами {len(images)} изображений страниц одного типа документа, которые находятся в хаотичном порядке.
            Анализируя содержимое предоставленных страниц документа, определите логический порядок страниц и выведите их в виде цифр через запятую.
            Страницы содержат различные разделы договора о беспроцентном займе, включая условия займа, порядок передачи и возврата суммы займа, ответственность сторон, форс-мажорные обстоятельства, разрешение споров, изменения и досрочное расторжение договора, а также заключительные положения.
            Пожалуйста, проанализируйте текст на каждой странице и укажите правильный порядок только в виде порядка страниц через запятую.""")
    
    print(question)
    
    model_answer = model.predict_on_images(
        images=images, question=question
    )
    
    subprocess.run(["nvidia-smi"])

    print(model_answer)
