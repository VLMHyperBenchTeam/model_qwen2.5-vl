import subprocess

from model_interface.model_factory import ModelFactory

if __name__ == "__main__":
    # Инициализируем модель одной строкой
    model = ModelFactory.initialize_qwen_model(
        model_name="Qwen2.5-VL-7B-Instruct",
        cache_dir="model_cache",
        device_map="cuda:0"
    )

    # отвечаем на вопрос о по одной картинке
    image_path = "example_docs/1.png"
    question = "Опиши документ."
    model_answer = model.predict_on_image(image=image_path, prompt=question)
    print(model_answer)
    
    subprocess.run(["nvidia-smi"])

    print(model_answer)
