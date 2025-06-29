import subprocess
from typing import Any
from model_qwen2_5_vl import initialize_qwen_model

if __name__ == "__main__":
    # Инициализируем модель одной строкой
    model: Any = initialize_qwen_model(model_name="Qwen2.5-VL-7B-Instruct")

    # отвечаем на вопрос о по нескольким картинкам сразу
    image_path1 = "example_docs/1.png"
    image_path2 = "example_docs/2.png"
    
    images=[image_path1, image_path2]
    
    prompt = f"Количество поданных страниц документов - {len(images)}. Ответьте на вопрос: Кто подписал документ?"
    
    print(prompt)
    
    model_answer = model.predict_on_images(
        images=images, prompt=prompt
    )
    
    subprocess.run(["nvidia-smi"])

    print(model_answer)
