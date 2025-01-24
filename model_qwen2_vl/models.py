from typing import Any, List

import torch
from model_interface.model_interface import ModelInterface
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


class Qwen2VLModel(ModelInterface):
    def __init__(
        self,
        model_name="Qwen2-VL-2B-Instruct",
        system_prompt="",
        cache_dir="model_cache",
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.cache_dir = cache_dir
        
        # default: Load the model on the available device(s)
        model_path = f"Qwen/{model_path}"
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
            cache_dir=self.cache_dir,
        )

        # default processor
        self.processor = AutoProcessor.from_pretrained(model_path, cache_dir=cache_dir)

    @staticmethod
    def get_message(image, question):
        message = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": question},
            ],
        }

        return message

    def predict_on_image(self, image, question):
        messages = [self.get_message(image, question)]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0]

    # TODO: реализовать метод
    def predict_on_images(self, images: List[Any], question: str) -> str:
        """Реализация метода для предсказания на основе нескольких изображений.

        Args:
            images (List[Any]): Список изображений, на основе которых делается предсказание.
                               Тип элементов списка может быть специфичным для реализации.
            question (str): Промпт-вопрос по изображениям.

        Returns:
            str: Строка с ответом от модели.
        """
        return f"{self.model_name} predict_on_images"
