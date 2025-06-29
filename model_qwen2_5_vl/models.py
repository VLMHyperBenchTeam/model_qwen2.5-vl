"""models.py
Модуль содержит реализацию `Qwen2_5_VLModel` — обёртки вокруг мультимодальной
инструкционной модели Qwen-2.5-VL. Класс предоставляет удобные методы для
инференса по одному или нескольким изображениям, принимая как пути к файлам,
объекты PIL/PyTorch, так и URL.

Google-style docstrings на русском языке используются для единообразной
документации API.
"""

from typing import Any, List, Optional

import torch
from model_interface.model_interface import ModelInterface
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


class Qwen2_5_VLModel(ModelInterface):
    """Инференс-класс для моделей семейства Qwen-2.5-VL.

    Класс инкапсулирует логику загрузки весов, подготовку входных данных и
    генерацию ответов. Предоставляет методы для работы как с одним, так и с
    несколькими изображениями.
    """

    def __init__(
        self,
        model_name: str = "Qwen2.5-VL-3B-Instruct",
        system_prompt: str = "",
        cache_dir: str = "model_cache",
        device_map: Optional[Any] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
    ) -> None:
        """Инициализирует модель.

        Args:
            model_name (str): Полное имя модели в репозитории HuggingFace.
            system_prompt (str): Системный промпт, добавляемый ко всем запросам.
            cache_dir (str): Директория для кэширования весов/токенизатора.
            device_map (Any): Карта устройств (например, ``"cuda:0"`` или
                ``"auto"``). Если ``None``, значение ``"auto"`` будет выбрано
                автоматически.
            min_pixels (int): Минимальное количество пикселей в изображении.
            max_pixels (int): Максимальное количество пикселей в изображении.
        """
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.cache_dir = cache_dir
        self.framework = "Hugging_Face"

        # Допустим разработчик хочет изменить ограничения на размер изображения.
        # Позволяем переопределить значения через параметры конструктора.
        self.min_pixels = min_pixels if min_pixels is not None else 256 * 28 * 28
        self.max_pixels = max_pixels if max_pixels is not None else 1536 * 28 * 28  # 1280 * 28 * 28
        
        if device_map is None:
            device_map="auto"

        # Проверяем доступность flash_attn без импорта модуля (избегаем F401)
        import importlib.util  # локальный импорт, чтобы не тянуть в глобалы

        if importlib.util.find_spec("flash_attn") is not None:
            attn_implementation = "flash_attention_2"
            print("INFO: Используется FlashAttention2 для оптимизации производительности")
        else:
            attn_implementation = "eager"
            print("WARNING: flash_attn не установлен, используется стандартная реализация внимания")

        # default: Load the model on the available device(s)
        model_path = f"Qwen/{model_name}"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            attn_implementation=attn_implementation,
            cache_dir=self.cache_dir,
        )

        # default processor
        self.processor = AutoProcessor.from_pretrained(model_path, cache_dir=cache_dir)

    def get_message(self, image: Any, prompt: str) -> dict:
        """Формирует сообщение в формате Qwen-VL.

        Args:
            image (Any): Изображение — путь к файлу, URL либо готовый объект
                (PIL.Image, ``torch.Tensor`` и т.д.).
            prompt (str): Текстовый промпт, задаваемый модели.

        Returns:
            dict: Словарь, соответствующий ожидаемому формату чата Qwen.
        """
        message = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                },
                {"type": "text", "text": prompt},
            ],
        }

        return message

    # ------------------------------------------------------------------
    # Внутренние сервисные методы
    # ------------------------------------------------------------------

    def _generate_answer(self, messages: List[dict]) -> str:
        """Запускает пайплайн инференса для переданных сообщений.

        Args:
            messages (List[dict]): Список сообщений в формате Qwen-VL chat.

        Returns:
            str: Сгенерированный моделью ответ.
        """
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # process_vision_info может возвращать 2 или 3 элемента в зависимости от версии
        vision_info = process_vision_info(messages)  # type: ignore
        if isinstance(vision_info, tuple) and len(vision_info) == 3:  # type: ignore[arg-type]
            image_inputs, video_inputs, _ = vision_info  # compat с более старыми версиями, где возвращается 3 значения
        else:
            image_inputs, video_inputs = vision_info
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        # Приводим к list для избежания ошибок статического анализа ("Never is not iterable")
        in_ids_list = list(inputs.input_ids)  # type: ignore[arg-type]
        gen_ids_list = list(generated_ids)  # type: ignore[arg-type]

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(in_ids_list, gen_ids_list, strict=False)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def get_messages(self, images: List[Any], prompt: str) -> List[dict]:
        """Формирует список сообщений (один элемент) для нескольких изображений.

        Args:
            images (List[Any]): Коллекция изображений (пути, URL или объекты).
            prompt (str): Текстовый промпт.

        Returns:
            List[dict]: Список сообщений, совместимых с Qwen-VL chat.
        """
        return [
            {
                "role": "user",
                "content": [
                    *[
                        {
                            "type": "image",
                            "image": img,
                            "min_pixels": self.min_pixels,
                            "max_pixels": self.max_pixels,
                        }
                        for img in images
                    ],
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    # ------------------------------------------------------------------
    # Публичные методы инференса
    # ------------------------------------------------------------------

    def predict_on_image(self, image: Any, prompt: str) -> str:
        """Генерирует ответ по одному изображению.

        Args:
            image (Any): Изображение или ссылка/путь к нему.
            prompt (str): Промпт, описывающий требуемый ответ.

        Returns:
            str: Сгенерированный текстовый ответ модели.
        """
        messages = [self.get_message(image, prompt)]
        return self._generate_answer(messages)

    def predict_on_images(self, images: List[Any], prompt: str) -> str:
        """Генерирует ответ по списку изображений.

        Args:
            images (List[Any]): Список изображений (пути, URL или объекты).
            prompt (str): Промпт, задаваемый модели.

        Returns:
            str: Сгенерированный текстовый ответ модели.
        """
        return self._generate_answer(self.get_messages(images, prompt))


# Автоматическая регистрация модели в ModelFactory при импорте
from model_interface.model_factory import ModelFactory

ModelFactory.register_model("Qwen2.5-VL", "model_qwen2_5_vl.models:Qwen2_5_VLModel")
