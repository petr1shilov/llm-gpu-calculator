# LLM GPU Calculator

Интерактивный калькулятор на **Streamlit** для оценки требований к GPU при работе с большими языковыми моделями (LLM).  
Позволяет рассчитать:

- 📦 сколько VRAM нужно для запуска модели,
- 🚀 примерную скорость генерации токенов (TPS),
- 💵 стоимость одного токена,
- 📈 эффективность (TPS/варюта),
- 🤖 рекомендации по выбору GPU или multi-GPU, если одной не хватает.

---

## Запуск локально

1. **Клонируйте репозиторий**
```bash
git clone [https://github.com/petr1shilov/llm-gpu-calculator](https://github.com/petr1shilov/llm-gpu-calculator.git)
cd llm-gpu-calculator
```
2.**Запустите приложение**
```bash
streamlit run Новая llm_gpu_app_full_v4.py
```

3. **Структура проекта**
```bash
llm-gpu-calculator/
│
├── app.py            # Основное Streamlit-приложение
├── LICENSE.txt       # Лицензия (Proprietary – All Rights Reserved)
├── README.md         # Описание проекта
└── requirements.txt  # Зависимости
```

Proprietary – All Rights Reserved
© 2025 Шилов Петр. Все права защищены.
Распространение, публикация, перепродажа, публичный хостинг (SaaS) и коммерческое использование запрещены без письменного разрешения автора.
См. файл LICENSE.txt для подробных условий.
Контакты для получения коммерческой лицензии:
[shilovpetr64@gmail.com]
