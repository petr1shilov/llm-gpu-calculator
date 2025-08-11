import streamlit as st
import math

GPUS = {
    "T4": {"vram": 16, "fp16_tflops": 65, "supports_fp8": False, "base_tps_8b_fp16": 20},
    "A10": {"vram": 24, "fp16_tflops": 125, "supports_fp8": True, "base_tps_8b_fp16": 42},
    "A100_40G": {"vram": 40, "fp16_tflops": 312, "supports_fp8": True, "base_tps_8b_fp16": 130},
    "A100_80G": {"vram": 80, "fp16_tflops": 312, "supports_fp8": True, "base_tps_8b_fp16": 135},
    "H100_80G": {"vram": 80, "fp16_tflops": 1517, "supports_fp8": True, "base_tps_8b_fp16": 200},
    "L40": {"vram": 48, "fp16_tflops": 362, "supports_fp8": True, "base_tps_8b_fp16": 150},
    "RTX_4090": {"vram": 24, "fp16_tflops": 330, "supports_fp8": False, "base_tps_8b_fp16": 100},
    "RTX_3090": {"vram": 24, "fp16_tflops": 285, "supports_fp8": False, "base_tps_8b_fp16": 80},
    "L4": {"vram": 24, "fp16_tflops": 120, "supports_fp8": True, "base_tps_8b_fp16": 40}
}

PRECISION_BYTES = {
    "FP32": 4,
    "FP16": 2,
    "FP8": 1,
    "INT8": 1,
    "INT4": 0.5
}

def calculate_memory(parameters_b, precision_bytes, batch_size, context_length, num_layers):
    model_memory = (parameters_b * 1e9 * precision_bytes) / (1024 ** 3)
    kv_cache_memory = (batch_size * context_length * num_layers * 2 * precision_bytes) / (1024 ** 3)
    overhead = model_memory * 0.2
    total_memory = model_memory + kv_cache_memory + overhead
    return total_memory, model_memory, kv_cache_memory

def estimate_token_cost(price_per_hour, tps):
    if price_per_hour == 0 or tps == 0:
        return 0.0
    return round(price_per_hour / (tps * 3600), 10)

def estimate_efficiency(tps, price_per_hour):
    if price_per_hour == 0:
        return 0.0
    return round(tps / price_per_hour, 2)

def estimate_tps_for_gpu(gpu, parameters_b, precision, base_tps_8b_fp16):
    scale_factor = 8 / parameters_b
    tps = base_tps_8b_fp16 * scale_factor
    if precision in ["FP8", "INT8"]:
        tps *= 1.5
    elif precision == "INT4":
        tps *= 2.0
    elif precision == "FP32":
        tps *= 0.5
    return tps

def recommend_gpu(total_memory, precision, required_tps, parameters_b):
    recommendations = []
    for gpu, specs in GPUS.items():
        if total_memory > specs["vram"]:
            continue
        if precision == "FP8" and not specs["supports_fp8"]:
            continue
        estimated_tps = estimate_tps_for_gpu(gpu, parameters_b, precision, specs["base_tps_8b_fp16"])
        if estimated_tps >= required_tps:
            recommendations.append((gpu, estimated_tps, specs["vram"]))
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations

def suggest_multi_gpu(total_memory, min_tps, precision, parameters_b):
    options = []
    for gpu, specs in GPUS.items():
        if precision == "FP8" and not specs["supports_fp8"]:
            continue
        tps = estimate_tps_for_gpu(gpu, parameters_b, precision, specs["base_tps_8b_fp16"])
        gpus_needed = math.ceil(total_memory / specs["vram"])
        if tps * gpus_needed >= min_tps:
            options.append((gpu, gpus_needed, tps * gpus_needed))
    return options

def estimate_latency(max_traffic, mean_tokens_per_query, tps):
    return (max_traffic * mean_tokens_per_query) / (tps * 8)

st.set_page_config(page_title="GPU калькулятор для LLM", page_icon="🧠")

st.title("📊 Калькулятор оценки GPU для LLM")
st.markdown("""
Этот калькулятор помогает оценить:

- 📦 сколько VRAM нужно для запуска модели,
- 🚀 примерную скорость генерации токенов (TPS),
- 💵 стоимость одного токена,
- 📈 эффективность (TPS/варюта),
- 🤖 рекомендации по выбору GPU или multi-GPU, если одной не хватает.
""")

with st.sidebar:
    st.header("🧠 Параметры модели")
    st.markdown("""
    - **Параметры модели** — в миллиардах (например, 7 для LLaMA-3 7B).
    - **Batch size** — количество одновременно обрабатываемых запросов.
    - **Sequence length** — длина последовательности в токенах.
    - **Hidden dimension** — размер скрытого слоя модели.
    - **Количество слоёв** — количество блоков трансформера.
    - **Точность** — влияет на объём памяти и TPS.
    """)
    model_params_billion = st.number_input("Параметры модели (в миллиардах)", value=7, step=1)
    batch_size = st.number_input("Batch size", value=4, step=1)
    seq_len = st.number_input("Sequence length", value=1024, step=128)
    hidden_dim = st.number_input("Hidden dimension", value=4096, step=256)
    n_layers = st.number_input("Количество слоёв", value=32, step=1)
    precision = st.selectbox("Точность", options=["FP32", "FP16", "FP8", "INT8", "INT4"], index=1)
    dtype_bytes = PRECISION_BYTES[precision]

    st.header("🛠 Парметры для примерных тестов")
    st.markdown("""
    - *Максимальный трафик за раз* — пиковый загруз сайта.
    - *Среднее количество такенов в одном запросе* — количество токенов в одном вопросе(в стреднем).
    - *maximum latency* - Максимальное время ответа модели (сек)
    """)
    max_traffic = st.number_input("Максимальный трафик за раз", value=500, step=10)
    mean_tokens_per_query = st.number_input("Среднее количество такенов в одном запросе", value=100, step=10)
    max_latency = st.number_input("maximum latency", value=7, step=1)

    st.header("💻 Параметры GPU")
    currency = st.selectbox("Валюта ввода", ["₽", "$"], index=0)
    if currency == "$":
        usd_to_rub = st.number_input("Курс USD→RUB", value=90.0, step=1.0)

    st.markdown("""
    - *TFLOPS GPU* — пиковая вычислительная мощность карты (например, A10 ≈ 150 TFLOPS).
    - *Стоимость аренды* — цена аренды GPU за час (в выбранной валюте).
    """)
    base_tflops = st.number_input("TFLOPS GPU", value=150.0, step=10.0)
    if currency == "₽":
        price_per_hour = st.number_input("Стоимость аренды GPU (₽/час)", value=90.0, step=1.0)
    else:
        price_per_hour_usd = st.number_input("Стоимость аренды GPU ($/час)", value=0.9, step=0.1)
        # usd_to_rub = st.number_input("Курс USD→RUB", value=90.0, step=1.0)
        price_per_hour = price_per_hour_usd * usd_to_rub
        st.caption(f"Пересчёт: {price_per_hour_usd} $/час × {usd_to_rub} = {price_per_hour:.2f} ₽/час")
    required_tps = st.number_input("Желаемый TPS", value=50.0, step=1.0)

if st.button("Рассчитать"):
    total_mem, model_mem, kv_mem = calculate_memory(model_params_billion, dtype_bytes, batch_size, seq_len, n_layers)
    tps = estimate_tps_for_gpu("custom", model_params_billion, precision, base_tflops)
    cost = estimate_token_cost(price_per_hour, tps)
    efficiency = estimate_efficiency(tps, price_per_hour)
    latency = estimate_latency(max_traffic, mean_tokens_per_query, tps)

    st.subheader("📈 Результаты")
    st.metric("Вес модели (GB)", round(model_mem, 2))
    st.metric("Объём KV-кэша (GB)", round(kv_mem, 2))
    st.metric("Общий объём VRAM (GB)", round(total_mem, 2))
    st.metric("Оценка TPS", round(tps, 2))
    st.metric("Стоимость токена (₽)", f"{cost:.8f} ₽")
    st.metric("Эффективность (TPS/₽)", efficiency)
    st.metric("Скорость ответа (сек)", round(latency, 2))

    st.subheader("🧠 Рекомендованные GPU")
    recs = recommend_gpu(total_mem, precision, required_tps, model_params_billion)
    if recs:
        for gpu_name, est_tps, gpu_vram in recs:
            st.markdown(f"✅ **{gpu_name}** — TPS ≈ `{round(est_tps)}`, VRAM: `{gpu_vram} GB`")
    else:
        st.warning("❌ Нет подходящего GPU. Ниже предложены multi-GPU варианты.")
        multi_gpu = suggest_multi_gpu(total_mem, required_tps, precision, model_params_billion)
        if multi_gpu:
            st.subheader("💡 Multi-GPU Варианты")
            for gpu_name, num_gpus, total_tps in multi_gpu:
                st.markdown(f"- {num_gpus}× **{gpu_name}** — Общий TPS ≈ `{round(total_tps)}`")
        else:
            st.error("Невозможно достичь цели даже с multi-GPU. Попробуйте уменьшить batch/context или использовать квантование.")
