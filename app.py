import os
from typing import Tuple

import streamlit as st
from PIL import Image


DEFAULT_MODEL = os.getenv("VLM_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28


st.set_page_config(
    page_title="本地 VLM Demo",
    page_icon="🖼️",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return processor, model


def build_prompt(question: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]


def infer(processor, model, image: Image.Image, question: str) -> str:
    import torch

    # 壓小圖片，降低視覺 token 數量，減少顯存壓力。
    image = image.copy()
    image.thumbnail((1280, 1280))

    messages = build_prompt(question)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    try:
        inputs = inputs.to(model.device)
    except Exception:
        pass

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )

    trimmed_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    decoded = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return decoded[0].strip() if decoded else "模型沒有回傳內容。"


def image_summary(image: Image.Image, uploaded_file_name: str) -> Tuple[str, str]:
    width, height = image.size
    mode = image.mode
    return uploaded_file_name, f"{width} x {height} px, mode={mode}"


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.2rem;
        max-width: 1600px;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    .hero {
        padding: 1.35rem 1.5rem;
        border-radius: 1.1rem;
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #7c3aed 100%);
        color: white;
        box-shadow: 0 14px 32px rgba(15, 23, 42, 0.24);
        margin-bottom: 1.1rem;
    }
    .hero h1 {
        font-size: 2.25rem;
        margin: 0 0 0.2rem 0;
    }
    .hero p {
        margin: 0;
        opacity: 0.92;
        font-size: 1.03rem;
    }
    .panel {
        padding: 1rem 1.1rem;
        border: 1px solid rgba(148, 163, 184, 0.3);
        border-radius: 1rem;
        background: rgba(255, 255, 255, 0.72);
    }
    .small-note {
        font-size: 0.95rem;
        color: #475569;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="hero">
        <h1>本地 VLM Demo</h1>
        <p>使用 Qwen/Qwen2-VL-2B-Instruct 在你的 GPU 上本地推理。</p>
    </div>
    """,
    unsafe_allow_html=True,
)

top_left, top_mid, top_right = st.columns([1.0, 1.6, 1.0], gap="large")

with top_left:
    uploaded_file = st.file_uploader(
        "圖片",
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed",
    )

with top_mid:
    question = st.text_area(
        "問題",
        height=120,
        value="請描述這張圖片的主要內容，並指出關鍵細節。",
        label_visibility="collapsed",
    )

with top_right:
    model_name = st.text_input(
        "模型",
        value=DEFAULT_MODEL,
        label_visibility="collapsed",
    )
    run = st.button("開始分析", type="primary", use_container_width=True, disabled=uploaded_file is None)
    st.caption("第一次執行會從 Hugging Face 下載權重，之後使用本機快取。")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    result_col, preview_col = st.columns([1.3, 0.9], gap="large")

    with preview_col:
        st.image(image, caption="預覽", use_container_width=True)
        name, info = image_summary(image, uploaded_file.name)
        st.markdown(
            f"""
            <div class="panel">
                <div><strong>{name}</strong></div>
                <div class="small-note">{info}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with result_col:
        st.markdown(
            """
            <div class="panel">
                <strong>提示</strong>
                <ul style="margin: 0.6rem 0 0 1.4rem; line-height: 1.7;">
                    <li>若遇到 GPU 記憶體不足，請先改小圖片或關閉其他 GPU 程式。</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if run:
        with st.spinner("正在載入模型並推理..."):
            try:
                processor, model = load_model(model_name)
                answer = infer(processor, model, image, question.strip())
                st.success("完成")
                st.markdown("### 模型回答")
                st.write(answer)
            except Exception as exc:
                message = str(exc).lower()
                if "out of memory" in message or "cuda" in message:
                    st.error(
                        "GPU 記憶體不足。這張圖片或目前設定對顯卡來說太吃資源。"
                        "請改用更小的圖片、先關閉其他 GPU 程式，或改用更小的 VLM / 4-bit 量化。"
                    )
                else:
                    st.error("推理失敗。可能原因包含模型下載、記憶體，或 transformers 版本問題。")
                st.exception(exc)
else:
    st.info("請先上傳圖片開始。")

with st.expander("進階說明", expanded=False):
    st.markdown(
        """
        - 模型：`Qwen/Qwen2-VL-2B-Instruct`
        - 建議使用 GPU。
        - 模型第一次下載後會快取在本機。
        - 你可以用 `VLM_MODEL` 環境變數切換模型。
        """
    )
