import os
from typing import Tuple

import streamlit as st
from PIL import Image


DEFAULT_MODEL = os.getenv("VLM_MODEL", "Qwen/Qwen2-VL-2B-Instruct")


st.set_page_config(
    page_title="Local VLM Demo",
    page_icon="🖼️",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
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
            max_new_tokens=128,
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
    return f"{uploaded_file_name}", f"{width} x {height} px, mode={mode}"


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }
    .hero {
        padding: 1.5rem 1.75rem;
        border-radius: 1.25rem;
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #7c3aed 100%);
        color: white;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.28);
    }
    .hero h1 {
        font-size: 2.25rem;
        margin-bottom: 0.25rem;
    }
    .hero p {
        margin: 0;
        opacity: 0.92;
        font-size: 1.02rem;
    }
    .panel {
        padding: 1rem 1.1rem;
        border: 1px solid rgba(148, 163, 184, 0.3);
        border-radius: 1rem;
        background: rgba(255, 255, 255, 0.72);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="hero">
        <h1>Local VLM Demo</h1>
        <p>使用本地可執行的 Qwen2-VL-2B-Instruct，直接上傳圖片並提問。</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.subheader("1. 上傳圖片")
    uploaded_file = st.file_uploader(
        "支援 PNG、JPG、JPEG、WEBP",
        type=["png", "jpg", "jpeg", "webp"],
    )

    image = None
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="預覽圖片", use_container_width=True)

    st.subheader("2. 輸入問題")
    question = st.text_area(
        "例如：這張圖片裡有哪些物件？請幫我描述場景。",
        height=120,
        value="請描述這張圖片的主要內容，並指出你看到的重點。",
    )

    st.subheader("3. 執行設定")
    model_name = st.text_input("模型名稱", value=DEFAULT_MODEL)
    st.caption("第一次執行會從 Hugging Face 下載模型，之後會使用本機快取。")

    run = st.button(
        "開始分析",
        type="primary",
        use_container_width=True,
        disabled=image is None,
    )

with right:
    st.subheader("使用說明")
    st.markdown(
        """
        <div class="panel">
        <ul>
        <li>這個版本改成本地 VLM，不需要 OpenAI API key。</li>
        <li>模型使用 <code>Qwen/Qwen2-VL-2B-Instruct</code>。</li>
        <li>任務主要做圖片描述、情境問答、基本 OCR 與視覺理解。</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("建議問題")
    st.code(
        "1. 這張圖裡有幾個人？\n"
        "2. 圖中的場景看起來像什麼地方？\n"
        "3. 請整理圖片中可見的文字與物件。",
        language="text",
    )

    if image is not None:
        st.subheader("圖片資訊")
        file_name, info = image_summary(image, uploaded_file.name if uploaded_file else "image")
        st.write(file_name)
        st.write(info)

if image is not None and run:
    with st.spinner("載入模型與推理中，第一次會比較久..."):
        try:
            processor, model = load_model(model_name)
            answer = infer(processor, model, image, question.strip())
            st.success("完成")
            st.markdown("### 模型回答")
            st.write(answer)
        except Exception as exc:
            st.error(
                "推理失敗。可能原因包含：模型下載失敗、記憶體不足、或 transformers 版本太舊。"
            )
            st.exception(exc)
