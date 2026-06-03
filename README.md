# Local VLM Demo

This project runs a local vision-language model with Streamlit.

## Model

- `Qwen/Qwen2-VL-2B-Instruct`
- Hugging Face model card: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
- License: Apache 2.0

## Features

- Upload an image
- Ask a question about the image
- Run inference locally without an API key

## Install

```bash
pip install -r requirements.txt
```

If `torch` installation fails, install the correct wheel for your platform from the official PyTorch site first:

https://pytorch.org/get-started/locally/

## Run

```bash
streamlit run app.py
```

## First Run

- The first launch downloads the model from Hugging Face
- Later launches use the local cache
- GPU is recommended, but CPU fallback is possible and will be slower

## Change the Model

You can override the default model with `VLM_MODEL`:

```powershell
$env:VLM_MODEL="Qwen/Qwen2-VL-2B-Instruct"
```
