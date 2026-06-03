# 本地 VLM Demo

這是一個使用 `Qwen/Qwen2-VL-2B-Instruct` 的本地視覺語言模型範例。

## 功能

- 上傳圖片
- 輸入問題
- 在本機 GPU 上直接推理

## 模型

- `Qwen/Qwen2-VL-2B-Instruct`
- Hugging Face 模型頁：https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
- 授權：Apache 2.0

## 安裝

```bash
pip install -r requirements.txt
```

如果 `torch` 安裝失敗，請先到官方網站安裝對應你平台的版本：

https://pytorch.org/get-started/locally/

## 執行

```bash
streamlit run app.py
```

## 第一次執行

- 第一次會從 Hugging Face 下載模型權重
- 之後會使用本機快取
- 建議使用 GPU
- 如果遇到顯存不足，請先改小圖片或關閉其他 GPU 程式

## 切換模型

可以透過 `VLM_MODEL` 環境變數指定模型：

```powershell
$env:VLM_MODEL="Qwen/Qwen2-VL-2B-Instruct"
```
