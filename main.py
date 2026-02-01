from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient
import os
import io

# 初始化 FastAPI
app = FastAPI(title="Leapcell AI Station")

# 允許跨域 (CORS) - 讓其他網站或 APK 能調用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 設定 ---
HF_TOKEN = os.getenv("HF_TOKEN")

# 文字模型 (Chat)
TEXT_MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"
# 圖片模型 (Image) - FLUX.1-schnell 速度快且質量高，適合免費版
IMAGE_MODEL_ID = "black-forest-labs/FLUX.1-schnell"
# 備用圖片模型: "stabilityai/stable-diffusion-3.5-large"

# 初始化客戶端
client = InferenceClient(token=HF_TOKEN)

@app.get("/", response_class=HTMLResponse)
def read_root():
    """
    這是一個簡單的測試前端，包含文字聊天和圖片生成功能。
    您可以將這段 HTML 替換為您的 APK 下載頁。
    """
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Leapcell AI Station</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f0f2f5; }
            .container { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            input, button { padding: 10px; margin: 5px 0; width: 100%; box-sizing: border-box; }
            button { background: #007bff; color: white; border: none; cursor: pointer; border-radius: 5px; }
            button:hover { background: #0056b3; }
            #result-img { max-width: 100%; margin-top: 10px; border-radius: 5px; display: none; }
            .loading { color: #666; font-style: italic; display: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎨 AI 圖片生成試驗</h1>
            <p>使用模型: FLUX.1-schnell (Free Tier)</p>
            
            <input type="text" id="prompt" placeholder="輸入提示詞 (例如: A futuristic city in cyberpunk style)" value="A cute robot holding a flower, high quality">
            <button onclick="generateImage()">生成圖片 (Generate)</button>
            
            <p id="loading" class="loading">正在生成中，請稍候... (約需 5-10 秒)</p>
            <p id="error" style="color: red; display: none;"></p>
            <img id="result-img" alt="Generated Image" />
            
            <hr style="margin: 30px 0;">
            
            <h3>📥 APK 下載</h3>
            <a href="https://github.com/YOUR_USER/YOUR_REPO/releases">前往 GitHub 下載最新 APK</a>
        </div>

        <script>
            async function generateImage() {
                const prompt = document.getElementById('prompt').value;
                const img = document.getElementById('result-img');
                const loading = document.getElementById('loading');
                const error = document.getElementById('error');
                
                if(!prompt) return alert("請輸入提示詞");

                img.style.display = 'none';
                error.style.display = 'none';
                loading.style.display = 'block';

                try {
                    // 調用後端 API
                    const response = await fetch(`/api/generate-image?prompt=${encodeURIComponent(prompt)}`);
                    
                    if (!response.ok) throw new Error(await response.text());
                    
                    // 將二進制圖片數據轉換為 Blob URL 顯示
                    const blob = await response.blob();
                    const url = URL.createObjectURL(blob);
                    
                    img.src = url;
                    img.style.display = 'block';
                } catch (e) {
                    error.innerText = "生成失敗: " + e.message;
                    error.style.display = 'block';
                } finally {
                    loading.style.display = 'none';
                }
            }
        </script>
    </body>
    </html>
    """

@app.get("/api/generate-image")
async def generate_image(prompt: str):
    """
    圖片生成 API
    回傳: 直接回傳 PNG 圖片流 (Binary)
    """
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="Server Error: Missing HF Token")

    try:
        # 調用 Hugging Face 的 text_to_image
        # 回傳的是 PIL.Image 對象
        image = client.text_to_image(prompt, model=IMAGE_MODEL_ID)
        
        # 將 PIL Image 轉為 Bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # 回傳圖片數據流 (Media Type image/png)
        return Response(content=img_byte_arr.getvalue(), media_type="image/png")

    except Exception as e:
        print(f"Error: {e}")
        # 處理常見錯誤 (如 Rate Limit, Model Loading)
        raise HTTPException(status_code=503, detail=f"Image Generation Failed: {str(e)}")

# 文字 API (保留之前的)
@app.post("/api/chat")
async def generate_chat(prompt: str):
    try:
        messages = [{"role": "user", "content": prompt}]
        response = client.chat_completion(messages=messages, model=TEXT_MODEL_ID, max_tokens=500)
        return {"result": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
