import os

print("正在生成專案檔案...")

# 1. 定義 requirements.txt 內容
requirements_content = """fastapi
uvicorn
huggingface_hub
python-dotenv
requests"""

# 2. 定義 main.py 內容 (修復版: 使用 requests 繞過付費牆)
main_content = r'''from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient
import os
import requests
import time
import io

# 初始化 FastAPI
app = FastAPI(title="Leapcell AI Station")

# 允許跨域 (CORS)
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

# 圖片模型 (Direct API)
# 使用 requests 直接調用 API 可避免被路由到付費節點 (如 fal-ai)
IMAGE_MODEL_ID = "stabilityai/stable-diffusion-3.5-large"
IMAGE_API_URL = f"https://api-inference.huggingface.co/models/{IMAGE_MODEL_ID}"

# 初始化文字客戶端
client = InferenceClient(token=HF_TOKEN)

@app.get("/", response_class=HTMLResponse)
def read_root():
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
            .status { font-size: 0.8em; color: #888; margin-top: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎨 AI 圖片生成 (Fix 402 Error)</h1>
            <p class="status">Model: stabilityai/stable-diffusion-3.5-large</p>
            
            <input type="text" id="prompt" placeholder="輸入提示詞 (例如: Cyberpunk city, neon lights)" value="A futuristic city with flying cars, high quality, 8k">
            <button onclick="generateImage()">生成圖片 (Generate)</button>
            
            <p id="loading" class="loading">正在請求 HF 免費 API... (首次啟動可能需 30 秒)</p>
            <p id="error" style="color: red; display: none;"></p>
            <img id="result-img" alt="Generated Image" />
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
                    const response = await fetch(`/api/generate-image?prompt=${encodeURIComponent(prompt)}`);
                    if (!response.ok) {
                        const errText = await response.text();
                        throw new Error(errText);
                    }
                    const blob = await response.blob();
                    img.src = URL.createObjectURL(blob);
                    img.style.display = 'block';
                } catch (e) {
                    error.innerText = "錯誤: " + e.message;
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
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="Server Error: Missing HF Token")

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt}

    # 重試邏輯：處理模型載入 (503)
    max_retries = 5
    for attempt in range(max_retries):
        try:
            print(f"Requesting HF API (Attempt {attempt+1})...")
            response = requests.post(IMAGE_API_URL, headers=headers, json=payload)
            
            if response.status_code == 200:
                # 成功
                return Response(content=response.content, media_type="image/png")
            
            elif response.status_code == 503:
                # 模型載入中
                error_data = response.json()
                estimated_time = error_data.get("estimated_time", 10)
                print(f"Model loading, waiting {estimated_time}s...")
                time.sleep(min(estimated_time, 10))
                continue
            
            else:
                # 其他錯誤
                print(f"Error: {response.text}")
                raise HTTPException(status_code=response.status_code, detail=f"HF API Error: {response.text}")

        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")

    raise HTTPException(status_code=503, detail="Model is too busy or taking too long to load. Please try again later.")

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
'''

# 3. 寫入檔案
with open("requirements.txt", "w", encoding="utf-8") as f:
    f.write(requirements_content)
    print("✅ 已建立 requirements.txt")

with open("main.py", "w", encoding="utf-8") as f:
    f.write(main_content)
    print("✅ 已建立 main.py")

print("\n檔案建立完成！請執行以下命令推送：")
print("git add main.py requirements.txt")
print("git commit -m 'Fix 402 payment error'")
print("git push")
