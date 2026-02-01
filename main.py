from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient
import os
import requests
import time
import io

app = FastAPI(title="Leapcell AI Station")

# 允許跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.getenv("HF_TOKEN")

# 文字模型
TEXT_MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"
text_client = InferenceClient(token=HF_TOKEN)

# --- 圖片模型清單 ---
MODEL_MAP = {
    # [通用/穩定]
    "v1-5": "runwayml/stable-diffusion-v1-5",       # 最穩定
    "dreamshaper": "Lykon/dreamshaper-8",           # 高品質

    # [動漫/無審查] (NSFW Friendly)
    "anything-v5": "stablediffusionapi/anything-v5", 
    
    # [藝術/風格化] (NSFW Friendly)
    "openjourney": "prompthero/openjourney",         
    
    # [次世代] (需授權/易失敗)
    "sd3.5": "stabilityai/stable-diffusion-3.5-large",
    "sd3.5-turbo": "stabilityai/stable-diffusion-3.5-large-turbo",

    # [實驗性]
    "flux-nsfw": "Heartsync/Flux-NSFW-uncensored"
}

DEFAULT_MODEL = MODEL_MAP["v1-5"]

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Leapcell AI Gen (Router Fix)</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: 'Segoe UI', sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #1a1a1a; color: #eee; }
            .container { background: #2d2d2d; padding: 30px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
            h1 { color: #fff; margin-bottom: 5px; }
            .subtitle { color: #aaa; margin-bottom: 25px; }
            
            /* Controls */
            .controls { display: flex; flex-direction: column; gap: 15px; margin-bottom: 20px; }
            .row { display: flex; gap: 10px; flex-wrap: wrap; }
            
            select, input { padding: 12px; border: 1px solid #444; border-radius: 8px; font-size: 16px; background: #3d3d3d; color: white; }
            select { flex: 1; min-width: 200px; cursor: pointer; }
            input { flex: 2; min-width: 200px; }
            
            button { padding: 12px 24px; background: #e91e63; color: white; border: none; cursor: pointer; border-radius: 8px; font-weight: bold; transition: 0.2s; font-size: 16px; min-width: 100px; }
            button:hover { background: #c2185b; }
            
            /* Result Area */
            #result-container { min-height: 300px; display: flex; flex-direction: column; align-items: center; justify-content: center; border: 2px dashed #444; border-radius: 8px; margin-top: 20px; background: #222; overflow: hidden; }
            #result-img { max-width: 100%; display: none; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }
            .loading { color: #888; font-style: italic; display: none; }
            
            /* API Panel */
            .api-panel { margin-top: 40px; background: #000; color: #00ff9d; padding: 20px; border-radius: 8px; font-family: 'Consolas', monospace; font-size: 13px; border: 1px solid #333; position: relative; }
            .api-panel h3 { color: #fff; border-bottom: 1px solid #333; padding-bottom: 10px; margin-top: 0; }
            .label { color: #666; display: inline-block; width: 120px; }
            .copy-btn { position: absolute; top: 15px; right: 15px; background: #333; border: 1px solid #555; color: white; padding: 4px 10px; font-size: 11px; cursor: pointer; border-radius: 4px; }
            
            optgroup { color: #ccc; background: #222; font-style: normal; font-weight: bold; }
            option { color: #fff; background: #333; padding: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎨 Leapcell AI Generator</h1>
            <p class="subtitle">Multi-Model Image Generation API</p>
            
            <div class="controls">
                <div class="row">
                    <select id="model-select" onchange="updateApiInfo()">
                        <optgroup label="✨ 推薦 (Recommended)">
                            <option value="v1-5">Stable Diffusion v1.5 (最穩定)</option>
                            <option value="dreamshaper">DreamShaper 8 (高品質)</option>
                        </optgroup>
                        <optgroup label="🔞 動漫/寬鬆 (Uncensored-ish)">
                            <option value="anything-v5">Anything v5 (動漫/無審查)</option>
                            <option value="openjourney">OpenJourney (藝術/寬鬆)</option>
                        </optgroup>
                        <optgroup label="🚀 次世代 (New Gen)">
                            <option value="sd3.5">SD 3.5 Large (需授權)</option>
                            <option value="sd3.5-turbo">SD 3.5 Turbo (快)</option>
                        </optgroup>
                        <optgroup label="🧪 實驗性 (Experimental)">
                            <option value="flux-nsfw">Flux NSFW (易失敗)</option>
                        </optgroup>
                    </select>
                </div>
                <div class="row">
                    <input type="text" id="prompt" placeholder="Enter prompt..." value="1girl, masterpiece, best quality, cyberpunk city background">
                    <button onclick="generateImage()">Generate</button>
                </div>
            </div>
            
            <div id="result-container">
                <p id="placeholder" style="color: #555;">Image will appear here</p>
                <p id="loading" class="loading">⚡ Processing... (Waiting for GPU)</p>
                <p id="error" style="color: #ff5252; display: none; padding: 20px; text-align: center;"></p>
                <img id="result-img" alt="Generated Image" />
            </div>

            <!-- API Panel -->
            <div class="api-panel">
                <h3>🔌 API Integration</h3>
                <button class="copy-btn" onclick="copyApiUrl()">Copy URL</button>
                
                <div><span class="label">Endpoint:</span><span id="api-full-url">Loading...</span></div>
                <div><span class="label">Model ID:</span><span id="current-model-id">...</span></div>
                
                <div style="margin-top: 15px; border-top: 1px dashed #333; padding-top: 10px;">
                    <span class="label" style="display:block; margin-bottom:5px;">Python Example:</span>
                    <code id="code-example" style="white-space: pre-wrap; color: #a5d6ff;">Loading...</code>
                </div>
            </div>
        </div>

        <script>
            const MODEL_IDS = {
                "v1-5": "runwayml/stable-diffusion-v1-5",
                "dreamshaper": "Lykon/dreamshaper-8",
                "anything-v5": "stablediffusionapi/anything-v5",
                "openjourney": "prompthero/openjourney",
                "sd3.5": "stabilityai/stable-diffusion-3.5-large",
                "sd3.5-turbo": "stabilityai/stable-diffusion-3.5-large-turbo",
                "flux-nsfw": "Heartsync/Flux-NSFW-uncensored"
            };

            window.onload = updateApiInfo;

            function updateApiInfo() {
                const modelKey = document.getElementById('model-select').value;
                const host = window.location.origin;
                const fullUrl = `${host}/api/generate-image`;
                
                document.getElementById('api-full-url').innerText = fullUrl;
                document.getElementById('current-model-id').innerText = MODEL_IDS[modelKey];
                
                const code = `import requests\n\nresponse = requests.get(\n    "${fullUrl}",\n    params={"prompt": "1girl, cat ears", "model": "${modelKey}"}\n)\n\nwith open("out.png", "wb") as f:\n    f.write(response.content)`;
                document.getElementById('code-example').innerText = code;
            }

            function copyApiUrl() {
                navigator.clipboard.writeText(document.getElementById('api-full-url').innerText);
                alert("URL Copied!");
            }

            async function generateImage() {
                const prompt = document.getElementById('prompt').value;
                const modelKey = document.getElementById('model-select').value;
                const img = document.getElementById('result-img');
                const loading = document.getElementById('loading');
                const error = document.getElementById('error');
                const placeholder = document.getElementById('placeholder');
                
                if(!prompt) return alert("Please enter a prompt");

                img.style.display = 'none';
                placeholder.style.display = 'none';
                error.style.display = 'none';
                loading.style.display = 'block';

                try {
                    const response = await fetch(`/api/generate-image?model=${modelKey}&prompt=${encodeURIComponent(prompt)}`);
                    if (!response.ok) throw new Error(await response.text());
                    
                    const blob = await response.blob();
                    img.src = URL.createObjectURL(blob);
                    img.style.display = 'block';
                } catch (e) {
                    error.innerText = "Error: " + e.message;
                    error.style.display = 'block';
                } finally {
                    loading.style.display = 'none';
                }
            }
        </script>
    </body>
    </html>
    """

@app.get("/api/info")
async def get_api_info():
    has_token = bool(HF_TOKEN)
    return JSONResponse({
        "status": "online",
        "models": list(MODEL_MAP.keys()),
        "token_set": has_token
    })

@app.get("/api/generate-image")
async def generate_image(prompt: str, model: str = "v1-5"):
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="Server Error: Missing HF Token")

    target_model_id = MODEL_MAP.get(model, DEFAULT_MODEL)
    
    # 使用唯一的合法 Router URL
    api_url = f"https://router.huggingface.co/hf-inference/models/{target_model_id}"
    
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt}

    print(f"Requesting: {model} ({target_model_id}) via Router")

    # 重試邏輯
    for attempt in range(3):
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=50)
            
            if response.status_code == 200:
                return Response(content=response.content, media_type="image/png")
            
            elif response.status_code == 503:
                # Cold start (模型載入中)
                print("Model loading (503)... waiting")
                time.sleep(8)
                continue 
            
            # 處理各種失敗情況 (404 Not Found, 402 Payment, 410 Gone, 400 Bad Request)
            elif response.status_code in [404, 402, 400, 410, 401, 403]:
                print(f"Model {model} failed with {response.status_code}. Msg: {response.text[:100]}")
                
                # 如果這不是 v1-5，嘗試切換到 v1-5
                if target_model_id != DEFAULT_MODEL:
                    print("Falling back to v1-5...")
                    fallback_url = f"https://router.huggingface.co/hf-inference/models/{DEFAULT_MODEL}"
                    fallback_resp = requests.post(fallback_url, headers=headers, json=payload, timeout=50)
                    
                    if fallback_resp.status_code == 200:
                        return Response(content=fallback_resp.content, media_type="image/png")
                    else:
                        print(f"Fallback v1-5 failed: {fallback_resp.status_code}")
                        
                        # 絕境求生: 嘗試 SD 2.1
                        if fallback_resp.status_code in [404, 410, 400]:
                             print("Trying final fallback: SD 2.1")
                             fallback_url_2 = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-2-1"
                             fallback_resp_2 = requests.post(fallback_url_2, headers=headers, json=payload, timeout=50)
                             if fallback_resp_2.status_code == 200:
                                 return Response(content=fallback_resp_2.content, media_type="image/png")

            # 如果沒有成功 return，且沒有觸發 fallback 或 fallback 失敗
            if attempt == 2:
                # 拋出最後一次的錯誤訊息
                raise HTTPException(status_code=response.status_code, detail=f"HF Error: {response.text}")

        except Exception as e:
            print(f"Exception: {str(e)}")
            if attempt == 2:
                raise HTTPException(status_code=500, detail=str(e))
            time.sleep(2)

    raise HTTPException(status_code=503, detail="Service busy or model unavailable.")

# Text Chat API
@app.post("/api/chat")
async def generate_chat(prompt: str):
    try:
        messages = [{"role": "user", "content": prompt}]
        response = text_client.chat_completion(messages=messages, model=TEXT_MODEL_ID, max_tokens=500)
        return {"result": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
