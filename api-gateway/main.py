# api-gateway/main.py
from fastapi import FastAPI, Request
from prometheus_fastapi_instrumentator import Instrumentator
import httpx, os, time, ssl

app = FastAPI(title="AI Platform API Gateway")
Instrumentator().instrument(app).expose(app)

VLLM_URL = os.environ.get("VLLM_URL", os.environ.get("VLLM_NGROK_URL", "http://localhost:8001"))
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")

@app.post("/api/v1/chat")
async def chat(request: Request):
    body = await request.json()
    query = body.get("query", "")
    if not query:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="query is required")
    start = time.time()

    context = ""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            search_resp = await client.post(
                f"{QDRANT_URL}/collections/documents/points/search",
                json={"vector": body.get("embedding", [0.0] * 384), "limit": 3}
            )
            if search_resp.status_code == 200:
                context_data = search_resp.json().get("result", [])
                if context_data:
                    context = str(context_data)
    except Exception as e:
        print(f"Qdrant search skipped: {e}")

    prompt = f"Query: {query}" + (f"\nContext: {context}" if context else "")
    
    try:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        async with httpx.AsyncClient(timeout=30, verify=ssl_context) as client:
            llm_resp = await client.post(
                f"{VLLM_URL}/v1/chat/completions",
                json={
                    "model": "Qwen/Qwen2.5-0.5B-Instruct",
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            result = llm_resp.json()
            answer = result.get("choices", [{}])[0].get("message", {}).get("content", "No response")
            model = result.get("model", "unknown")
    except Exception as e:
        answer = f"Error connecting to LLM: {str(e)}"
        model = "error"

    latency = (time.time() - start) * 1000
    return {"answer": answer, "latency_ms": round(latency, 2), "model": model}

@app.get("/health")
def health():
    return {"status": "ok"}