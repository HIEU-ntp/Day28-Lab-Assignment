from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import random

app = FastAPI()

class ChatRequest(BaseModel):
    model: str = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
    messages: list[dict]
    temperature: float = 0.7
    max_tokens: int = 512

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    user_message = ""
    for msg in request.messages:
        if msg.get("role") == "user":
            user_message = msg.get("content", "")
            break
    
    responses = [
        "Platform engineering is the practice of building and maintaining software infrastructure that enables rapid and reliable application development and deployment. It focuses on automating workflows, improving developer experience, and ensuring system reliability.",
        "Event-driven architecture (EDA) is a software design pattern where components communicate through events rather than direct calls. In AI platforms, this decouples data ingestion, processing, and serving layers, enabling better scalability and resilience.",
        "Kafka is a distributed event streaming platform that serves as the backbone for data pipelines in AI platforms. It provides durable message storage, replay capability, and horizontal scalability.",
        "Feature stores in ML platforms serve as the central repository for managing and serving ML features. They ensure consistency between training and inference, enable feature sharing across models, and provide point-in-time correctness.",
        "Observability in AI platforms involves collecting logs, metrics, and traces to understand system behavior. Tools like Prometheus, Grafana, and LangSmith help monitor model performance, track data lineage, and debug issues."
    ]
    
    return {
        "id": f"chatcmpl-{random.randint(100000, 999999)}",
        "object": "chat.completion",
        "created": 1234567890,
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": random.choice(responses)
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(user_message),
            "completion_tokens": 50,
            "total_tokens": len(user_message) + 50
        }
    }

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)