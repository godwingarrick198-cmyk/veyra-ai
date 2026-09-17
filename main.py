from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Veyra AI", version="0.1.0")

class ResumeRequest(BaseModel):
    text: str
    target_role: str | None = None

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "veyra-ai"}

@app.post("/api/resume/audit")
def audit(request: ResumeRequest):
    text = request.text.strip()
    if not text:
        return {"error": "Resume text is required"}
    return {
        "target_role": request.target_role,
        "word_count": len(text.split()),
        "message": "Resume received. AI audit service is ready for provider configuration.",
        "issues": [],
        "suggestions": [],
    }
