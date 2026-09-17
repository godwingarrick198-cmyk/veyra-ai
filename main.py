import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Veyra AI", version="0.2.0", description="AI-powered CV optimization API")

class ResumeRequest(BaseModel):
    text: str = Field(min_length=20)
    target_role: str | None = None

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "veyra-ai", "version": app.version}

def local_audit(text: str, role: str | None):
    lower = text.lower()
    sections = {"contact": any(x in lower for x in ["@", "phone", "linkedin"]), "experience": any(x in lower for x in ["experience", "employment"]), "education": "education" in lower, "skills": "skills" in lower}
    issues = [f"Missing or unclear {name} section" for name, present in sections.items() if not present]
    if len(text.split()) < 180: issues.append("Resume may be too brief for a competitive application")
    return {"target_role": role, "word_count": len(text.split()), "sections": sections, "issues": issues, "suggestions": ["Use measurable achievements", "Match keywords to the target role", "Keep formatting ATS-friendly"], "score": max(0, 100 - len(issues) * 10)}

@app.post("/api/resume/audit")
def audit(request: ResumeRequest):
    return local_audit(request.text.strip(), request.target_role)

@app.post("/api/resume/rewrite")
def rewrite(request: ResumeRequest):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"mode": "local", "message": "Set GEMINI_API_KEY to enable AI rewriting.", "audit": local_audit(request.text, request.target_role)}
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
        prompt = f"Rewrite this resume for {request.target_role or 'a professional role'}. Preserve facts; do not invent experience. Return plain text with strong achievement bullets.\n\n{request.text}"
        return {"mode": "gemini", "rewritten_resume": model.generate_content(prompt).text, "target_role": request.target_role}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"AI provider error: {exc}")
