import json
import os
import re

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Veyra AI",
    version="0.3.0",
    description="AI-powered CV optimization API",
)


class ResumeRequest(BaseModel):
    text: str = Field(min_length=20)
    target_role: str | None = None


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "veyra-ai", "version": app.version}


def local_audit(text: str, role: str | None):
    lower = text.lower()
    sections = {
        "contact": any(x in lower for x in ["@", "phone", "linkedin"]),
        "experience": any(x in lower for x in ["experience", "employment"]),
        "education": "education" in lower,
        "skills": "skills" in lower,
    }
    issues = [
        f"Missing or unclear {name} section"
        for name, present in sections.items()
        if not present
    ]
    if len(text.split()) < 180:
        issues.append("Resume may be too brief for a competitive application")
    return {
        "target_role": role,
        "word_count": len(text.split()),
        "sections": sections,
        "issues": issues,
        "suggestions": [
            "Use measurable achievements",
            "Match keywords to the target role",
            "Keep formatting ATS-friendly",
        ],
        "score": max(0, 100 - len(issues) * 10),
        "mode": "local",
    }


def _extract_json(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^\s*\`\`\`(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*\`\`\`\s*$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Gemini did not return a JSON object")
    return json.loads(cleaned[start : end + 1])


def gemini_audit(text: str, role: str | None) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return local_audit(text, role)

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        model = genai.GenerativeModel(model_name)

        prompt = f"""
You are Veyra AI, a professional resume/CV analyst.

Analyze the resume below for the target role: {role or "not specified"}.

Rules:
- Evaluate only information actually present in the resume.
- Never invent experience, education, skills, employers, dates, metrics, or qualifications.
- A section counts as present only when there is enough evidence for it.
- Score the resume from 0 to 100 using your professional assessment of completeness, clarity, ATS readiness, relevance to the target role, and evidence of impact.
- Give specific, actionable findings rather than generic advice.
- Return ONLY valid JSON. No markdown, no code fences.

Required JSON shape:
{{
  "target_role": string or null,
  "word_count": integer,
  "sections": {{
    "contact": boolean,
    "experience": boolean,
    "education": boolean,
    "skills": boolean
  }},
  "score": integer,
  "summary": string,
  "issues": [string],
  "suggestions": [string],
  "strengths": [string],
  "missing_keywords": [string]
}}

Resume:
{text}
"""

        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,
                "response_mime_type": "application/json",
            },
        )
        result = _extract_json(response.text)

        required = [
            "target_role",
            "word_count",
            "sections",
            "score",
            "summary",
            "issues",
            "suggestions",
            "strengths",
            "missing_keywords",
        ]
        if any(key not in result for key in required):
            raise ValueError("Gemini returned an incomplete audit")

        result["mode"] = "gemini"
        result["score"] = max(0, min(100, int(result["score"])))
        return result

    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gemini audit failed: {exc}")


@app.post("/api/resume/audit")
def audit(request: ResumeRequest):
    return gemini_audit(request.text.strip(), request.target_role)


@app.post("/api/resume/rewrite")
def rewrite(request: ResumeRequest):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {
            "mode": "local",
            "message": "Set GEMINI_API_KEY to enable AI rewriting.",
            "audit": local_audit(request.text, request.target_role),
        }

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        )
        prompt = (
            f"Rewrite this resume for {request.target_role or 'a professional role'}. "
            "Preserve facts; do not invent experience. "
            "Return plain text with strong achievement bullets.\n\n"
            f"{request.text}"
        )
        return {
            "mode": "gemini",
            "rewritten_resume": model.generate_content(prompt).text,
            "target_role": request.target_role,
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"AI provider error: {exc}")
