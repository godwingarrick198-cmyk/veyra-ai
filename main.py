import json
import os
import re

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Veyra AI", version="0.5.0", description="AI-powered CV optimization API")


class ResumeRequest(BaseModel):
    text: str = Field(min_length=20)
    target_role: str | None = None


class MatchRequest(BaseModel):
    resume: str = Field(min_length=20)
    job_description: str = Field(min_length=20)
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
    issues = [f"Missing or unclear {name} section" for name, present in sections.items() if not present]
    if len(text.split()) < 180:
        issues.append("Resume may be too brief for a competitive application")
    return {
        "target_role": role,
        "word_count": len(text.split()),
        "sections": sections,
        "issues": issues,
        "suggestions": ["Use measurable achievements", "Match keywords to the target role", "Keep formatting ATS-friendly"],
        "score": max(0, 100 - len(issues) * 10),
        "mode": "local",
    }


def _extract_json(text: str) -> dict:
    cleaned = text.strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Gemini did not return a JSON object")
    return json.loads(cleaned[start:end + 1])


def _gemini_model():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY is not configured")
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gemini setup failed: {exc}")


def gemini_audit(text: str, role: str | None) -> dict:
    if not os.getenv("GEMINI_API_KEY"):
        return local_audit(text, role)
    try:
        model = _gemini_model()
        prompt = f"""
You are Veyra AI, a professional resume/CV analyst.
Analyze the resume below for the target role: {role or "not specified"}.
Evaluate only information actually present. Never invent experience, education, skills, employers, dates, metrics, or qualifications.
Give specific actionable findings. Return ONLY valid JSON.

Required JSON:
{{
  "target_role": string or null,
  "word_count": integer,
  "sections": {{"contact": boolean, "experience": boolean, "education": boolean, "skills": boolean}},
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
        response = model.generate_content(prompt, generation_config={"temperature": 0.2, "response_mime_type": "application/json"})
        result = _extract_json(response.text)
        required = ["target_role", "word_count", "sections", "score", "summary", "issues", "suggestions", "strengths", "missing_keywords"]
        if any(key not in result for key in required):
            raise ValueError("Gemini returned an incomplete audit")
        result["mode"] = "gemini"
        result["score"] = max(0, min(100, int(result["score"])))
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gemini audit failed: {exc}")


def gemini_rewrite(text: str, role: str | None) -> dict:
    try:
        model = _gemini_model()
        prompt = f"""
You are Veyra AI, a professional CV/resume writer.
Rewrite the supplied resume for the target role: {role or "a professional role"}.

NON-NEGOTIABLE FACTUALITY RULES:
- Never invent employers, titles, dates, degrees, certifications, skills, achievements, metrics, responsibilities, links, or contact details.
- Preserve every factual claim while improving wording.
- Never add a metric unless it exists in the source.
- If information is missing, list it in needs_user_input.
- A skills list is evidence only that a person lists or claims that skill. It is NOT evidence that the skill was used at a particular employer or in a particular job.
- Never move a skill, tool, technology, qualification, or personal attribute into an employer's experience section unless the source explicitly connects it to that employer or role.
- Never infer job duties from a job title.
- Never infer projects, backend/frontend work, collaboration, leadership, achievements, responsibilities, or outcomes from a skill, job title, or generic statement.
- If the source only says "Junior Developer at ABC Ltd" and separately lists Python/FastAPI/Git as skills, do not claim those technologies were used at ABC Ltd.
- Keep unsupported experience bullets generic or omit them. Do not manufacture plausible-sounding bullets.
- The professional summary must also remain source-grounded. Do not describe "hands-on experience" with a technology unless the source explicitly establishes that experience.

Optimize for ATS readability, a strong summary, concise achievement-oriented bullets only where supported, relevant terminology without keyword stuffing, and clear sections.

Return ONLY valid JSON:
{{
  "target_role": string or null,
  "professional_summary": string,
  "rewritten_resume": string,
  "changes_made": [string],
  "needs_user_input": [string],
  "ats_keywords_used": [string],
  "warnings": [string]
}}

Resume:
{text}
"""
        response = model.generate_content(prompt, generation_config={"temperature": 0.2, "response_mime_type": "application/json"})
        result = _extract_json(response.text)
        required = ["target_role", "professional_summary", "rewritten_resume", "changes_made", "needs_user_input", "ats_keywords_used", "warnings"]
        if any(key not in result for key in required):
            raise ValueError("Gemini returned an incomplete rewrite")
        result["mode"] = "gemini"
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gemini rewrite failed: {exc}")


def gemini_match(resume: str, job_description: str, role: str | None) -> dict:
    if not os.getenv("GEMINI_API_KEY"):
        return {
            "target_role": role,
            "match_score": 0,
            "matched_keywords": [],
            "missing_keywords": [],
            "matched_requirements": [],
            "missing_requirements": [],
            "strengths": [],
            "gaps": ["GEMINI_API_KEY is not configured, so an AI job match cannot be calculated."],
            "recommendations": ["Configure Gemini and run the match again."],
            "mode": "local",
        }

    try:
        model = _gemini_model()
        prompt = f"""
You are Veyra AI, a professional CV-to-job-description matching engine.

Compare the candidate resume against the job description for the target role: {role or "not specified"}.

IMPORTANT FACTUALITY RULES:
- Use only evidence explicitly present in the resume and job description.
- Never claim the candidate has a skill, qualification, experience, certification, degree, achievement, or responsibility that is not explicitly supported by the resume.
- Do not treat a keyword appearing in the resume as proof of professional experience unless the resume connects it to an experience, project, education, or other relevant context.
- Distinguish clearly between matched evidence and missing evidence.
- Do not penalize the candidate for information that the job description does not require.
- Do not invent missing requirements or keywords.
- A keyword match is not automatically a requirement match.
- Base match_score on overall evidence alignment, not raw keyword frequency.
- Keep the analysis useful and specific.

Return ONLY valid JSON:
{{
  "target_role": string or null,
  "match_score": integer,
  "matched_keywords": [string],
  "missing_keywords": [string],
  "matched_requirements": [string],
  "missing_requirements": [string],
  "strengths": [string],
  "gaps": [string],
  "recommendations": [string]
}}

RESUME:
{resume}

JOB DESCRIPTION:
{job_description}
"""
        response = model.generate_content(prompt, generation_config={"temperature": 0.2, "response_mime_type": "application/json"})
        result = _extract_json(response.text)
        required = [
            "target_role", "match_score", "matched_keywords", "missing_keywords",
            "matched_requirements", "missing_requirements", "strengths", "gaps",
            "recommendations"
        ]
        if any(key not in result for key in required):
            raise ValueError("Gemini returned an incomplete job match")
        result["mode"] = "gemini"
        result["match_score"] = max(0, min(100, int(result["match_score"])))
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gemini job match failed: {exc}")


@app.post("/api/resume/audit")
def audit(request: ResumeRequest):
    return gemini_audit(request.text.strip(), request.target_role)


@app.post("/api/resume/rewrite")
def rewrite(request: ResumeRequest):
    return gemini_rewrite(request.text.strip(), request.target_role)


@app.post("/api/resume/match")
def match(request: MatchRequest):
    return gemini_match(
        request.resume.strip(),
        request.job_description.strip(),
        request.target_role,
    )
