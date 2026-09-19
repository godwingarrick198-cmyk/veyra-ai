import json
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

app = FastAPI(title="Veyra AI", version="0.6.0", description="AI-powered CV optimization API")

DB_PATH = os.getenv("VEYRA_DB_PATH", "./veyra.db")
FLW_BASE_URL = "https://api.flutterwave.com/v3"


class ResumeRequest(BaseModel):
    text: str = Field(min_length=20)
    target_role: str | None = None


class MatchRequest(BaseModel):
    resume: str = Field(min_length=20)
    job_description: str = Field(min_length=20)
    target_role: str | None = None


class PaymentRequest(BaseModel):
    service: str
    customer_name: str = Field(min_length=2)
    customer_email: str = Field(min_length=5)
    redirect_url: str | None = None


class VerifyPaymentRequest(BaseModel):
    transaction_id: int
    tx_ref: str


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = db()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS payments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tx_ref TEXT UNIQUE NOT NULL,
            service TEXT NOT NULL,
            amount INTEGER NOT NULL,
            currency TEXT NOT NULL,
            customer_name TEXT NOT NULL,
            customer_email TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            flutterwave_transaction_id INTEGER,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


@app.on_event("startup")
def startup():
    init_db()


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
        required = ["target_role", "match_score", "matched_keywords", "missing_keywords", "matched_requirements", "missing_requirements", "strengths", "gaps", "recommendations"]
        if any(key not in result for key in required):
            raise ValueError("Gemini returned an incomplete job match")
        result["mode"] = "gemini"
        result["match_score"] = max(0, min(100, int(result["match_score"])))
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gemini job match failed: {exc}")


def _flutterwave_secret() -> str:
    secret = os.getenv("FLUTTERWAVE_SECRET_KEY") or os.getenv("FLW_SECRET_KEY")
    if not secret:
        raise HTTPException(status_code=503, detail="Flutterwave secret key is not configured")
    return secret


def _flutterwave_headers() -> dict:
    return {
        "Authorization": f"Bearer {_flutterwave_secret()}",
        "Content-Type": "application/json",
    }


def _service_prices() -> dict[str, int]:
    return {
        "cv_audit": int(os.getenv("VEYRA_CV_AUDIT_PRICE_NGN", "1000")),
        "cv_rewrite": int(os.getenv("VEYRA_CV_REWRITE_PRICE_NGN", "2500")),
        "job_match": int(os.getenv("VEYRA_JOB_MATCH_PRICE_NGN", "1500")),
        "tailored_cv": int(os.getenv("VEYRA_TAILORED_CV_PRICE_NGN", "5000")),
    }


def _get_payment(tx_ref: str):
    conn = db()
    row = conn.execute("SELECT * FROM payments WHERE tx_ref = ?", (tx_ref,)).fetchone()
    conn.close()
    return row


async def _verify_flutterwave_transaction(transaction_id: int, expected_tx_ref: str, expected_amount: int) -> dict:
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(
                f"{FLW_BASE_URL}/transactions/{transaction_id}/verify",
                headers=_flutterwave_headers(),
            )
        if response.status_code >= 400:
            raise HTTPException(status_code=502, detail="Flutterwave transaction verification failed")
        payload = response.json()
        data = payload.get("data") or {}
        if (
            payload.get("status") != "success"
            or data.get("status") != "successful"
            or data.get("tx_ref") != expected_tx_ref
            or data.get("currency") != "NGN"
            or float(data.get("amount", 0)) < float(expected_amount)
        ):
            raise HTTPException(status_code=400, detail="Payment verification did not match the expected transaction")
        return data
    except HTTPException:
        raise
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Flutterwave connection failed: {exc}")


@app.get("/api/payments/services")
def payment_services():
    return {
        "currency": "NGN",
        "services": [
            {"service": name, "amount": amount}
            for name, amount in _service_prices().items()
        ],
    }


@app.post("/api/payments/create")
async def create_payment(request: PaymentRequest):
    prices = _service_prices()
    if request.service not in prices:
        raise HTTPException(status_code=400, detail="Unknown Veyra service")
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", request.customer_email):
        raise HTTPException(status_code=400, detail="Invalid customer email")

    amount = prices[request.service]
    tx_ref = f"VEYRA-{uuid.uuid4().hex.upper()}"
    redirect_url = request.redirect_url or os.getenv(
        "FLUTTERWAVE_REDIRECT_URL",
        "https://veyra-ai-xe3o.onrender.com/payment/callback",
    )

    payload = {
        "tx_ref": tx_ref,
        "amount": amount,
        "currency": "NGN",
        "redirect_url": redirect_url,
        "customer": {
            "email": request.customer_email,
            "name": request.customer_name,
        },
        "customizations": {
            "title": "Veyra AI",
            "description": f"Payment for {request.service.replace('_', ' ')}",
        },
        "meta": {"service": request.service},
    }

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(
                f"{FLW_BASE_URL}/payments",
                headers=_flutterwave_headers(),
                json=payload,
            )
        if response.status_code >= 400:
            raise HTTPException(status_code=502, detail="Flutterwave could not create the payment")
        data = response.json()
        if data.get("status") != "success" or not data.get("data", {}).get("link"):
            raise HTTPException(status_code=502, detail="Flutterwave returned an invalid checkout response")
    except HTTPException:
        raise
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Flutterwave connection failed: {exc}")

    now = datetime.now(timezone.utc).isoformat()
    conn = db()
    conn.execute(
        """
        INSERT INTO payments
        (tx_ref, service, amount, currency, customer_name, customer_email, status, created_at, updated_at)
        VALUES (?, ?, ?, 'NGN', ?, ?, 'pending', ?, ?)
        """,
        (tx_ref, request.service, amount, request.customer_name, request.customer_email, now, now),
    )
    conn.commit()
    conn.close()

    return {
        "status": "success",
        "tx_ref": tx_ref,
        "service": request.service,
        "amount": amount,
        "currency": "NGN",
        "checkout_url": data["data"]["link"],
    }


@app.post("/api/payments/verify")
async def verify_payment(request: VerifyPaymentRequest):
    payment = _get_payment(request.tx_ref)
    if not payment:
        raise HTTPException(status_code=404, detail="Payment reference not found")

    data = await _verify_flutterwave_transaction(
        request.transaction_id,
        request.tx_ref,
        int(payment["amount"]),
    )

    now = datetime.now(timezone.utc).isoformat()
    conn = db()
    conn.execute(
        """
        UPDATE payments
        SET status = 'paid', flutterwave_transaction_id = ?, updated_at = ?
        WHERE tx_ref = ?
        """,
        (int(data["id"]), now, request.tx_ref),
    )
    conn.commit()
    conn.close()

    return {
        "status": "paid",
        "tx_ref": request.tx_ref,
        "service": payment["service"],
        "amount": payment["amount"],
        "currency": payment["currency"],
        "flutterwave_transaction_id": data["id"],
    }


@app.get("/api/payments/{tx_ref}")
def payment_status(tx_ref: str):
    payment = _get_payment(tx_ref)
    if not payment:
        raise HTTPException(status_code=404, detail="Payment reference not found")
    return dict(payment)


@app.post("/api/payments/webhook")
async def flutterwave_webhook(
    request: Request,
    verif_hash: str | None = Header(default=None, alias="verif-hash"),
):
    secret_hash = os.getenv("FLUTTERWAVE_SECRET_HASH") or os.getenv("FLW_SECRET_HASH")
    if not secret_hash:
        raise HTTPException(status_code=503, detail="Flutterwave webhook secret hash is not configured")
    if not verif_hash or verif_hash != secret_hash:
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    payload = await request.json()
    event = payload.get("event")
    data = payload.get("data") or {}
    if event != "charge.completed" or not data.get("id") or not data.get("tx_ref"):
        return {"status": "ignored"}

    payment = _get_payment(data["tx_ref"])
    if not payment:
        return {"status": "ignored"}

    if payment["status"] == "paid":
        return {"status": "ok", "message": "Already processed"}

    verified = await _verify_flutterwave_transaction(
        int(data["id"]),
        data["tx_ref"],
        int(payment["amount"]),
    )

    now = datetime.now(timezone.utc).isoformat()
    conn = db()
    conn.execute(
        """
        UPDATE payments
        SET status = 'paid', flutterwave_transaction_id = ?, updated_at = ?
        WHERE tx_ref = ? AND status != 'paid'
        """,
        (int(verified["id"]), now, data["tx_ref"]),
    )
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.get("/payment/callback")
def payment_callback():
    return {
        "status": "returned",
        "message": "Payment completed or cancelled. Veyra will verify the transaction before unlocking the service.",
    }


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
