"""
AnemiaSense — Vercel Serverless Function Gateway
Provides serverless API routing with proxy capability to full deep learning models.
"""

import os
import sys
import json
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import httpx

# Optional: If a dedicated inference backend is configured (Render, Railway, Hugging Face, Modal)
BACKEND_URL = os.environ.get("BACKEND_URL", "").rstrip("/")

app = FastAPI(title="AnemiaSense Vercel Serverless API", version="3.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    if BACKEND_URL:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.get(f"{BACKEND_URL}/api/health")
                return res.json()
        except Exception as e:
            return {"status": "degraded", "backend_error": str(e), "platform": "Vercel Edge"}
    return {
        "status": "online",
        "service": "AnemiaSense Vercel Serverless Gateway",
        "version": "3.1.0",
        "backend_url_configured": bool(BACKEND_URL),
        "platform": "Vercel Serverless"
    }


@app.get("/api/metadata")
async def metadata():
    if BACKEND_URL:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.get(f"{BACKEND_URL}/api/metadata")
                return res.json()
        except Exception:
            pass
    return {
        "eye_model": {"architecture": "MobileNetV2 Multitask", "task": "Hemoglobin Estimation & Anemia Risk"},
        "nail_model": {"architecture": "MobileNetV3Large", "task": "Hemoglobin Regression"}
    }


@app.get("/api/history/stats")
async def history_stats():
    if BACKEND_URL:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.get(f"{BACKEND_URL}/api/history/stats")
                return res.json()
        except Exception:
            pass
    try:
        p = Path(__file__).resolve().parent.parent / "data" / "history.json"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                history = json.load(f)
            total = len(history)
            hb_vals = [
                h.get("screening_results", {}).get("estimated_hemoglobin_g_dl")
                for h in history
                if h.get("screening_results", {}).get("estimated_hemoglobin_g_dl") is not None
            ]
            avg_hb = round(sum(hb_vals) / len(hb_vals), 1) if hb_vals else 0.0
            anemic = [
                h for h in history
                if h.get("screening_results", {}).get("severity_level") in ["MILD", "MODERATE", "SEVERE"]
            ]
            return {
                "status": "success",
                "stats": {
                    "total_screenings": total,
                    "average_hb": avg_hb,
                    "anemia_count": len(anemic),
                    "anemia_percentage": round((len(anemic) / total * 100), 1) if total else 0
                }
            }
    except Exception:
        pass
    return {"status": "success", "stats": {"total_screenings": 0, "average_hb": 0.0, "anemia_count": 0, "anemia_percentage": 0}}


@app.get("/api/history")
async def history_list(limit: int = 50, severity: str = "ALL", search: str = ""):
    if BACKEND_URL:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.get(f"{BACKEND_URL}/api/history", params={"limit": limit, "severity": severity, "search": search})
                return res.json()
        except Exception:
            pass
    try:
        p = Path(__file__).resolve().parent.parent / "data" / "history.json"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                history = json.load(f)
            return {"status": "success", "history": history[:limit]}
    except Exception:
        pass
    return {"status": "success", "history": []}


@app.post("/api/predict")
@app.post("/predict/eye")
async def predict_eye(
    file: UploadFile = File(...),
    gender: str = Form("M"),
    age: int = Form(30),
    dietary_preference: str = Form("non-vegetarian")
):
    if BACKEND_URL:
        file_bytes = await file.read()
        files = {"file": (file.filename, file_bytes, file.content_type)}
        data = {"gender": gender, "age": str(age), "dietary_preference": dietary_preference}
        async with httpx.AsyncClient(timeout=60.0) as client:
            res = await client.post(f"{BACKEND_URL}/predict/eye", files=files, data=data)
            return Response(content=res.content, status_code=res.status_code, media_type="application/json")

    # Local fallback if tensorflow environment is available
    try:
        from web.app import predict_eye as local_predict_eye
        return await local_predict_eye(file=file, gender=gender, age=age, dietary_preference=dietary_preference)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=(
                "Inference server is not connected. "
                "Set BACKEND_URL in your Vercel Project Settings to your free backend (e.g. Render/Railway), "
                "or enable VERCEL_SUPPORT_LARGE_FUNCTIONS=1. Error: " + str(e)
            )
        )


@app.post("/api/predict/nail")
@app.post("/predict/nail")
async def predict_nail(
    file: UploadFile = File(...),
    gender: str = Form("M"),
    age: int = Form(30),
    dietary_preference: str = Form("non-vegetarian")
):
    if BACKEND_URL:
        file_bytes = await file.read()
        files = {"file": (file.filename, file_bytes, file.content_type)}
        data = {"gender": gender, "age": str(age), "dietary_preference": dietary_preference}
        async with httpx.AsyncClient(timeout=60.0) as client:
            res = await client.post(f"{BACKEND_URL}/predict/nail", files=files, data=data)
            return Response(content=res.content, status_code=res.status_code, media_type="application/json")

    # Local fallback if tensorflow environment is available
    try:
        from web.app import predict_nail as local_predict_nail
        return await local_predict_nail(file=file, gender=gender, age=age, dietary_preference=dietary_preference)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=(
                "Inference server is not connected. "
                "Set BACKEND_URL in your Vercel Project Settings to your free backend (e.g. Render/Railway), "
                "or enable VERCEL_SUPPORT_LARGE_FUNCTIONS=1. Error: " + str(e)
            )
        )
