from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .gemini_assist import gemini_is_configured, scan_with_gemini
from .pipeline import calibrate_image, scan_image
from .translation import SUPPORTED_LANGUAGES, get_language

ROOT_DIR = Path(__file__).resolve().parents[2]
SAMPLES_DIR = ROOT_DIR / "sample-images"
DIST_DIR = ROOT_DIR / "dist"

app = FastAPI(title="BrailleLens AI API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if SAMPLES_DIR.exists():
    app.mount("/sample-images", StaticFiles(directory=SAMPLES_DIR), name="sample-images")


async def _read_image_from_request(
    request: Request,
    file: UploadFile | None,
    image_base64: str | None,
) -> bytes:
    if file is not None:
        return await file.read()
    if image_base64:
        try:
            return base64.b64decode(image_base64.split(",")[-1])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid base64 image.") from exc
    if request.headers.get("content-type", "").startswith("application/json"):
        payload = await request.json()
        encoded = payload.get("image_base64") or payload.get("image")
        if encoded:
            return base64.b64decode(encoded.split(",")[-1])
    raise HTTPException(status_code=400, detail="Provide an image file or base64 image.")


def _parse_calibration(raw: str | None) -> dict | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed.get("profile", parsed) if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "BrailleLens AI",
        "version": app.version,
        "gemini_configured": gemini_is_configured(),
    }


@app.get("/api/languages")
def languages() -> dict:
    return {"languages": SUPPORTED_LANGUAGES}


@app.get("/api/samples")
def samples() -> dict:
    samples_file = SAMPLES_DIR / "samples.json"
    if not samples_file.exists():
        return {"samples": []}
    return {"samples": json.loads(samples_file.read_text(encoding="utf-8"))}


@app.get("/api/samples/{file_name}")
def sample_file(file_name: str) -> FileResponse:
    path = SAMPLES_DIR / file_name
    if not path.exists() or path.parent != SAMPLES_DIR:
        raise HTTPException(status_code=404, detail="Sample not found.")
    return FileResponse(path)


@app.post("/api/scan")
async def scan(
    request: Request,
    file: Annotated[UploadFile | None, File()] = None,
    image_base64: Annotated[str | None, Form()] = None,
    language: Annotated[str, Form()] = "en",
    calibration_profile: Annotated[str | None, Form()] = None,
    scan_variant: Annotated[str, Form()] = "auto",
    scan_engine: Annotated[str, Form()] = "local",
    debug: Annotated[bool, Form()] = False,
) -> dict:
    try:
        image_bytes = await _read_image_from_request(request, file, image_base64)
        language_code = get_language(language)["code"]
        local_result = scan_image(image_bytes, language_code, _parse_calibration(calibration_profile), debug, scan_variant)
        local_result["engine"] = "local"
        if scan_engine == "gemini":
            gemini_result = scan_with_gemini(image_bytes, file.content_type if file else "image/jpeg")
            return {
                **local_result,
                "engine": "gemini",
                "ai_assist": gemini_result,
                "text": gemini_result.get("text") or local_result.get("text", ""),
                "translated_text": gemini_result.get("text") or local_result.get("translated_text", ""),
                "confidence": gemini_result.get("confidence", 0) or local_result.get("confidence", 0),
                "warnings": list(dict.fromkeys((gemini_result.get("warnings") or []) + local_result.get("warnings", []))),
            }
        if scan_engine == "hybrid":
            gemini_result = scan_with_gemini(image_bytes, file.content_type if file else "image/jpeg")
            local_result["engine"] = "hybrid"
            local_result["ai_assist"] = gemini_result
            if gemini_result.get("text") and gemini_result.get("confidence", 0) >= local_result.get("confidence", 0):
                local_result["text"] = gemini_result["text"]
                local_result["translated_text"] = gemini_result["text"]
                local_result["confidence"] = gemini_result.get("confidence", local_result["confidence"])
                local_result["warnings"] = list(dict.fromkeys((gemini_result.get("warnings") or []) + local_result.get("warnings", [])))
            return local_result
        local_result["ai_assist"] = {"available": gemini_is_configured()}
        return local_result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/calibrate")
async def calibrate(
    request: Request,
    file: Annotated[UploadFile | None, File()] = None,
    image_base64: Annotated[str | None, Form()] = None,
    expected_text: Annotated[str, Form()] = "",
) -> dict:
    try:
        image_bytes = await _read_image_from_request(request, file, image_base64)
        return calibrate_image(image_bytes, expected_text)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


if DIST_DIR.exists():
    app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets"), name="assets")


    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str) -> FileResponse:
        requested = DIST_DIR / full_path
        if full_path and requested.exists() and requested.is_file():
            return FileResponse(requested)
        return FileResponse(DIST_DIR / "index.html")
