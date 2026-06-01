# BrailleLens AI

**Camera-Based Physical Braille Reader with Multilingual Voice Guidance**

BrailleLens AI is a free, local assistive technology prototype that reads real physical Braille from camera images or uploaded photos. It detects physical dot structures, reconstructs Braille cells, translates Grade 1 English Braille into text, and can speak the result in the user's selected language.

> BrailleLens AI is not a Unicode Braille translator. It is a camera-based assistive reader that detects real physical Braille dots, reconstructs Braille cells, translates them into English, and can present or speak the result in the user's preferred language.

## Why It Matters

Braille is essential for many visually impaired users, but caregivers, teachers, volunteers, and accessibility workers may not always read Braille by touch. BrailleLens AI helps bridge that gap by turning physical Braille into readable and spoken output.

## Architecture

```text
Camera / Upload
↓
Image Quality Check
↓
Preprocessing
↓
Hybrid Dot Detection
↓
Grid + Cell Segmentation
↓
Braille Pattern Recognition
↓
English Text
↓
Optional Local Translation
↓
Speech Output + Debug Overlay
```

## Tech Stack

- Frontend: React, Vite, Tailwind CSS, browser camera APIs, Web Speech API
- Backend: Python, FastAPI, OpenCV, NumPy
- Translation: local demo dictionary for English, Hindi, Spanish, and French
- Deployment: local-only, no paid APIs, no cloud dependency

## Features

- Live camera scanning and image upload
- Physical Braille dot detection from real images
- Hybrid OpenCV detector: contours, blobs, and Hough circle fallback
- Document/page crop, CLAHE contrast enhancement, adaptive thresholding, morphology, and quality scoring
- Embossed-dot response layer for raised Braille shadows and highlights
- Line-aware segmentation for multi-line Braille pages
- 2-column x 3-row Braille cell reconstruction
- Grade 1 English Braille mapping
- Multilingual text output: English, Hindi, Spanish, French
- Optional Gemini Assist scan engine for difficult real photos
- Voice Mode using browser `speechSynthesis`
- Debounced voice guidance for scan readiness and warnings
- Scan Quality Meter: focus, lighting, alignment, dot grid, readiness
- Debug overlay for judges: threshold image, dot centers, rows/columns, cell boxes, confidence labels
- Calibration mode for dot spacing and row spacing
- Manual correction mode with interactive six-dot cells
- Sample Demo View for reliable demos under poor webcam conditions

## Local Setup

### 1. Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

Optional Gemini Assist:

```powershell
copy .env.example .env
notepad .env
```

Add:

```text
GEMINI_API_KEY=your_free_tier_key_here
GEMINI_MODEL=gemini-2.5-flash
```

Gemini Assist is optional. Without this key, the app still works with the local OpenCV detector.

### 2. Frontend

In a second terminal:

```powershell
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

## API

### `GET /api/health`

Returns backend status.

### `GET /api/languages`

Returns supported output languages and speech locales.

### `POST /api/scan`

Accepts an image file or base64 image plus optional language, calibration profile, and debug flag.

Returns:

- recognized English text
- translated output
- cell-level Braille patterns
- dot positions
- confidence scores
- quality scores
- processing metrics
- optional debug images

### `POST /api/calibrate`

Accepts a known sample image and returns a session calibration profile.

## Sample Dataset

The `/sample-images` folder contains generated demo samples:

- clear printed Braille
- tilted Braille
- low-light Braille
- blurry Braille

Each sample has an expected output label in `sample-images/samples.json`. These samples are intentionally included so judges can run the same physical-dot pipeline even if live webcam lighting is unreliable.

## Physical Braille Detection

The backend does not read Unicode Braille symbols. It processes camera images:

1. Normalize image size.
2. Crop/deskew the bright paper region when a page is visible on a darker background.
3. Convert to grayscale.
4. Enhance local contrast with CLAHE.
5. Flatten uneven lighting with background subtraction.
6. Build an embossed-dot response map from top-hat and black-hat morphology.
7. Use adaptive thresholding for uneven lighting.
8. Clean the binary image with morphology.
9. Detect physical dot candidates with contours, blobs, embossed response peaks, and Hough circles.
10. Filter dots by area, circularity, radius, neighbor consistency, and confidence.
11. Cluster dot centroids into rows.
12. Segment rows into Braille text lines.
13. Estimate per-line columns and cell pitch.
14. Group grid positions into 2-column x 3-row Braille cells.
15. Convert dot positions into Grade 1 English Braille patterns.

## Real Photo Guidance

Real embossed Braille photos are harder than printed demo sheets. For best results:

- Fill the frame with one page or one paragraph of Braille.
- Use side lighting so raised dots cast visible shadows.
- Keep the camera parallel to the page.
- Avoid motion blur.
- Use Sample Demo View if live lighting is unreliable during judging.

The app is intentionally local-first. Paid vision APIs are not required for the core detector, and many general OCR APIs do not expose the dot-level geometry needed for Braille reconstruction. A future optional cloud/LLM fallback could help describe low-confidence images, but the winning technical proof remains the local physical-dot pipeline and debug overlay.

## Scan Engines

BrailleLens AI supports three scan engines:

- **Local OpenCV**: free, offline, explainable dot detection with debug overlays.
- **Gemini Assist**: optional API mode for difficult real-world photos. Requires `GEMINI_API_KEY`.
- **Hybrid Best**: runs local detection and Gemini Assist when configured, then prefers the higher-confidence result.

For hackathon judging, Local OpenCV demonstrates the real physical-dot pipeline. Gemini Assist can be shown as an optional accessibility fallback for hard photos, not as the only method.

## Performance Notes

The UI reports these metrics after each scan:

- processing time
- dots detected
- cells detected
- overall confidence
- scan warnings

Target demo performance is under 500 ms per still image on typical laptop hardware, depending on image size and OpenCV installation.

## Accessibility Features

- High contrast interface
- Large controls
- Keyboard-accessible buttons and inputs
- ARIA labels and live status region
- Voice Mode toggle
- Speak Result, Stop Speaking, Repeat Guidance controls
- Debounced spoken guidance
- Manual correction for uncertain cells

## Limitations

- Prototype focuses on Grade 1 English Braille input.
- Multilingual mode translates recognized English output, not non-English Braille systems.
- Full contracted Grade 2 Braille is future work.
- Local translation is a small demo dictionary unless offline translation models are added later.
- Browser speech voices vary by device and browser.
- Very poor lighting, extreme blur, or irregular handmade dots can reduce accuracy.

## Future Improvements

- Grade 2 contracted Braille
- Larger offline translation model support
- Better perspective correction
- Continuous video reading mode
- Mobile-native haptics
- Tiny local ML fallback for hard lighting conditions
- More languages and Braille tables
- Offline PWA packaging

## Demo Video

Demo video link: `TODO`
