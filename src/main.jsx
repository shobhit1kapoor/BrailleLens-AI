import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  Camera,
  CheckCircle2,
  CircleDot,
  Eye,
  Gauge,
  Languages,
  Mic,
  RefreshCcw,
  Settings2,
  SlidersHorizontal,
  Speaker,
  Square,
  Upload,
  Volume2,
  VolumeX,
  Wand2
} from "lucide-react";
import "./styles.css";

const API_BASE = "";
const PATTERN_TO_CHAR = {
  "": " ",
  1: "a",
  12: "b",
  14: "c",
  145: "d",
  15: "e",
  124: "f",
  1245: "g",
  125: "h",
  24: "i",
  245: "j",
  13: "k",
  123: "l",
  134: "m",
  1345: "n",
  135: "o",
  1234: "p",
  12345: "q",
  1235: "r",
  234: "s",
  2345: "t",
  136: "u",
  1236: "v",
  2456: "w",
  1346: "x",
  13456: "y",
  1356: "z",
  2: ",",
  23: ";",
  25: ":",
  256: ".",
  235: "!",
  236: "?",
  3: "'",
  36: "-"
};

const DEMO_TRANSLATIONS = {
  "hello world": {
    hi: "नमस्ते दुनिया",
    es: "hola mundo",
    fr: "bonjour le monde"
  },
  hello: {
    hi: "नमस्ते",
    es: "hola",
    fr: "bonjour"
  }
};

const GUIDANCE = {
  en: {
    ready: "Ready to scan",
    blurry: "Too blurry. Hold the camera steady or move closer.",
    lowLight: "Lighting is low. Move the page toward a brighter area.",
    detected: "Braille detected",
    reading: "Reading Braille now",
    uncertain: "The scan result may be uncertain. Please rescan or use manual correction.",
    noResult: "No recognized Braille text yet."
  },
  hi: {
    ready: "स्कैन के लिए तैयार",
    blurry: "छवि धुंधली है। कैमरा स्थिर रखें या पास जाएं।",
    lowLight: "रोशनी कम है। पेज को अधिक रोशनी में रखें।",
    detected: "ब्रेल मिला",
    reading: "अब ब्रेल पढ़ रहा है",
    uncertain: "स्कैन परिणाम अनिश्चित हो सकता है। कृपया फिर से स्कैन करें या सुधार करें।",
    noResult: "अभी कोई ब्रेल पाठ पहचाना नहीं गया।"
  },
  es: {
    ready: "Listo para escanear",
    blurry: "La imagen está borrosa. Mantén la cámara estable o acércate.",
    lowLight: "La iluminación es baja. Mueve la página a una zona más brillante.",
    detected: "Braille detectado",
    reading: "Leyendo braille ahora",
    uncertain: "El resultado puede ser incierto. Vuelve a escanear o usa la corrección manual.",
    noResult: "Todavía no hay texto braille reconocido."
  },
  fr: {
    ready: "Prêt à scanner",
    blurry: "L'image est floue. Stabilisez la caméra ou rapprochez-vous.",
    lowLight: "La lumière est faible. Placez la page dans une zone plus claire.",
    detected: "Braille détecté",
    reading: "Lecture du braille",
    uncertain: "Le résultat peut être incertain. Recommencez ou corrigez manuellement.",
    noResult: "Aucun texte braille reconnu pour le moment."
  }
};

const SCAN_MODES = [
  { value: "auto", label: "Auto Enhanced", hint: "Tries all local scanners and chooses the best result." },
  { value: "balanced", label: "Balanced", hint: "Best default for clear printed or clean embossed samples." },
  { value: "printed", label: "Printed / High Contrast", hint: "Optimized for dark dots or crisp printed Braille." },
  { value: "embossed", label: "Embossed Photo", hint: "Uses shadow/highlight response for raised Braille." },
  { value: "sensitive", label: "Low Light Sensitive", hint: "More aggressive detection for dim photos; may find more noise." }
];

const SCAN_ENGINES = [
  { value: "local", label: "Local OpenCV", hint: "Free, offline, explainable physical-dot detector." },
  { value: "gemini", label: "Gemini Assist", hint: "Optional free-tier API assist for difficult real photos." },
  { value: "hybrid", label: "Hybrid Best", hint: "Runs local scan and asks Gemini when configured." }
];

function translateLocally(text, language) {
  if (language === "en") return text;
  const normalized = text.toLowerCase().trim();
  if (DEMO_TRANSLATIONS[normalized]?.[language]) return DEMO_TRANSLATIONS[normalized][language];
  return normalized
    .split(/\s+/)
    .map((word) => DEMO_TRANSLATIONS[word]?.[language] ?? word)
    .join(" ");
}

function classNames(...items) {
  return items.filter(Boolean).join(" ");
}

async function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function ScoreBar({ label, value, detail }) {
  const percent = Math.round((value || 0) * 100);
  const tone = percent >= 75 ? "good" : percent >= 45 ? "warn" : "bad";
  return (
    <div className="meter-row">
      <div>
        <span>{label}</span>
        <strong>{detail}</strong>
      </div>
      <div className="meter-track" aria-hidden="true">
        <div className={`meter-fill ${tone}`} style={{ width: `${percent}%` }} />
      </div>
      <b>{percent}%</b>
    </div>
  );
}

function statusFor(value, good = "Good", weak = "Needs improvement") {
  if (value >= 0.72) return good;
  if (value >= 0.45) return "Fair";
  return weak;
}

function BrailleCellEditor({ cell, onToggle }) {
  const active = new Set((cell.pattern || "").split(""));
  return (
    <div className="cell-editor" aria-label={`Braille cell ${cell.index + 1}, current character ${cell.char || "unknown"}`}>
      <div className="cell-editor-header">
        <span>Cell {cell.index + 1}</span>
        <strong>{cell.char || "?"}</strong>
      </div>
      <div className="six-dot-grid">
        {[1, 4, 2, 5, 3, 6].map((position) => (
          <button
            key={position}
            type="button"
            aria-label={`Toggle dot ${position}`}
            className={classNames("dot-toggle", active.has(String(position)) && "active")}
            onClick={() => onToggle(cell.index, position)}
          >
            {position}
          </button>
        ))}
      </div>
    </div>
  );
}

function useSpeech({ voiceMode, language, selectedVoiceName, rate, pitch, volume }) {
  const [voices, setVoices] = useState([]);
  const lastSpokenRef = useRef({ key: "", time: 0 });

  useEffect(() => {
    const loadVoices = () => setVoices(window.speechSynthesis?.getVoices?.() || []);
    loadVoices();
    window.speechSynthesis?.addEventListener?.("voiceschanged", loadVoices);
    return () => window.speechSynthesis?.removeEventListener?.("voiceschanged", loadVoices);
  }, []);

  const preferredVoice = useMemo(() => {
    const localePrefix = { en: "en", hi: "hi", es: "es", fr: "fr" }[language] || "en";
    return (
      voices.find((voice) => voice.name === selectedVoiceName) ||
      voices.find((voice) => voice.lang?.toLowerCase().startsWith(localePrefix)) ||
      voices.find((voice) => voice.lang?.toLowerCase().startsWith("en"))
    );
  }, [voices, selectedVoiceName, language]);

  const speak = (text, { force = false, key = text } = {}) => {
    if (!text || !window.speechSynthesis) return;
    const now = Date.now();
    if (!force && lastSpokenRef.current.key === key && now - lastSpokenRef.current.time < 5500) return;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.voice = preferredVoice || null;
    utterance.rate = Number(rate);
    utterance.pitch = Number(pitch);
    utterance.volume = Number(volume);
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
    lastSpokenRef.current = { key, time: now };
  };

  const speakAuto = (text, options) => {
    if (voiceMode) speak(text, options);
  };

  const stop = () => window.speechSynthesis?.cancel();

  return { voices, preferredVoice, speak, speakAuto, stop };
}

function App() {
  const [view, setView] = useState("scan");
  const [languages, setLanguages] = useState([]);
  const [language, setLanguage] = useState("en");
  const [voiceMode, setVoiceMode] = useState(true);
  const [debug, setDebug] = useState(true);
  const [scanMode, setScanMode] = useState("auto");
  const [scanEngine, setScanEngine] = useState("local");
  const [geminiConfigured, setGeminiConfigured] = useState(false);
  const [status, setStatus] = useState("Choose a sample, upload an image, or start the camera.");
  const [result, setResult] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);
  const [samples, setSamples] = useState([]);
  const [cameraOn, setCameraOn] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const [calibration, setCalibration] = useState(null);
  const [expectedText, setExpectedText] = useState("hello world");
  const [selectedVoiceName, setSelectedVoiceName] = useState("");
  const [rate, setRate] = useState(0.95);
  const [pitch, setPitch] = useState(1);
  const [volume, setVolume] = useState(1);
  const [isDragging, setIsDragging] = useState(false);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const streamRef = useRef(null);
  const { voices, preferredVoice, speak, speakAuto, stop } = useSpeech({
    voiceMode,
    language,
    selectedVoiceName,
    rate,
    pitch,
    volume
  });

  useEffect(() => {
    fetch(`${API_BASE}/api/languages`)
      .then((response) => response.json())
      .then((data) => setLanguages(data.languages || []))
      .catch(() => setLanguages([{ code: "en", name: "English", speech_locale: "en-US" }]));
    fetch(`${API_BASE}/api/samples`)
      .then((response) => response.json())
      .then((data) => setSamples(data.samples || []))
      .catch(() => setSamples([]));
    fetch(`${API_BASE}/api/health`)
      .then((response) => response.json())
      .then((data) => setGeminiConfigured(Boolean(data.gemini_configured)))
      .catch(() => setGeminiConfigured(false));
  }, []);

  useEffect(() => {
    return () => {
      streamRef.current?.getTracks?.().forEach((track) => track.stop());
    };
  }, []);

  useEffect(() => {
    if (!result) return;
    const guidance = result.confidence < 0.48 ? localText("uncertain") : `${result.translated_text || result.text}`;
    speakAuto(
      result.confidence < 0.48
        ? guidance
        : `${language === "en" ? "Detected Braille text" : "Braille"}: ${result.translated_text || result.text}`,
      { key: `result-${result.text}-${language}` }
    );
  }, [result]);

  function localText(key) {
    return GUIDANCE[language]?.[key] || GUIDANCE.en[key];
  }

  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false
      });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setCameraOn(true);
      setStatus(localText("detected"));
      speakAuto(localText("detected"), { key: "camera-start" });
    } catch {
      setStatus("Camera permission was blocked. Use image upload or sample demo mode.");
    }
  }

  function stopCamera() {
    streamRef.current?.getTracks?.().forEach((track) => track.stop());
    streamRef.current = null;
    setCameraOn(false);
  }

  async function captureFrame() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) return null;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.9);
  }

  function appendScanOptions(body) {
    body.append("language", language);
    body.append("scan_variant", scanMode);
    body.append("scan_engine", scanEngine);
    body.append("debug", String(debug));
    if (calibration) body.append("calibration_profile", JSON.stringify(calibration.profile || calibration));
  }

  async function submitScan(body, previewImage) {
    setIsScanning(true);
    setStatus(localText("reading"));
    speakAuto(localText("reading"), { key: "reading" });
    appendScanOptions(body);
    try {
      const response = await fetch(`${API_BASE}/api/scan`, { method: "POST", body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Scan failed");
      setResult(data);
      setSelectedImage(previewImage);
      setView("result");
      const warning = data.warnings?.find((item) => item.includes("blurry") || item.includes("Lighting"));
      if (warning?.includes("blurry")) speakAuto(localText("blurry"), { key: "blurry" });
      if (warning?.includes("Lighting")) speakAuto(localText("lowLight"), { key: "light" });
      setStatus(data.warnings?.[0] || localText("ready"));
    } catch (error) {
      setStatus(error.message === "Failed to fetch" ? "Upload failed. Check that the backend is running and try a smaller image." : error.message);
    } finally {
      setIsScanning(false);
    }
  }

  async function scanDataUrl(dataUrl) {
    const body = new FormData();
    body.append("image_base64", dataUrl);
    await submitScan(body, dataUrl);
  }

  async function scanUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    await scanFile(file);
    event.target.value = "";
  }

  async function scanFile(file) {
    if (!file.type.startsWith("image/")) {
      setStatus("Please drop or upload a JPG, PNG, or other image file.");
      return;
    }
    const previewDataUrl = await fileToDataUrl(file);
    const body = new FormData();
    body.append("file", file, file.name || "braille-upload.png");
    await submitScan(body, previewDataUrl);
  }

  async function handleDrop(event) {
    event.preventDefault();
    setIsDragging(false);
    const file = event.dataTransfer.files?.[0];
    if (file) await scanFile(file);
  }

  async function scanCamera() {
    const dataUrl = await captureFrame();
    if (dataUrl) await scanDataUrl(dataUrl);
  }

  async function scanSample(sample) {
    const response = await fetch(`/sample-images/${sample.file}`);
    const blob = await response.blob();
    const file = new File([blob], sample.file, { type: blob.type || "image/png" });
    const dataUrl = await fileToDataUrl(file);
    await scanDataUrl(dataUrl);
  }

  async function calibrateFromCurrent() {
    const dataUrl = selectedImage || (await captureFrame());
    if (!dataUrl) {
      setStatus("Capture or upload an image before calibration.");
      return;
    }
    const body = new FormData();
    body.append("image_base64", dataUrl);
    body.append("expected_text", expectedText);
    const response = await fetch(`${API_BASE}/api/calibrate`, { method: "POST", body });
    const data = await response.json();
    setCalibration(data);
    setStatus("Calibration profile saved for this session.");
  }

  function toggleCell(index, position) {
    if (!result) return;
    const cells = result.cells.map((cell) => {
      if (cell.index !== index) return cell;
      const dots = new Set((cell.pattern || "").split("").filter(Boolean));
      const key = String(position);
      if (dots.has(key)) dots.delete(key);
      else dots.add(key);
      const pattern = [...dots].sort().join("");
      return {
        ...cell,
        pattern,
        char: PATTERN_TO_CHAR[pattern] ?? "?",
        confidence: 1,
        corrected: true
      };
    });
    const text = cells.map((cell) => cell.char || "").join("").replace(/\s+/g, " ").trim();
    setResult({
      ...result,
      cells,
      text,
      translated_text: translateLocally(text, language),
      confidence: 1
    });
  }

  const quality = result?.quality;
  const languageMeta = languages.find((item) => item.code === language);
  const languageVoices = voices.filter((voice) => voice.lang?.toLowerCase().startsWith(language));

  return (
    <main className="app-shell">
      <section className="topbar" aria-label="Application header">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true">
            <span />
            <span />
            <span />
            <span />
            <span />
            <span />
          </div>
          <div>
            <span className="eyebrow">Physical Braille Reader</span>
            <h1>BrailleLens AI</h1>
            <p>Reader console</p>
          </div>
        </div>
        <div className="topbar-actions">
          <span className={classNames("service-pill", geminiConfigured && "online")}>
            {geminiConfigured ? "AI Assist ready" : "Local mode"}
          </span>
          <label className="select-label">
            <Languages size={18} aria-hidden="true" />
            <span>Output Language</span>
            <select value={language} onChange={(event) => setLanguage(event.target.value)} aria-label="Output language">
              {languages.map((item) => (
                <option key={item.code} value={item.code}>
                  {item.name}
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            className={classNames("toggle-button", voiceMode && "active")}
            onClick={() => setVoiceMode((value) => !value)}
            aria-pressed={voiceMode}
          >
            {voiceMode ? <Volume2 size={20} /> : <VolumeX size={20} />}
            Voice Mode
          </button>
        </div>
      </section>

      <nav className="view-tabs" aria-label="Primary views">
        {[
          ["scan", Camera, "Live Scan"],
          ["result", CheckCircle2, "Result"],
          ["debug", Eye, "Debug"],
          ["calibration", SlidersHorizontal, "Calibration"],
          ["samples", Wand2, "Samples"]
        ].map(([key, Icon, label]) => (
          <button key={key} type="button" className={view === key ? "active" : ""} onClick={() => setView(key)}>
            <Icon size={18} />
            {label}
          </button>
        ))}
      </nav>

      <div className="status-strip" role="status" aria-live="polite">
        <Activity size={20} aria-hidden="true" />
        <span>{status}</span>
      </div>

      {view === "scan" && (
        <section className="main-grid">
          <div className="scanner-panel">
            <div className="camera-frame">
              {cameraOn ? (
                <video ref={videoRef} autoPlay playsInline muted aria-label="Live camera preview" />
              ) : selectedImage ? (
                <img src={selectedImage} alt="Selected Braille sample preview" />
              ) : (
                <div className="camera-placeholder">
                  <CircleDot size={64} />
                  <span>Start camera or upload physical Braille image</span>
                </div>
              )}
              <canvas ref={canvasRef} className="hidden-canvas" aria-hidden="true" />
            </div>
            <div className="button-row">
              {!cameraOn ? (
                <button type="button" className="primary" onClick={startCamera}>
                  <Camera size={22} />
                  Start Camera
                </button>
              ) : (
                <button type="button" onClick={stopCamera}>
                  <Square size={20} />
                  Stop Camera
                </button>
              )}
              <button type="button" className="primary" onClick={scanCamera} disabled={!cameraOn || isScanning}>
                <Gauge size={22} />
                {isScanning ? "Scanning..." : "Scan Frame"}
              </button>
              <label className="file-button">
                <Upload size={22} />
                Upload Image
                <input ref={fileInputRef} type="file" accept="image/*" onChange={scanUpload} />
              </label>
            </div>
            <button
              type="button"
              className={classNames("drop-zone", isDragging && "dragging")}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(event) => {
                event.preventDefault();
                setIsDragging(true);
              }}
              onDragEnter={(event) => {
                event.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              aria-label="Drop a Braille image here or click to choose an image file"
            >
              <Upload size={30} aria-hidden="true" />
              <span>{isDragging ? "Release to scan this image" : "Drag and drop Braille photo here"}</span>
              <small>JPG, PNG, or phone camera image</small>
            </button>
          </div>

          <aside className="side-panel">
            <h2>Scan Quality Meter</h2>
            <label className="select-label scan-mode-select">
              <Wand2 size={18} aria-hidden="true" />
              <span>Scan Engine</span>
              <select value={scanEngine} onChange={(event) => setScanEngine(event.target.value)} aria-label="Scan engine">
                {SCAN_ENGINES.map((engine) => (
                  <option key={engine.value} value={engine.value}>
                    {engine.label}
                  </option>
                ))}
              </select>
              <small>
                {SCAN_ENGINES.find((engine) => engine.value === scanEngine)?.hint}
                {scanEngine !== "local" && !geminiConfigured ? " Add GEMINI_API_KEY in backend/.env to enable it." : ""}
              </small>
            </label>
            <label className="select-label scan-mode-select">
              <SlidersHorizontal size={18} aria-hidden="true" />
              <span>Scan Mode</span>
              <select value={scanMode} onChange={(event) => setScanMode(event.target.value)} aria-label="Scan mode">
                {SCAN_MODES.map((mode) => (
                  <option key={mode.value} value={mode.value}>
                    {mode.label}
                  </option>
                ))}
              </select>
              <small>{SCAN_MODES.find((mode) => mode.value === scanMode)?.hint}</small>
            </label>
            {quality ? (
              <>
                <ScoreBar label="Focus" value={quality.focus} detail={statusFor(quality.focus)} />
                <ScoreBar label="Lighting" value={quality.brightness} detail={statusFor(quality.brightness, "Good", "Low / harsh")} />
                <ScoreBar label="Alignment" value={quality.alignment} detail={statusFor(quality.alignment, "Good", "Tilted")} />
                <ScoreBar label="Dot grid" value={quality.dot_grid} detail={statusFor(quality.dot_grid, "Detected", "Weak / not found")} />
                <ScoreBar label="Readiness" value={quality.readiness} detail={statusFor(quality.readiness, "Ready", "Needs improvement")} />
              </>
            ) : (
              <p className="muted">No scan loaded.</p>
            )}
            <div className="voice-card">
              <h3>Voice Guidance</h3>
              <p>{voiceMode ? "Automatic guidance and results are spoken." : "Automatic speech is paused."}</p>
              <div className="button-row compact">
                <button type="button" onClick={() => speak(result ? `Detected Braille text: ${result.translated_text || result.text}` : localText("noResult"), { force: true })}>
                  <Speaker size={18} />
                  Speak Result
                </button>
                <button type="button" onClick={stop}>
                  <VolumeX size={18} />
                  Stop
                </button>
                <button type="button" onClick={() => speak(status, { force: true })}>
                  <RefreshCcw size={18} />
                  Repeat Guidance
                </button>
              </div>
            </div>
          </aside>
        </section>
      )}

      {view === "result" && (
        <section className="result-layout">
          <div className="result-card">
            <span className="eyebrow">Recognized English</span>
            <h2>{result?.text || "No scan yet"}</h2>
            <span className="eyebrow">Translated Output</span>
            <h3>{result?.translated_text || "Scan physical Braille to generate output"}</h3>
            <div className="metric-grid">
              <Metric label="Confidence" value={result ? `${Math.round(result.confidence * 100)}%` : "0%"} />
              <Metric label="Processing" value={result ? `${result.metrics.processing_ms} ms` : "-"} />
              <Metric label="Dots" value={result?.metrics.dots_detected ?? "-"} />
              <Metric label="Cells" value={result?.metrics.cells_detected ?? "-"} />
              <Metric label="Mode" value={result?.metrics.selected_variant ?? "-"} />
              <Metric label="Variants" value={result?.metrics.variants_tried ?? "-"} />
              <Metric label="Engine" value={result?.engine ?? "-"} />
            </div>
            {result?.ai_assist && (
              <div className="ai-assist-box">
                <strong>AI Assist</strong>
                <span>
                  {result.ai_assist.available
                    ? result.ai_assist.text
                      ? `Gemini ${result.ai_assist.model || ""} returned a result.`
                      : "Gemini Assist is configured, but the free API did not return a usable result. Local OpenCV output is shown."
                    : "Gemini Assist is not configured."}
                </span>
                {result.ai_assist.text && <small>{result.ai_assist.text}</small>}
                {result.ai_assist.warnings?.map((warning) => (
                  <small key={warning}>{warning}</small>
                ))}
              </div>
            )}
            {result?.warnings?.length > 0 && (
              <div className="warning-box">
                <strong>Scan notes</strong>
                {result.warnings.map((warning) => (
                  <span key={warning}>{warning}</span>
                ))}
              </div>
            )}
            <div className="button-row">
              <button type="button" className="primary" onClick={() => setView("scan")}>
                <RefreshCcw size={20} />
                Scan Again
              </button>
              <button type="button" onClick={() => speak(`Detected Braille text: ${result?.translated_text || result?.text || ""}`, { force: true })}>
                <Speaker size={20} />
                Speak Result
              </button>
              <button type="button" onClick={stop}>
                <VolumeX size={20} />
                Stop Speaking
              </button>
            </div>
          </div>
          <div className="correction-panel">
            <h2>Manual Correction</h2>
            <p>Review uncertain cells.</p>
            <div className="cell-list">
              {result?.cells?.map((cell) => (
                <BrailleCellEditor key={cell.index} cell={cell} onToggle={toggleCell} />
              ))}
            </div>
          </div>
        </section>
      )}

      {view === "debug" && (
        <section className="debug-layout">
          <div className="debug-header">
            <div>
              <h2>Judge Debug Overlay</h2>
              <p>Detection layers and cell geometry.</p>
            </div>
            <button type="button" className={debug ? "active" : ""} onClick={() => setDebug((value) => !value)}>
              <Eye size={20} />
              Debug Overlay
            </button>
          </div>
          <div className="image-grid">
            <ImagePanel title="Original / Captured" src={selectedImage} />
            <ImagePanel title="Dot Enhanced Image" src={result?.debug?.dot_enhanced_image_base64 ? `data:image/png;base64,${result.debug.dot_enhanced_image_base64}` : null} />
            <ImagePanel title="Separated Dot Mask" src={result?.debug?.separated_threshold_image_base64 ? `data:image/png;base64,${result.debug.separated_threshold_image_base64}` : null} />
            <ImagePanel title="Rough Threshold Image" src={result?.debug?.threshold_image_base64 ? `data:image/png;base64,${result.debug.threshold_image_base64}` : null} />
            <ImagePanel title="Embossed Dot Response" src={result?.debug?.embossed_response_base64 ? `data:image/png;base64,${result.debug.embossed_response_base64}` : null} />
            <ImagePanel title="Detection Overlay" src={result?.debug?.overlay_image_base64 ? `data:image/png;base64,${result.debug.overlay_image_base64}` : null} />
          </div>
          <pre className="json-panel">{JSON.stringify(result?.debug || {}, null, 2)}</pre>
          {result?.alternatives?.length > 0 && (
            <div className="alternatives-panel">
              <h3>Scanner Alternatives</h3>
              {result.alternatives.map((item) => (
                <div key={item.variant} className="alternative-row">
                  <strong>{item.variant}</strong>
                  <span>{Math.round((item.score || 0) * 100)} score</span>
                  <small>{item.text || "No text"}</small>
                </div>
              ))}
            </div>
          )}
        </section>
      )}

      {view === "calibration" && (
        <section className="calibration-layout">
          <div className="result-card">
            <h2>Calibration Mode</h2>
            <p>Scan a known Braille sample so the app can learn dot size, spacing, row height, and cell spacing for this demo session.</p>
            <label className="text-field">
              Known sample text
              <input value={expectedText} onChange={(event) => setExpectedText(event.target.value)} />
            </label>
            <div className="button-row">
              <button type="button" className="primary" onClick={calibrateFromCurrent}>
                <Settings2 size={20} />
                Calibrate Current Image
              </button>
              <label className="file-button">
                <Upload size={20} />
                Upload + Scan First
                <input type="file" accept="image/*" onChange={scanUpload} />
              </label>
            </div>
          </div>
          <pre className="json-panel">{JSON.stringify(calibration || { message: "No calibration profile yet." }, null, 2)}</pre>
        </section>
      )}

      {view === "samples" && (
        <section className="sample-layout">
          <div className="debug-header">
            <div>
              <h2>Sample Demo View</h2>
              <p>Built-in validation set.</p>
            </div>
          </div>
          <div className="sample-grid">
            {samples.map((sample) => (
              <button key={sample.file} type="button" className="sample-card" onClick={() => scanSample(sample)}>
                <img src={`/sample-images/${sample.file}`} alt={`${sample.expected} sample`} />
                <span>{sample.expected}</span>
                <small>{sample.type} · {sample.notes}</small>
              </button>
            ))}
          </div>
        </section>
      )}

      <section className="settings-panel" aria-label="Voice settings">
        <div>
          <Mic size={20} aria-hidden="true" />
          <span>Speech voice: {preferredVoice?.name || `default ${languageMeta?.speech_locale || "voice"}`}</span>
        </div>
        <select value={selectedVoiceName} onChange={(event) => setSelectedVoiceName(event.target.value)} aria-label="Speech voice">
          <option value="">Best available voice</option>
          {(languageVoices.length ? languageVoices : voices).map((voice) => (
            <option key={`${voice.name}-${voice.lang}`} value={voice.name}>
              {voice.name} ({voice.lang})
            </option>
          ))}
        </select>
        <label>
          Rate
          <input type="range" min="0.6" max="1.4" step="0.05" value={rate} onChange={(event) => setRate(event.target.value)} />
        </label>
        <label>
          Pitch
          <input type="range" min="0.7" max="1.3" step="0.05" value={pitch} onChange={(event) => setPitch(event.target.value)} />
        </label>
        <label>
          Volume
          <input type="range" min="0" max="1" step="0.05" value={volume} onChange={(event) => setVolume(event.target.value)} />
        </label>
      </section>
    </main>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ImagePanel({ title, src }) {
  return (
    <figure className="image-panel">
      <figcaption>{title}</figcaption>
      {src ? <img src={src} alt={title} /> : <div className="empty-image">Run a debug scan to view this layer.</div>}
    </figure>
  );
}

createRoot(document.getElementById("root")).render(<App />);
