SUPPORTED_LANGUAGES = [
    {"code": "en", "name": "English", "speech_locale": "en-US", "mode": "source"},
    {"code": "hi", "name": "Hindi", "speech_locale": "hi-IN", "mode": "built-in-demo"},
    {"code": "es", "name": "Spanish", "speech_locale": "es-ES", "mode": "built-in-demo"},
    {"code": "fr", "name": "French", "speech_locale": "fr-FR", "mode": "built-in-demo"},
]

PHRASE_TRANSLATIONS = {
    "hello world": {
        "hi": "नमस्ते दुनिया",
        "es": "hola mundo",
        "fr": "bonjour le monde",
    },
    "hello": {
        "hi": "नमस्ते",
        "es": "hola",
        "fr": "bonjour",
    },
    "braille": {
        "hi": "ब्रेल",
        "es": "braille",
        "fr": "braille",
    },
    "accessibility": {
        "hi": "सुगम्यता",
        "es": "accesibilidad",
        "fr": "accessibilité",
    },
}

WORD_TRANSLATIONS = {
    "hi": {
        "hello": "नमस्ते",
        "world": "दुनिया",
        "braille": "ब्रेल",
        "text": "पाठ",
        "scan": "स्कैन",
        "reader": "रीडर",
        "accessibility": "सुगम्यता",
    },
    "es": {
        "hello": "hola",
        "world": "mundo",
        "braille": "braille",
        "text": "texto",
        "scan": "escaneo",
        "reader": "lector",
        "accessibility": "accesibilidad",
    },
    "fr": {
        "hello": "bonjour",
        "world": "monde",
        "braille": "braille",
        "text": "texte",
        "scan": "scan",
        "reader": "lecteur",
        "accessibility": "accessibilité",
    },
}

GUIDANCE_TRANSLATIONS = {
    "Too blurry": {"hi": "छवि धुंधली है", "es": "La imagen está borrosa", "fr": "L'image est floue"},
    "Lighting is low": {"hi": "रोशनी कम है", "es": "La iluminación es baja", "fr": "La lumière est faible"},
    "Ready to scan": {"hi": "स्कैन के लिए तैयार", "es": "Listo para escanear", "fr": "Prêt à scanner"},
    "Braille detected": {"hi": "ब्रेल मिला", "es": "Braille detectado", "fr": "Braille détecté"},
    "Hold steady": {"hi": "स्थिर रखें", "es": "Mantén estable", "fr": "Restez stable"},
    "Reading Braille now": {"hi": "अब ब्रेल पढ़ रहा है", "es": "Leyendo braille ahora", "fr": "Lecture du braille"},
}


def get_language(code: str) -> dict:
    return next((lang for lang in SUPPORTED_LANGUAGES if lang["code"] == code), SUPPORTED_LANGUAGES[0])


def translate_text(text: str, language: str) -> str:
    language = language or "en"
    if language == "en":
        return text
    normalized = " ".join(text.lower().split())
    if normalized in PHRASE_TRANSLATIONS and language in PHRASE_TRANSLATIONS[normalized]:
        return PHRASE_TRANSLATIONS[normalized][language]
    dictionary = WORD_TRANSLATIONS.get(language, {})
    translated_words = [dictionary.get(word.strip(".,!?;:"), word) for word in normalized.split()]
    return " ".join(translated_words)


def translate_guidance(message: str, language: str) -> str:
    if language == "en":
        return message
    return GUIDANCE_TRANSLATIONS.get(message, {}).get(language, message)
