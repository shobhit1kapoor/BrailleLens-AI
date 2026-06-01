LETTER_TO_PATTERN = {
    "a": "1",
    "b": "12",
    "c": "14",
    "d": "145",
    "e": "15",
    "f": "124",
    "g": "1245",
    "h": "125",
    "i": "24",
    "j": "245",
    "k": "13",
    "l": "123",
    "m": "134",
    "n": "1345",
    "o": "135",
    "p": "1234",
    "q": "12345",
    "r": "1235",
    "s": "234",
    "t": "2345",
    "u": "136",
    "v": "1236",
    "w": "2456",
    "x": "1346",
    "y": "13456",
    "z": "1356",
}

PATTERN_TO_CHAR = {pattern: char for char, pattern in LETTER_TO_PATTERN.items()}

PUNCTUATION = {
    "2": ",",
    "23": ";",
    "25": ":",
    "256": ".",
    "235": "!",
    "236": "?",
    "3": "'",
    "36": "-",
}

NUMBER_SIGN = "3456"
CAPITAL_SIGN = "6"
SPACE_PATTERN = ""

NUMBER_MAP = {
    "1": "1",
    "12": "2",
    "14": "3",
    "145": "4",
    "15": "5",
    "124": "6",
    "1245": "7",
    "125": "8",
    "24": "9",
    "245": "0",
}


def pattern_to_char(pattern: str, number_mode: bool = False, capitalize: bool = False) -> tuple[str, bool, bool]:
    """Translate a single Grade 1 Braille pattern.

    Returns char, new_number_mode, new_capitalize.
    """
    normalized = "".join(sorted(pattern))
    if normalized == SPACE_PATTERN:
        return " ", False, False
    if normalized == NUMBER_SIGN:
        return "", True, False
    if normalized == CAPITAL_SIGN:
        return "", number_mode, True
    if number_mode and normalized in NUMBER_MAP:
        return NUMBER_MAP[normalized], True, False
    if normalized in PATTERN_TO_CHAR:
        char = PATTERN_TO_CHAR[normalized]
        if capitalize:
            char = char.upper()
        return char, number_mode, False
    if normalized in PUNCTUATION:
        return PUNCTUATION[normalized], False, False
    return "?", False, False


def translate_patterns(patterns: list[str]) -> str:
    number_mode = False
    capitalize = False
    output: list[str] = []
    for pattern in patterns:
        char, number_mode, capitalize = pattern_to_char(pattern, number_mode, capitalize)
        output.append(char)
    return "".join(output).replace("  ", " ").strip()


def text_to_patterns(text: str) -> list[str]:
    patterns: list[str] = []
    for char in text.lower():
        if char == " ":
            patterns.append(SPACE_PATTERN)
        elif char in LETTER_TO_PATTERN:
            patterns.append(LETTER_TO_PATTERN[char])
        elif char in {v: k for k, v in PUNCTUATION.items()}:
            reverse = {v: k for k, v in PUNCTUATION.items()}
            patterns.append(reverse[char])
    return patterns
