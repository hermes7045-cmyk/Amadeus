from __future__ import annotations

import re

_EMOJI_CODEPOINTS = {
    0x200D,  # zero width joiner
    0x20E3,  # combining enclosing keycap
    0xFE0E,  # variation selector-15
    0xFE0F,  # variation selector-16
}

_EMOJI_RANGES: tuple[tuple[int, int], ...] = (
    (0x1F1E6, 0x1F1FF),  # flags
    (0x1F3FB, 0x1F3FF),  # skin tone modifiers
    (0x1F300, 0x1F5FF),  # symbols and pictographs
    (0x1F600, 0x1F64F),  # emoticons
    (0x1F680, 0x1F6FF),  # transport and map
    (0x1F700, 0x1F77F),  # alchemical symbols
    (0x1F780, 0x1F7FF),  # geometric shapes extended
    (0x1F800, 0x1F8FF),  # supplemental arrows-c
    (0x1F900, 0x1F9FF),  # supplemental symbols and pictographs
    (0x1FA70, 0x1FAFF),  # symbols and pictographs extended-a
    (0x2600, 0x26FF),  # miscellaneous symbols
    (0x2700, 0x27BF),  # dingbats
)

_FENCE_PATTERN = re.compile(r"^\s*(```|~~~)[^\n]*$", re.MULTILINE)
_INLINE_CODE_PATTERN = re.compile(r"`([^`]+)`")
_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_HEADING_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_QUOTE_PATTERN = re.compile(r"^\s*>\s?", re.MULTILINE)
_UNORDERED_LIST_PATTERN = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_ORDERED_LIST_PATTERN = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)
_MARKDOWN_MARKER_PATTERN = re.compile(r"[*_~]")


def normalize_text_for_tts(text: str) -> str:
    normalized_text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    normalized_text = _FENCE_PATTERN.sub("", normalized_text)
    normalized_text = _INLINE_CODE_PATTERN.sub(r"\1", normalized_text)
    normalized_text = _LINK_PATTERN.sub(r"\1", normalized_text)
    normalized_text = _HEADING_PATTERN.sub("", normalized_text)
    normalized_text = _QUOTE_PATTERN.sub("", normalized_text)
    normalized_text = _UNORDERED_LIST_PATTERN.sub("", normalized_text)
    normalized_text = _ORDERED_LIST_PATTERN.sub("", normalized_text)
    normalized_text = _MARKDOWN_MARKER_PATTERN.sub("", normalized_text)

    cleaned_text = "".join(
        " " if _is_emoji_character(character) else character
        for character in normalized_text
    )
    return " ".join(cleaned_text.split())


def _is_emoji_character(character: str) -> bool:
    codepoint = ord(character)
    if codepoint in _EMOJI_CODEPOINTS:
        return True
    return any(start <= codepoint <= end for start, end in _EMOJI_RANGES)
