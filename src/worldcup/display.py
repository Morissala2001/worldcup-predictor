"""Small presentation helpers shared by the CLI and the web app."""

from __future__ import annotations

# ISO 3166-1 alpha-2 codes, used to build flag emoji. Teams not listed get a white flag.
ISO_CODES = {
    "Algeria": "DZ", "Argentina": "AR", "Australia": "AU", "Austria": "AT", "Belgium": "BE",
    "Brazil": "BR", "Cameroon": "CM", "Canada": "CA", "Chile": "CL", "Colombia": "CO",
    "Croatia": "HR", "Czech Republic": "CZ", "Denmark": "DK", "Ecuador": "EC", "Egypt": "EG",
    "France": "FR", "Germany": "DE", "Ghana": "GH", "Greece": "GR", "Iran": "IR",
    "Italy": "IT", "Ivory Coast": "CI", "Japan": "JP", "Mexico": "MX", "Morocco": "MA",
    "Netherlands": "NL", "Nigeria": "NG", "Norway": "NO", "Peru": "PE", "Poland": "PL",
    "Portugal": "PT", "Saudi Arabia": "SA", "Senegal": "SN", "Serbia": "RS", "South Korea": "KR",
    "Spain": "ES", "Sweden": "SE", "Switzerland": "CH", "Tunisia": "TN", "Turkey": "TR",
    "Ukraine": "UA", "United States": "US", "Uruguay": "UY",
}

_TAG_BASE = 0xE0000
_BLACK_FLAG = "\U0001F3F4"
_CANCEL_TAG = "\U000E007F"


def flag(team: str) -> str:
    """Flag emoji of a team (England and Scotland have their own), or a white flag if unknown."""
    if team == "England":
        return _BLACK_FLAG + "".join(chr(_TAG_BASE + ord(c)) for c in "gbeng") + _CANCEL_TAG
    if team == "Scotland":
        return _BLACK_FLAG + "".join(chr(_TAG_BASE + ord(c)) for c in "gbsct") + _CANCEL_TAG
    code = ISO_CODES.get(team)
    if code is None:
        return "\U0001F3F3️"
    return "".join(chr(0x1F1E6 + ord(c) - ord("A")) for c in code)


def label(team: str) -> str:
    """'🇫🇷 France'."""
    return f"{flag(team)} {team}"


def percent_split(prob_a: float) -> tuple[int, int]:
    """Whole percentages for team A and team B that always add up to 100.

    A is rounded exactly like `f"{prob_a:.0%}"`, the way the original notebook printed it, and B
    is the remainder. Rounding both independently shows 92 % + 7 % when the odds are 92.5 / 7.5.
    """
    percent_a = int(f"{prob_a:.0%}"[:-1])
    return percent_a, 100 - percent_a
