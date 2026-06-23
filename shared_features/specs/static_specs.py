import re

STATIC_PATTERNS = [
    # Elevation and coast patterns removed
    r"\bslope\b",
    r"\baspect\b",
    r"\blandcover\b",
    r"\burban\b",
]

def is_static_feature(name: str) -> bool:
    n = name.lower()
    return any(re.search(p, n) for p in STATIC_PATTERNS)