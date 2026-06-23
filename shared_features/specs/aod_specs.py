import re

AOD_PATTERNS = [
    r"\baod\b",
    r"\bmaiac\b",
    r"\boptical[_-]?depth\b",
    r"\bod(047|055|066)\b",
    r"\boptical[_-]?depth[_-]?(047|055|066)\b",
    r"\bdeep[_-]?blue\b",
    # Add missing patterns for actual AOD features:
    r"\baod[_-]?uncertainty\b",
    r"\bqa[_-]?(cloudmask|adjacency|aod|n_pixels)\b",
    r"\baod[_-]?(total|files|window)",
    r"^optical_depth_\d+$",  # exact patterns like optical_depth_047
]

def is_aod_feature(name: str) -> bool:
    n = name.lower()
    return any(re.search(p, n) for p in AOD_PATTERNS)