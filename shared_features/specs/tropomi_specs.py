import re

TROPOMI_PATTERNS = [
    r"\btropomi\b",
    r"^(no2|so2|co|hcho|o3|ch4|nh3|aai|alh|cloud)_",  # compound prefix patterns
    r"\baer(_|-)?ai\b",
    r"\bqa(_|-)?value\b",
    # Statistical patterns for TROPOMI compounds:
    r"^(no2|so2|co|hcho|o3|ch4|nh3|aai|alh|cloud)_(median|mean|std|min|max|p\d+|iqr|mad|range|cv|skew|kurt|n_pixels|window_size_used|window_coverage|radius_km_used|file_available|qa_pass_fraction|qa_available|qa_threshold|scale_factor|add_offset)\b",
    r"^(no2|so2|co|hcho|o3|ch4|nh3|aai|alh|cloud)_delta_",
    r"^(no2|so2|co|hcho|o3|ch4|nh3|aai|alh|cloud)_.*_r(05|1|2)\b",  # multiscale patterns
]

def is_tropomi_feature(name: str) -> bool:
    n = name.lower()
    return any(re.search(p, n) for p in TROPOMI_PATTERNS)