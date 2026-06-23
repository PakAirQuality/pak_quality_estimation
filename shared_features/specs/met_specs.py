import re

# Put anything you consider "met-like" here.
# These are regex patterns matched against feature column names.
MET_PATTERNS = [
    # Core meteorological variables (exact prefix matches)
    r"^(t2m|d2m|u10|v10|u100|v100|sp|tp|blh|rh|vpd|ws10|ws100|vc|msl|tcc|clr|q|theta)\b",
    # Derived variables  
    r"^(wd10|wd100|dws|dwd|vci|stagnant|highrh|calm\d+)\b",
    # Seasonal flags
    r"^(heating_season_flag|burning_season_flag)$", 
    # Tendency variables
    r"^(blh|mslp|sp|dt)_tend\b",
    # Temporal patterns
    r"_daily_(mean|max|min|sum)$",
    r"_roll(mean|std|min|max)_(\d+)d$",
    r"_lag(\d+)d$",
    r"_anom_(\d+)d$",
    # Wind direction circular statistics
    r"^wd10_(sin|cos|var)(_rm_\d+d)?$",
    # Day-of-year encodings (these are computed in inference)
    r"^doy_(sin|cos)(_\d+)?$",
]

def is_met_feature(name: str) -> bool:
    n = name.lower()
    return any(re.search(p, n) for p in MET_PATTERNS)