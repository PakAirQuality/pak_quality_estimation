def roll_name(var: str, stat: str, window_days: int) -> str:
    return f"{var}_roll_{stat}_{int(window_days)}d"

def lag_name(var: str, lag_days: int) -> str:
    return f"{var}_lag_{int(lag_days)}d"

def daily_name(var: str, stat: str) -> str:
    return f"{var}_daily_{stat}"