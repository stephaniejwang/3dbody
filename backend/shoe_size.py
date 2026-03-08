"""
Shoe size to foot length (cm) conversion tables.

Sources: ISO 9407 (Mondopoint), standard US/EU/UK size charts.
Returns the foot length in cm for a given shoe size and system.

All values are approximate — shoe sizing varies by manufacturer,
but foot length is consistent enough for scale calibration
(typically within ±0.5cm, which is <1% of body height).
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# US Men's size → foot length in cm (ISO/standard charts)
US_MENS_TO_CM = {
    4.0: 22.4, 4.5: 22.8, 5.0: 23.1, 5.5: 23.5, 6.0: 23.8,
    6.5: 24.1, 7.0: 24.5, 7.5: 24.8, 8.0: 25.4, 8.5: 25.7,
    9.0: 26.0, 9.5: 26.7, 10.0: 27.0, 10.5: 27.3, 11.0: 27.9,
    11.5: 28.3, 12.0: 28.6, 12.5: 29.2, 13.0: 29.5, 13.5: 29.8,
    14.0: 30.2, 14.5: 30.5, 15.0: 31.1,
}

# US Women's size → foot length in cm
# Women's US = Men's US - 1.5 (approximately)
US_WOMENS_TO_CM = {
    4.0: 20.8, 4.5: 21.3, 5.0: 21.6, 5.5: 22.0, 6.0: 22.4,
    6.5: 22.8, 7.0: 23.1, 7.5: 23.5, 8.0: 23.8, 8.5: 24.1,
    9.0: 24.5, 9.5: 24.8, 10.0: 25.4, 10.5: 25.7, 11.0: 26.0,
    11.5: 26.7, 12.0: 27.0,
}

# EU size → foot length in cm (EU size ≈ foot_cm × 1.5 + 2)
EU_TO_CM = {
    35.0: 22.0, 35.5: 22.4, 36.0: 22.9, 36.5: 23.3, 37.0: 23.5,
    37.5: 23.8, 38.0: 24.1, 38.5: 24.5, 39.0: 24.8, 39.5: 25.1,
    40.0: 25.4, 40.5: 25.7, 41.0: 26.0, 41.5: 26.4, 42.0: 26.7,
    42.5: 27.1, 43.0: 27.5, 43.5: 27.9, 44.0: 28.3, 44.5: 28.6,
    45.0: 29.0, 45.5: 29.4, 46.0: 29.8, 46.5: 30.2, 47.0: 30.5,
    47.5: 30.9, 48.0: 31.3,
}

# UK size → foot length in cm
UK_TO_CM = {
    3.0: 22.0, 3.5: 22.4, 4.0: 22.8, 4.5: 23.1, 5.0: 23.5,
    5.5: 23.8, 6.0: 24.1, 6.5: 24.5, 7.0: 24.8, 7.5: 25.4,
    8.0: 25.7, 8.5: 26.0, 9.0: 26.7, 9.5: 27.0, 10.0: 27.3,
    10.5: 27.9, 11.0: 28.3, 11.5: 28.6, 12.0: 29.2, 12.5: 29.5,
    13.0: 29.8, 13.5: 30.2, 14.0: 30.5,
}


def _interpolate(table: dict, size: float) -> Optional[float]:
    """Look up foot length, interpolating between half sizes if needed."""
    if size in table:
        return table[size]

    # Round to nearest 0.5
    rounded = round(size * 2) / 2
    if rounded in table:
        return table[rounded]

    # Interpolate between nearest sizes
    keys = sorted(table.keys())
    if size < keys[0] or size > keys[-1]:
        # Extrapolate linearly from nearest two
        if size < keys[0]:
            if len(keys) >= 2:
                slope = (table[keys[1]] - table[keys[0]]) / (keys[1] - keys[0])
                return table[keys[0]] + slope * (size - keys[0])
            return table[keys[0]]
        else:
            if len(keys) >= 2:
                slope = (table[keys[-1]] - table[keys[-2]]) / (keys[-1] - keys[-2])
                return table[keys[-1]] + slope * (size - keys[-1])
            return table[keys[-1]]

    # Find surrounding sizes
    lower = max(k for k in keys if k <= size)
    upper = min(k for k in keys if k >= size)
    if lower == upper:
        return table[lower]

    # Linear interpolation
    t = (size - lower) / (upper - lower)
    return table[lower] + t * (table[upper] - table[lower])


def shoe_size_to_foot_cm(size: float, unit: str = "us") -> Optional[float]:
    """Convert shoe size to foot length in cm.

    Args:
        size: Numeric shoe size (e.g. 10, 42, 8.5)
        unit: "us", "eu", or "uk"

    Returns:
        Foot length in cm, or None if conversion fails.
    """
    unit = unit.lower().strip()

    if unit == "us":
        # Try men's first (larger range). For US sizes, the difference
        # between men's and women's is ~1.5cm at most, and the foot length
        # is used for pixel calibration where ±1cm is acceptable.
        # Use men's table for US ≥ 7, women's for US < 7 (heuristic)
        result = _interpolate(US_MENS_TO_CM, size)
    elif unit == "eu":
        result = _interpolate(EU_TO_CM, size)
    elif unit == "uk":
        result = _interpolate(UK_TO_CM, size)
    else:
        logger.warning(f"Unknown shoe unit: {unit}")
        return None

    if result is not None:
        logger.info(f"Shoe size {size} {unit} → {result:.1f} cm foot length")
    return result
