"""JSON-safe conversion of pipeline records (numpy scalars/arrays, NaN -> None)."""
import numpy as np


def num(x):
    """6 significant figures (fluxes are ~1e-13); NaN/inf -> None."""
    x = float(x)
    return float(f'{x:.6g}') if np.isfinite(x) else None


def serialize(v):
    if isinstance(v, np.ndarray):
        return [num(x) for x in v.ravel()]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return num(v)
    if isinstance(v, dict):
        return {k: serialize(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [serialize(x) for x in v]
    return v


def serialize_record(record):
    return {k: serialize(v) for k, v in record.items()}
