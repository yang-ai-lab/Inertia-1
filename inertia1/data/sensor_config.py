from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


def _as_list(x: Any) -> Optional[list[str]]:
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [str(s).strip() for s in x if str(s).strip()]
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        # allow comma-separated string
        if "," in s:
            return [p.strip() for p in s.split(",") if p.strip()]
        return [s]
    return [str(x).strip()]


def _norm_col(s: str) -> str:
    # normalize for fuzzy matching: lowercase and strip separators
    return "".join(ch for ch in s.lower() if ch.isalnum())


@dataclass(frozen=True)
class SensorColumnSpec:
    sensor: str
    x: str
    y: str
    z: str

    def as_list(self) -> list[str]:
        return [self.x, self.y, self.z]


def _prefixes_for_sensor(sensor: str) -> list[str]:
    s = sensor.strip().lower()
    if s in {"accel", "accelerometer", "accelerometre", "acc"}:
        return ["", "accel", "accelerometer", "acc"]
    if s in {"gyro", "gyroscope", "gyr"}:
        # include short 'g' prefix for GX/GY/GZ
        return ["gyro", "gyroscope", "gyr", "g"]
    if s in {"mag", "magnetometer"}:
        # include short 'm' prefix for MX/MY/MZ
        return ["mag", "magnetometer", "m"]
    return [s]


def infer_sensor_columns_from_schema(
    available_columns: Sequence[str],
    sensor_types: Sequence[str],
) -> list[SensorColumnSpec]:
    """Infer tri-axial (X,Y,Z) column triplets for each sensor type.

    This is intentionally conservative: it raises with a helpful message if
    it can't find a complete (X,Y,Z) set for any requested sensor.
    """

    norm_to_original: dict[str, str] = {}
    for c in available_columns:
        norm_to_original[_norm_col(str(c))] = str(c)

    specs: list[SensorColumnSpec] = []
    for sensor in sensor_types:
        prefixes = _prefixes_for_sensor(sensor)

        def find_axis(axis: str) -> Optional[str]:
            ax = axis.lower()
            for pref in prefixes:
                cand = _norm_col(f"{pref}{ax}")
                if cand in norm_to_original:
                    return norm_to_original[cand]
                cand = _norm_col(f"{pref}_{ax}")
                if cand in norm_to_original:
                    return norm_to_original[cand]
                cand = _norm_col(f"{pref}.{ax}")
                if cand in norm_to_original:
                    return norm_to_original[cand]
                cand = _norm_col(f"{pref}{ax.upper()}")
                if cand in norm_to_original:
                    return norm_to_original[cand]
            # also try bare axis (X/Y/Z) for accel-like layouts
            cand = _norm_col(ax)
            return norm_to_original.get(cand)

        cx = find_axis("x")
        cy = find_axis("y")
        cz = find_axis("z")

        missing = [a for a, v in (("X", cx), ("Y", cy), ("Z", cz)) if v is None]
        if missing:
            raise ValueError(
                "Could not infer columns for sensor_type="
                f"'{sensor}'. Missing axes={missing}. "
                f"Available columns (first 50): {list(available_columns)[:50]}"
            )

        specs.append(SensorColumnSpec(sensor=str(sensor), x=cx, y=cy, z=cz))

    return specs


def _find_first_file(data_root: str | Path, file_extension: str) -> Path:
    roots: list[Path]
    data_root_str = str(data_root)
    if "," in data_root_str:
        roots = [Path(p.strip()) for p in data_root_str.split(",") if p.strip()]
    else:
        roots = [Path(data_root_str)]

    for root in roots:
        if not root.exists():
            continue
        # recursive search; stop at first match
        for p in root.glob(f"**/*{file_extension}"):
            if p.is_file():
                return p

    raise FileNotFoundError(
        f"Could not find any '{file_extension}' files under data_root={data_root}"
    )


def resolve_data_columns_and_channels(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve sensor selection config into `data.parquet_columns` + `data.channels`.

    Supported patterns:
      - User sets `data.parquet_columns`: we infer `data.channels` if missing.
      - User sets `data.sensor_types` (or `data.sensors`): we infer parquet columns
        by looking at the schema of the first matching file.

    Notes:
      - `data.axes` keeps its meaning as *axes per sensor after reduction* (1 or 3).
      - Total model input channels becomes:
            axes==3: len(parquet_columns)
            axes==1: len(parquet_columns) / 3
        (requires tri-axial column sets).
    """

    if not isinstance(cfg, dict):
        return cfg

    dcfg = cfg.get("data", {}) or {}

    # Normalize sensor types key.
    sensor_types = _as_list(dcfg.get("sensor_types", None) or dcfg.get("sensors", None))

    parquet_columns = dcfg.get("parquet_columns", None)
    file_extension = str(dcfg.get("file_extension", ".parquet"))
    axes = int(dcfg.get("axes", 3))

    # If explicit columns provided, only fill channels if missing.
    if parquet_columns is not None:
        if isinstance(parquet_columns, str):
            # allow comma-separated string
            parquet_columns = [p.strip() for p in parquet_columns.split(",") if p.strip()]
        if not isinstance(parquet_columns, (list, tuple)):
            raise ValueError("data.parquet_columns must be a list (or null)")
        cols = [str(c) for c in parquet_columns]
        cfg = dict(cfg)
        cfg.setdefault("data", {})
        cfg["data"] = dict(cfg["data"])
        cfg["data"]["parquet_columns"] = cols
        cfg["data"].setdefault("channels", _channels_from_cols(cols, axes))
        return cfg

    if not sensor_types:
        return cfg

    if file_extension != ".parquet":
        raise ValueError(
            "data.sensor_types requires file_extension=.parquet (or set data.parquet_columns explicitly). "
            f"Got file_extension={file_extension}"
        )

    data_root = dcfg.get("data_root", None)
    if not data_root:
        raise ValueError("data.data_root must be set")

    first = _find_first_file(data_root, file_extension)

    import pandas as pd

    df = pd.read_parquet(first)
    avail = list(df.columns)

    specs = infer_sensor_columns_from_schema(avail, sensor_types)
    cols: list[str] = []
    for spec in specs:
        cols.extend(spec.as_list())

    cfg = dict(cfg)
    cfg.setdefault("data", {})
    cfg["data"] = dict(cfg["data"])
    cfg["data"]["parquet_columns"] = cols
    cfg["data"].setdefault("channels", _channels_from_cols(cols, axes))
    cfg["data"].setdefault("resolved_sensor_types", list(sensor_types))
    return cfg


def _channels_from_cols(cols: Sequence[str], axes: int) -> int:
    if axes not in (1, 3):
        # keep backwards compatibility
        return len(cols)

    if axes == 3:
        return len(cols)

    # axes == 1: reduce each tri-axial sensor to 1 channel
    if len(cols) % 3 != 0:
        raise ValueError(
            "data.axes=1 requires tri-axial columns grouped per sensor; "
            f"got len(parquet_columns)={len(cols)} not divisible by 3"
        )
    return len(cols) // 3
