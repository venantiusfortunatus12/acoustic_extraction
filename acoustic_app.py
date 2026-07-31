#!/usr/bin/env python3
"""
Acoustic extraction Streamlit app — HuggingFace Spaces compatible.
Upload paired audio + TextGrid files, extract f0 / intensity / formants.
"""

import math
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import parselmouth
import streamlit as st
import tgt
from io import BytesIO

st.title("Segment Acoustic Extraction")
st.markdown("Sociophonetics & Prosody Research — Seongwoo Kang")

# ── Helpers (same logic as CLI script) ───────────────────────────────────────

def read_textgrid(path: Path) -> tgt.core.TextGrid:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return tgt.read_textgrid(str(path), encoding=encoding)
        except UnicodeDecodeError:
            continue
    return tgt.read_textgrid(str(path))


def get_tier_case_insensitive(textgrid: tgt.core.TextGrid, tier_name: str) -> Any:
    wanted = tier_name.lower()
    for tier in textgrid.tiers:
        if tier.name.lower() == wanted:
            return tier
    available = ", ".join(t.name for t in textgrid.tiers)
    raise ValueError(f"Tier '{tier_name}' not found. Available: {available}")


def safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def rounded(value: float | None, ndigits: int = 3) -> float | None:
    return None if value is None else round(value, ndigits)


def value_at_time(obj: Any, method_name: str, *args: Any) -> float | None:
    try:
        return safe_float(getattr(obj, method_name)(*args))
    except Exception:
        return None


def extract_from_pair(
    audio_path: Path,
    textgrid_path: Path,
    tier_name: str,
    pitch_floor: float,
    pitch_ceiling: float,
    max_formant: float,
    n_formants: int,
    skip_empty: bool,
) -> list[dict[str, Any]]:
    sound     = parselmouth.Sound(str(audio_path))
    textgrid  = read_textgrid(textgrid_path)
    tier      = get_tier_case_insensitive(textgrid, tier_name)

    pitch     = sound.to_pitch_ac(time_step=0.0, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    intensity = sound.to_intensity()
    formant   = sound.to_formant_burg(time_step=0.0, max_number_of_formants=n_formants, maximum_formant=max_formant)

    rows: list[dict[str, Any]] = []

    for idx, interval in enumerate(tier.intervals, start=1):
        label = interval.text.strip()
        if skip_empty and not label:
            continue

        start_s    = float(interval.start_time)
        end_s      = float(interval.end_time)
        duration_s = end_s - start_s
        if duration_s <= 0:
            continue

        mid = start_s + duration_s * 0.5

        rows.append({
            "file":             audio_path.name,
            "textgrid":         textgrid_path.name,
            "tier":             tier.name,
            "interval_index":   idx,
            "label":            label,
            "start_s":          rounded(start_s),
            "end_s":            rounded(end_s),
            "midpoint_s":       rounded(mid),
            "duration_ms":      rounded(duration_s * 1000, 2),
            "f0_Hz_50":         rounded(value_at_time(pitch,     "get_value_at_time", mid),    2),
            "intensity_dB_50":  rounded(value_at_time(intensity, "get_value",         mid),    2),
            "F1_Hz_50":         rounded(value_at_time(formant,   "get_value_at_time", 1, mid), 2),
            "F2_Hz_50":         rounded(value_at_time(formant,   "get_value_at_time", 2, mid), 2),
            "F3_Hz_50":         rounded(value_at_time(formant,   "get_value_at_time", 3, mid), 2),
        })

    return rows


# ── 1. File Upload ────────────────────────────────────────────────────────────
st.markdown("Upload paired audio and TextGrid files. Files are matched by filename stem.")

audio_files    = st.file_uploader(
    "Audio files (.wav / .aiff / .flac)",
    type=["wav", "WAV", "aiff", "aif", "AIFF", "flac", "FLAC"],
    accept_multiple_files=True
)
textgrid_files = st.file_uploader(
    "TextGrid files (.TextGrid)",
    type=["TextGrid", "textgrid"],
    accept_multiple_files=True
)

if not audio_files or not textgrid_files:
    st.info("Upload at least one audio file and its matching TextGrid to begin.")
    st.stop()

# ── 2. Sidebar parameters ─────────────────────────────────────────────────────
st.sidebar.header("Extraction parameters")

tier_name     = st.sidebar.text_input("Tier name", value="segment")
pitch_floor   = st.sidebar.number_input("Pitch floor (Hz)",   value=75.0,   step=5.0)
pitch_ceiling = st.sidebar.number_input("Pitch ceiling (Hz)", value=500.0,  step=10.0)
max_formant   = st.sidebar.number_input("Max formant (Hz)",   value=5500.0, step=100.0)
n_formants    = st.sidebar.slider("Number of formants", 3, 6, 5, 1)
keep_empty    = st.sidebar.checkbox("Keep empty intervals", value=False)

# ── 3. Match pairs by stem ────────────────────────────────────────────────────
audio_map    = {Path(f.name).stem: f for f in audio_files}
textgrid_map = {Path(f.name).stem: f for f in textgrid_files}

matched   = sorted(set(audio_map) & set(textgrid_map))
unmatched = sorted((set(audio_map) | set(textgrid_map)) - set(matched))

st.markdown(f"**{len(matched)} matched pair(s)** found.")
if unmatched:
    st.warning(f"No match for: {', '.join(unmatched)}")

if not matched:
    st.error("No matched audio/TextGrid pairs. Check that filenames share the same stem.")
    st.stop()

with st.expander("Matched pairs"):
    for stem in matched:
        st.write(f"{audio_map[stem].name}  +  {textgrid_map[stem].name}")

# ── 4. Run extraction ─────────────────────────────────────────────────────────
if st.button("Run extraction"):
    all_rows: list[dict] = []
    log_lines: list[str] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        for stem in matched:
            # Write uploads to disk (parselmouth needs file paths)
            audio_up = audio_map[stem]
            tg_up    = textgrid_map[stem]

            audio_path = tmp / audio_up.name
            tg_path    = tmp / tg_up.name

            audio_path.write_bytes(audio_up.read())
            tg_path.write_bytes(tg_up.read())

            try:
                rows = extract_from_pair(
                    audio_path    = audio_path,
                    textgrid_path = tg_path,
                    tier_name     = tier_name,
                    pitch_floor   = pitch_floor,
                    pitch_ceiling = pitch_ceiling,
                    max_formant   = max_formant,
                    n_formants    = n_formants,
                    skip_empty    = not keep_empty,
                )
                all_rows.extend(rows)
                log_lines.append(f"[ok]    {audio_up.name} — {len(rows)} interval(s)")
            except Exception as exc:
                log_lines.append(f"[error] {audio_up.name} — {exc}")

    with st.expander("Extraction log"):
        st.text("\n".join(log_lines))

    if not all_rows:
        st.error("No data extracted. Check tier name and file contents.")
        st.stop()

    df = pd.DataFrame(all_rows)
    st.success(f"Extraction complete — {len(df):,} interval(s) from {len(matched)} file(s)")

    # Preview
    st.subheader("Preview")
    st.dataframe(df, use_container_width=True)

    # Download
    buf = BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    st.download_button(
        "Download Excel",
        buf.getvalue(),
        "segment_acoustics.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    buf_csv = BytesIO()
    df.to_csv(buf_csv, index=False)
    st.download_button(
        "Download CSV",
        buf_csv.getvalue(),
        "segment_acoustics.csv",
        "text/csv"
    )
