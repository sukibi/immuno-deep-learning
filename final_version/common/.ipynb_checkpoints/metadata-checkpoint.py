"""
common/metadata.py — shared spreadsheet/label helpers for the data-prep and
clinical scripts.

Extracted from copies duplicated across build_clinical_v2, build_clinical_additional,
build_nifti_manifest, dicom_to_nifti, make_manifest and halo_check: the
strict-encoding table reader, the accent/case/punctuation-insensitive column
finder, and the French<->English disease map. Your own comments flag this exact
area as a silent-corruption risk (cp1252 mojibake; a slug that must match the
one-hot naming EXACTLY) — one tested copy removes the chance of the copies drifting.

TWO NORMALISERS, deliberately different — do not mix them:
  col_key(...)     accent- AND punctuation-stripped, alnum only. For matching a
                   COLUMN NAME ("Nom série2" == "nom_serie_2"). Never use it to
                   name a feature.
  onehot_slug(...) lowercase, non-alnum -> "_", accents KEPT (so they become "_").
                   For NAMING a one-hot feature column, matching build_clinical_v2
                   exactly ("Leucémie aigu" -> "leuc_mie_aigu"). Getting this wrong
                   silently zero-fills the immunodepression columns.
"""
from __future__ import annotations
import re
import unicodedata

import pandas as pd

FRENCH2EN = {"aspergillose": "aspergillosis", "tuberculose": "tuberculosis",
             "nocardiose": "nocardiosis", "mucormycose": "mucormycosis"}
EN2FRENCH = {"aspergillosis": "Aspergillose", "tuberculosis": "Tuberculose",
             "nocardiosis": "Nocardiose", "mucormycosis": "Mucormycose"}
# short forms seen in filenames / sheets -> canonical English
TOKEN_ALIASES = {
    "aspergillose": "aspergillosis", "aspergillosis": "aspergillosis", "asper": "aspergillosis",
    "tuberculose": "tuberculosis", "tuberculosis": "tuberculosis", "tb": "tuberculosis",
    "nocardiose": "nocardiosis", "nocardiosis": "nocardiosis", "nocardia": "nocardiosis",
    "mucormycose": "mucormycosis", "mucormycosis": "mucormycosis", "mucor": "mucormycosis",
}
MOJIBAKE = ("\u00c3", "\u00c2")   # tell-tale of a cp1252-decoded UTF-8 file


def _strip_accents(s):
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c))


def col_key(s):
    """Accent-, case-, BOM- and punctuation-insensitive key for column matching."""
    return re.sub(r"[^a-z0-9]", "", _strip_accents(s).lower())


def find_col(df, *names):
    """Return the actual column in df matching any of `names` fuzzily, else None."""
    lut = {col_key(c): c for c in df.columns}
    for n in names:
        if col_key(n) in lut:
            return lut[col_key(n)]
    return None


def read_table_strict(path, sep=None):
    """Read a CSV/TSV, trying STRICT encodings first. cp1252/latin-1 decode almost
    any byte sequence without raising, so putting them first silently yields
    mojibake; utf-8 fails loudly instead. Returns (df, encoding)."""
    last = None
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, sep=sep, engine="python", encoding=enc)
        except (UnicodeDecodeError, ValueError) as e:
            last = e
            continue
        df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
        if any(m in c for c in df.columns for m in MOJIBAKE):
            last = f"mojibake under {enc}"
            continue
        return df, enc
    raise SystemExit(f"could not decode {path} ({last})")


def to_english(value):
    """A French or English class name -> canonical English, or None."""
    v = _strip_accents(value).strip().lower()
    if v in FRENCH2EN:
        return FRENCH2EN[v]
    return v if v in FRENCH2EN.values() else None


def label_from_token(text):
    """Match any _/-/space-separated token (or a substring) of `text` against the
    alias table -> canonical English label, or None. For parsing file stems."""
    norm = _strip_accents(text).lower()
    for tok in re.split(r"[_\-\s]+", norm):
        if tok in TOKEN_ALIASES:
            return TOKEN_ALIASES[tok]
    for alias, lab in TOKEN_ALIASES.items():
        if alias in norm:
            return lab
    return None


def onehot_slug(value):
    """Lowercase, non-alphanumeric runs -> '_', accents KEPT (become '_'). Must
    match build_clinical_v2's one-hot naming EXACTLY. NOT for column matching."""
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")