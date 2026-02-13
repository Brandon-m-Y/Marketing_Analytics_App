import io
import csv
import pandas as pd

def detect_delimiter(sample: str) -> str:
    """Attempt to detect delimiter from a text sample."""
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "|", "\t"])
        return dialect.delimiter
    except Exception:
        return ","


def load_csv(file_bytes: bytes, delimiter: str | None = None) -> pd.DataFrame:
    """Load CSV/TXT bytes with optional delimiter override."""
    text = file_bytes.decode("utf-8", errors="replace")
    sample = text[:4096]

    if delimiter is None:
        delimiter = detect_delimiter(sample)

    return pd.read_csv(
        io.StringIO(text),
        sep=delimiter,
        engine="python",
        encoding_errors="replace",
        on_bad_lines="warn"
    )


def load_excel(file_bytes: bytes) -> pd.DataFrame:
    """Load Excel bytes."""
    return pd.read_excel(io.BytesIO(file_bytes))
