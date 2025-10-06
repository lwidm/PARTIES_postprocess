import numpy as np
from typing import Optional, List
import io
import re


def load_columns_from_txt_numpy(
    path: str, split_chars: str = ",;", comment_chars: str = "#%"
) -> List[np.ndarray]:
    """
    Load numeric columns from a text file and return a list of NumPy 1-D arrays.

    Args:
        path: path to file
        split_chars: characters treated as extra delimiters (each replaced by whitespace)
        comment_chars: characters that start comments (full-line or inline)

    Returns:
        List of 1-D numpy arrays, one per column.
    """
    esc_split: Optional[str] = re.escape(split_chars) if split_chars else ""
    esc_comment: Optional[str] = re.escape(comment_chars) if comment_chars else ""
    pattern_split: Optional[str] = f"[{esc_split}]+" if esc_split else None
    pattern_comment: Optional[str] = f"[{esc_comment}].*" if esc_comment else None

    lines: list[str] = []
    with open(path, "r", encoding="utf-8") as file:
        line: str
        for line in file:
            line = line.strip()
            if not line:
                continue
            # remove comments
            if pattern_comment is not None:
                line = re.sub(pattern_comment, "", line, count=1).strip()
                if line == "":
                    continue
            # split into columns
            if pattern_split is not None:
                line = re.sub(pattern_split, " ", line)
            line = " ".join(line.split())
            if line != "":
                lines.append(line)

    if len(lines) == 0:
        raise ValueError("no data found")

    data: np.ndarray = np.genfromtxt(io.StringIO("\n".join(lines)), dtype=np.float64, delimiter=None)
    if data.size == 0:
        raise ValueError("no numeric data parsed")
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return [data[:, i] for i in range(data.shape[1])]


