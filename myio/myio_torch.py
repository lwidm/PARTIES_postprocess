import torch
from myio import load_columns_from_txt

def load_columns_from_txt_torch(filename: str, split_chars: str = ",;", comment_chars: str = "#%"):
    """
    Same as io/io.py -> load_columns_from_txt(...) but returns a list of PyTorch tensors.
    """
    cols = load_columns_from_txt(filename, split_chars=split_chars, comment_chars=comment_chars)
    return [torch.from_numpy(np.ascontiguousarray(c)) for c in cols]
