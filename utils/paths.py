import os

def data_path(filename: str):
    """
    Returns absolute path to CSV files in /data folder.
    Works both locally and on Streamlit Cloud.
    """
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "data", filename)
