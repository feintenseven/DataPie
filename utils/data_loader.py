import pandas as pd
import os

def load_data(path):

    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        return pd.read_csv(path)

    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    elif ext == ".json":
        return pd.read_json(path)

    else:
        raise ValueError(f"暂不支持的文件类型: {ext}")