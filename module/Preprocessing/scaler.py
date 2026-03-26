import pandas as pd
from sklearn.preprocessing import StandardScaler


def standardize(df: pd.DataFrame, columns=None, return_scaler=False):
    """
    对指定列进行标准化（StandardScaler）

    Parameters
    ----------
    df : pd.DataFrame
    columns : list or None
        要标准化的列名
        - None: 默认对所有数值列做
    return_scaler : bool
        是否返回 scaler（用于后续 transform）

    Returns
    -------
    df_scaled : pd.DataFrame
    scaler (optional)
    """

    df = df.copy()

    # ✅ 如果没指定列 → 自动选数值列
    if columns is None:
        columns = df.select_dtypes(include="number").columns.tolist()

    # ❌ 没有数值列
    if len(columns) == 0:
        raise ValueError("No numeric columns to standardize.")

    # ❌ 检查是否有非数值列
    non_numeric = [col for col in columns if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric:
        raise ValueError(f"Non-numeric columns found: {non_numeric}")

    # ✅ 标准化
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])

    if return_scaler:
        return df, scaler

    return df