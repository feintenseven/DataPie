import pandas as pd

def one_hot_encoding_module(df, categorical_columns=None, drop_first=False, return_mapping=False):
    """
    独热编码模块

    参数:
    - df: pandas DataFrame
    - categorical_columns: list, 指定要编码的分类列（默认自动检测 object 或 category 类型列）
    - drop_first: bool, 是否删除第一列避免多重共线性
    - return_mapping: bool, 是否返回每列对应的编码映射字典

    返回:
    - result: dict，包含
        - "encoded_df": 编码后的 DataFrame
        - "mapping" (可选): 每列编码映射字典
    """
    df_copy = df.copy()

    # 自动检测分类列
    if categorical_columns is None:
        categorical_columns = df_copy.select_dtypes(include=['object', 'category']).columns.tolist()

    mapping = {}

    for col in categorical_columns:
        dummies = pd.get_dummies(df_copy[col], prefix=col, drop_first=drop_first)
        if return_mapping:
            mapping[col] = {v: f"{col}_{v}" for v in df_copy[col].unique()}
        df_copy = pd.concat([df_copy.drop(columns=[col]), dummies], axis=1)

    result = {"encoded_df": df_copy}
    if return_mapping:
        result["mapping"] = mapping

    return result