import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score


def single_fit_module(model, X, y, problem_type="regression", return_metrics=True):
    """
    单次训练-预测模块，通用任意 sklearn 模型

    参数:
    - model: sklearn 模型实例（已初始化）
    - X: 特征数据 (numpy array 或 pandas DataFrame)
    - y: 标签数据 (numpy array 或 pandas Series)
    - problem_type: "regression" 或 "classification"
    - return_metrics: 是否返回指标

    返回:
    - result: dict，包含训练好的模型、预测值、真实值，以及指标
    """
    # 如果是 DataFrame/Series，转 numpy
    if hasattr(X, "values"):
        X = X.values
    if hasattr(y, "values"):
        y = y.values

    # 训练
    model.fit(X, y)

    # 预测
    y_pred = model.predict(X)

    # 构造返回结果
    result = {
        "model": model,
        "X": X,
        "y_true": y,
        "y_pred": y_pred
    }

    if return_metrics:
        if problem_type == "regression":
            result["MSE"] = mean_squared_error(y, y_pred)
            result["R2"] = r2_score(y, y_pred)
        elif problem_type == "classification":
            result["Accuracy"] = accuracy_score(y, y_pred)
        else:
            raise ValueError("problem_type 必须是 'regression' 或 'classification'")

    return result