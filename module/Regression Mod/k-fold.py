import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, accuracy_score


def k_fold_cv(model, X, y, k=5, problem_type="regression", random_state=None):
    """
    通用 k 折交叉验证函数

    参数:
    - model: 已经初始化的 sklearn 模型
    - X: 特征数据，numpy array 或 pandas DataFrame
    - y: 标签，numpy array 或 pandas Series
    - k: 折数
    - problem_type: "regression" 或 "classification"
    - random_state: 随机种子，用于 KFold

    返回:
    - results: 每折的得分列表
    - mean_score: 平均得分
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    scores = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        if problem_type == "regression":
            score = mean_squared_error(y_val, y_pred)
        elif problem_type == "classification":
            score = accuracy_score(y_val, y_pred)
        else:
            raise ValueError("problem_type 必须是 'regression' 或 'classification'")

        scores.append(score)

    return scores, np.mean(scores)