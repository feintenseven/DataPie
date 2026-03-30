import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score


def kfold_regression(X: pd.DataFrame,
                     y: pd.Series,
                     model,
                     scale: bool = True,
                     n_splits: int = 5,
                     random_state: int = 42,
                     return_details: bool = False,
                     **kwargs):
    """
    通用 k-fold 回归函数
    """

    if scale:
        from module.Preprocessing.scaler import standardize

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = []
    rmse_scores = []
    r2_scores = []
    fold_details = []
    all_coefficients = []
    all_intercepts = []
    all_feature_importances = []  # 存储特征重要性（如果是树模型）

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

        # 处理标准化
        if scale:
            X_train_scaled, scaler = standardize(X_train, return_scaler=True)
            X_test_scaled = X_test.copy()
            common_cols = X_train_scaled.columns.tolist()
            X_test_scaled[common_cols] = scaler.transform(X_test[common_cols])
        else:
            X_train_scaled, X_test_scaled = X_train, X_test

        # 复制模型并训练
        clf = model.__class__(**model.get_params())
        clf.fit(X_train_scaled, y_train, **kwargs)
        y_pred = clf.predict(X_test_scaled)

        # 计算指标
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        rmse_scores.append(rmse)
        r2_scores.append(r2)

        # 保存结果
        results.append((y_test, y_pred))

        # 保存模型参数（根据模型类型）
        fold_info = {
            'fold': fold_idx + 1,
            'rmse': rmse,
            'r2': r2,
            'y_test': y_test.values,
            'y_pred': y_pred
        }

        # 线性模型：系数和截距
        if hasattr(clf, 'coef_'):
            all_coefficients.append(clf.coef_)
            all_intercepts.append(clf.intercept_)
            fold_info['coefficients'] = clf.coef_
            fold_info['intercept'] = clf.intercept_
            fold_info['model_type'] = 'linear'

        # 树模型：特征重要性
        elif hasattr(clf, 'feature_importances_'):
            all_feature_importances.append(clf.feature_importances_)
            fold_info['feature_importances'] = clf.feature_importances_
            fold_info['model_type'] = 'tree'

        # 其他模型
        else:
            fold_info['model_type'] = 'other'

        if return_details:
            fold_details.append(fold_info)

    # 计算整体指标
    rmse_mean = np.mean(rmse_scores)
    rmse_std = np.std(rmse_scores)
    r2_mean = np.mean(r2_scores)
    r2_std = np.std(r2_scores)

    result = {
        'rmse_mean': rmse_mean,
        'rmse_std': rmse_std,
        'r2_mean': r2_mean,
        'r2_std': r2_std,
        'predictions': results
    }

    # 添加模型解释信息
    if all_coefficients:
        # 线性模型：系数和截距
        mean_coef = np.mean(all_coefficients, axis=0)
        mean_intercept = np.mean(all_intercepts)

        formula = f"y = {mean_intercept:.4f}"
        for i, coef in enumerate(mean_coef):
            formula += f" + ({coef:.4f}) * {X.columns[i]}"

        result['model_type'] = 'linear'
        result['formula'] = formula
        result['coefficients_mean'] = dict(zip(X.columns, mean_coef))
        result['intercept_mean'] = mean_intercept
        result['coefficients_std'] = dict(zip(X.columns, np.std(all_coefficients, axis=0)))

        if return_details:
            result['all_coefficients'] = all_coefficients
            result['all_intercepts'] = all_intercepts

    elif all_feature_importances:
        # 树模型：特征重要性
        mean_importance = np.mean(all_feature_importances, axis=0)
        std_importance = np.std(all_feature_importances, axis=0)

        # 按重要性排序
        importance_dict = dict(zip(X.columns, mean_importance))
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

        result['model_type'] = 'tree'
        result['feature_importances_mean'] = importance_dict
        result['feature_importances_std'] = dict(zip(X.columns, std_importance))
        result['top_features'] = sorted_importance[:5]  # 前5个最重要的特征

        if return_details:
            result['all_feature_importances'] = all_feature_importances

    else:
        result['model_type'] = 'other'
        result['message'] = '该模型类型不支持提取系数或特征重要性'

    if return_details:
        result['folds_details'] = fold_details

    return result