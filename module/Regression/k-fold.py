from sklearn.model_selection import KFold
import pandas as pd


def kfold_regression(X: pd.DataFrame,
                     y: pd.Series,
                     model,  # 外部传入的模型对象
                     scale: bool = True,  # 是否标准化
                     n_splits: int = 5,  # k-fold 分割数量
                     random_state: int = 42,
                     **kwargs):  # 传给 model 的额外参数
    """
    通用 k-fold 回归函数

    Parameters
    ----------
    X : pd.DataFrame
        特征数据
    y : pd.Series
        标签
    model : object
        sklearn 风格回归模型实例，不在函数里创建
    scale : bool
        是否对特征进行标准化
    n_splits : int
        KFold 折数
    random_state : int
        KFold 随机种子
    **kwargs :
        传给 model.fit 的额外参数

    Returns
    -------
    results : list of tuples
        每一折的 (y_test, y_pred)
    """

    # 延迟 import，保证 utils/scaler 可以单独管理
    if scale:
        from module.Preprocessing.scaler import standardize

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

        # 处理标准化
        if scale:
            X_train_scaled, scaler = standardize(X_train, return_scaler=True)
            X_test_scaled = X_test.copy()
            X_test_scaled[X_train.columns] = scaler.transform(X_test)
        else:
            X_train_scaled, X_test_scaled = X_train, X_test

        # 使用传入的模型（保证每折是独立对象）
        clf = model.__class__(**model.get_params())  # 复制模型
        clf.fit(X_train_scaled, y_train, **kwargs)
        y_pred = clf.predict(X_test_scaled)

        results.append((y_test, y_pred))

    return results