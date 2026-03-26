# module/modeling/regression_simple.py

from sklearn.model_selection import train_test_split
import pandas as pd


def simple_regression(X: pd.DataFrame,
                      y: pd.Series,
                      model,  # 外部传入模型对象
                      scale: bool = True,  # 是否标准化
                      test_size: float = 0.2,  # 测试集比例
                      random_state: int = 42,
                      **kwargs):  # 传给 model.fit 的额外参数
    """
    非 k-fold 回归训练函数

    Parameters
    ----------
    X : pd.DataFrame
        特征数据
    y : pd.Series
        标签
    model : object
        sklearn 风格回归模型实例
    scale : bool
        是否对特征进行标准化
    test_size : float
        测试集占比
    random_state : int
        随机种子
    **kwargs :
        传给 model.fit 的额外参数

    Returns
    -------
    dict
        {
            'model': 训练好的模型对象,
            'scaler': 如果 scale=True 返回 scaler 对象，否则 None,
            'X_train_scaled': 训练集特征,
            'X_test_scaled': 测试集特征,
            'y_train': 训练集标签,
            'y_test': 测试集标签,
            'y_pred': 测试集预测值
        }
    """

    # 延迟 import，保证模块独立
    if scale:
        from module.Preprocessing.scaler import standardize

    # 拆分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 标准化处理
    if scale:
        X_train_scaled, scaler = standardize(X_train, return_scaler=True)
        X_test_scaled = X_test.copy()
        X_test_scaled[X_train.columns] = scaler.transform(X_test)
    else:
        X_train_scaled, X_test_scaled = X_train, X_test
        scaler = None

    # 使用传入模型的副本，保证独立
    clf = model.__class__(**model.get_params())
    clf.fit(X_train_scaled, y_train, **kwargs)
    y_pred = clf.predict(X_test_scaled)

    return {
        'model': clf,
        'scaler': scaler,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred
    }