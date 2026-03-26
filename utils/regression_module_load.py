from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


def load_model(model_type: str, **kwargs):
    """
    根据输入的 model_type 返回对应模型

    Parameters
    ----------
    model_type : str
        模型类型，例如:
        - "linear"
        - "ridge"
        - "lasso"
        - "rf"
        - "tree"

    kwargs :
        传给模型的参数，比如 alpha, n_estimators 等

    Returns
    -------
    model : sklearn model instance
    """

    model_type = model_type.lower()

    if model_type == "linear":
        return LinearRegression(**kwargs)

    elif model_type == "ridge":
        return Ridge(**kwargs)

    elif model_type == "lasso":
        return Lasso(**kwargs)

    elif model_type == "rf":
        return RandomForestRegressor(
        n_estimators=50,   # 默认别太大
        max_depth=5,       # 控制复杂度
        **kwargs)

    elif model_type == "tree":
        return DecisionTreeRegressor(**kwargs)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")