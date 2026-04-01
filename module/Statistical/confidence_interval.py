import numpy as np
from scipy import stats


def confidence_interval(data, func=np.mean, confidence=0.95, method='t'):
    """
    通用置信区间函数（基于 t 分布或 z 分布）

    Args:
        data (array-like): 样本数据
        func (callable): 统计量函数，如 np.mean, np.median, np.var
        confidence (float): 置信水平
        method (str): 't' 或 'z'，选择用 t 分布或 z 分布

    Returns:
        statistic, lower, upper: 统计量及置信区间下限和上限
    """
    data = np.array(data)
    n = len(data)
    statistic = func(data)

    # 样本标准误差
    std_err = np.std(data, ddof=1) / np.sqrt(n)

    if method == 't':
        h = std_err * stats.t.ppf((1 + confidence) / 2, df=n - 1)
    elif method == 'z':
        h = std_err * stats.norm.ppf((1 + confidence) / 2)
    else:
        raise ValueError("method must be 't' or 'z'")

    return statistic, statistic - h, statistic + h

