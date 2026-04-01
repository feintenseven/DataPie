import numpy as np


def bootstrap_estimate(data, stat_func=np.mean, n_resamples=1000, ci=0.95, random_state=None):
    """
    用 bootstrap 从小样本估计总体统计量和置信区间

    参数：
    - data: 1D array-like, 你的样本数据
    - stat_func: callable, 统计量函数，默认是均值 np.mean
    - n_resamples: int, 重抽样次数
    - ci: float, 置信区间，比如 0.95
    - random_state: int 或 None, 控制可复现

    返回：
    - stat_estimate: 原样本统计量
    - ci_lower: 下置信限
    - ci_upper: 上置信限
    """

    rng = np.random.default_rng(random_state)
    data = np.array(data)
    n = len(data)

    # 重抽样
    resampled_stats = []
    for _ in range(n_resamples):
        sample = rng.choice(data, size=n, replace=True)
        resampled_stats.append(stat_func(sample))

    resampled_stats = np.array(resampled_stats)

    # 原样本统计量
    stat_estimate = stat_func(data)

    # 置信区间
    lower = (1 - ci) / 2 * 100
    upper = (1 + ci) / 2 * 100
    ci_lower = np.percentile(resampled_stats, lower)
    ci_upper = np.percentile(resampled_stats, upper)

    return stat_estimate, ci_lower, ci_upper

a,b,c=bootstrap_estimate([1,2,3,3,6,9,10,2,3,10,4])
print(a,b,c)