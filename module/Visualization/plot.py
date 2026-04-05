import matplotlib.pyplot as plt
import seaborn as sns


def plot_graph(df, col1, col2, plot_type='histogram', **kwargs):
    """
    画图函数，根据不同的图形类型输出对应的图像。

    参数:
    df (pd.DataFrame): 输入的数据框
    col1 (str): 第一个列名
    col2 (str): 第二个列名
    plot_type (str): 图形类型 ['histogram', 'violin', 'box', 'line', 'bar', 'scatter']
    kwargs: 其他可能的参数，比如颜色、标题等

    返回:
    None, 显示图片
    """

    plt.figure(figsize=(8, 6))

    if plot_type == 'histogram':
        sns.histplot(df[col1], kde=True, **kwargs)
        plt.title(f'Histogram of {col1}')

    elif plot_type == 'violin':
        sns.violinplot(x=col1, y=col2, data=df, **kwargs)
        plt.title(f'Violin Plot of {col1} vs {col2}')

    elif plot_type == 'box':
        sns.boxplot(x=col1, y=col2, data=df, **kwargs)
        plt.title(f'Box Plot of {col1} vs {col2}')

    elif plot_type == 'line':
        sns.lineplot(x=col1, y=col2, data=df, **kwargs)
        plt.title(f'Line Plot of {col1} vs {col2}')

    elif plot_type == 'bar':
        sns.barplot(x=col1, y=col2, data=df, **kwargs)
        plt.title(f'Bar Plot of {col1} vs {col2}')

    elif plot_type == 'scatter':
        sns.scatterplot(x=col1, y=col2, data=df, **kwargs)
        plt.title(f'Scatter Plot of {col1} vs {col2}')

    else:
        print("Invalid plot type! Choose from ['histogram', 'violin', 'box', 'line', 'bar', 'scatter']")
        return

    plt.show()
