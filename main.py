import pandas as pd

# ===== utils =====
from utils.data_loader import load_data
from utils.regression_module_load import load_model

# ===== preprocessing =====
from module.Preprocessing.one_hot_encoder import one_hot_encoding_module
from module.Preprocessing.scaler import standardize

# ===== regression =====
from module.Regression.k_fold import kfold_regression


def main():
    # ===== 1. load data =====
    df = load_data("mini.csv")

    target_col = "月收入"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # --- one hot ---
    df_onehot = one_hot_encoding_module(df, categorical_columns=['城市'], drop_first=True)["encoded_df"]

    # ===== 2. 定义 pipeline =====
    def pipeline(df):

        # ⚠️ 每个 fold 新 model
        model = load_model('linear')

        target_col = "月收入"
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # --- train ---
        result=kfold_regression(X=X,y=y,model=model,scale=False)

        # --- predict ---

        return result

    result=pipeline(df_onehot)

    print("=" * 50)
    print("线性回归 - 5折交叉验证结果")
    print("=" * 50)

    print(f"\n评估指标:")
    print(f"  平均 RMSE: {result['rmse_mean']:.4f} ± {result['rmse_std']:.4f}")
    print(f"  平均 R²:   {result['r2_mean']:.4f} ± {result['r2_std']:.4f}")

    print(f"\n回归公式:")
    print(f"  {result['formula']}")

    print(f"\n特征系数:")
    for feature, coef in result['coefficients_mean'].items():
        print(f"  {feature}: {coef:.4f} ± {result['coefficients_std'][feature]:.4f}")

    print(f"\n截距: {result['intercept_mean']:.4f}")


if __name__ == "__main__":
    main()