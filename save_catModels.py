import os
import joblib
import catboost as cb
import pandas as pd
import argparse

from maplight import get_fingerprints, scaler

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default="./cat_models", type=str, help='')
    parser.add_argument('--data_path', default='./data', type=str, help='')
    parser.add_argument('--target_property', default='caco2', type=str, required=True,
                        help='property need to be predicted')

    return parser.parse_args()

def train_and_save_with_scaler(X_train, y_train, attribute_name, save_dir="models", params=None, log_scale=False):
    """
    训练模型并保存，包含 scaler 和模型。
    参数:
        X_train (numpy.array): 训练特征
        y_train (numpy.array): 训练目标值
        attribute_name (str): 属性名称
        save_dir (str): 保存路径
        params (dict): CatBoost模型参数
        log_scale (bool): 是否对目标值进行对数变换
    """
    # 初始化Scaler
    Y_scaler = scaler(log=log_scale)
    Y_scaler.fit(y_train.values)
    train['Y_scale'] = Y_scaler.transform(y_train.values)

    # 设置默认参数
    if params is None:
        params = {'random_strength': 2, 'verbose': 0, 'loss_function': 'MAE'}

    # 训练CatBoost模型
    model = cb.CatBoostRegressor(**params)
    model.fit(X_train, train['Y_scale'].values)

    # 保存模型和scaler
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{attribute_name}_model.cbm")
    scaler_path = os.path.join(save_dir, f"{attribute_name}_scaler.pkl")

    model.save_model(model_path)
    Y_scaler.save(scaler_path)

    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")


if __name__ == '__main__':
    args = setup_args()

    save_dir = args.save_dir
    target_property = args.target_property
    train_path = args.data_path + target_property + '/train.csv'
    val_path = args.data_path + target_property + '/valid.csv'
    benchmark_config = {
        'caco2': ('regression', False),
        'lipo': ('regression', False),
        'logp': ('regression', False),
        'solubility': ('regression', False),
        'ppb': ('regression', False),
        'vdss': ('regression', True),
        'half_life': ('regression', True),
        'hum_mic_cl': ('regression', True),
        'ld50': ('regression', False),
        'esol': ('regression', False),
        'freesolv': ('regression', False),
        'water_sol': ('regression', False),
        'cm': ('regression', True),
        'ch': ('regression', True),
        'cyp_2c9': ('regression', True),
        'cyp_2d6': ('regression', True),
        'cyp_3a4': ('regression', True),
        'logd': ('regression', True),
        'mou_mic_cl': ('regression', True),
        'rat_mic_cl': ('regression', True),
    }
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    train = pd.concat([df_train, df_val])
    X_train = get_fingerprints(train['smiles'])
    y_train = train['value']
    
    task, log_scale = benchmark_config[target_property]
    params = {
        'random_strength': 2,
        'verbose': 0,
    }
    params['loss_function'] = 'MAE'
    train_and_save_with_scaler(X_train, y_train, target_property, save_dir = save_dir, params = params, log_scale = log_scale)
    print("cats all save over!")


