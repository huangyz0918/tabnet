import os
import neptune
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tabnet import TabNet

data_params = {
    "data_path": "data/covtype.csv",
    "target": "Cover_Type",
    "random_seed": 42,
    "model_save_dir": "runs/forest_cover",
    "model_save_name": "forest_cover",
    "categorical_variables": [],
}

train_params = {
    "batch_size": 16384,
    "run_self_supervised_training": False,
    "run_supervised_training": True,
    "early_stopping": True,
    "early_stopping_min_delta_pct": 0,
    "early_stopping_patience": 10,
    "max_epochs_supervised": 10,
    "max_epochs_self_supervised": 10,
    "epoch_save_frequency": 10,
    "train_generator_shuffle": True,
    "train_generator_n_workers": 0,
    "epsilon": 1e-7,
    "learning_rate": 0.02,
    "learning_rate_decay_factor": 0.95,
    "learning_rate_decay_step_rate": 1000,
    "sparsity_regularization": 0.0001,
    "p_mask": 0.7,
}

model_params = {
    "discrete_outputs": True,
    "categorical_variables": data_params["categorical_variables"],
}

if __name__ == "__main__":
    data = pd.read_csv(data_params["data_path"])
    X, y = (data[data.columns.difference([data_params["target"]])].values, data[data_params["target"]].values)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=data_params["random_seed"])
    y_train_copy, y_val_copy = (y_train.copy(), y_val.copy())

    enc = LabelEncoder()
    y_train_copy = enc.fit_transform(y_train_copy)
    y_val_copy = enc.fit_transform(y_val_copy)

    # make path
    if not os.path.exists(data_params["model_save_dir"]):
        os.makedirs(data_params["model_save_dir"])

    # initialize Neptune
    neptune.init('huangyz0918/tabnet')
    neptune.create_experiment(name='tabnet',
                            tags=['Pytorch', 'TabNet'],
                            params=model_params)

    # start training
    fc_tabnet_model = TabNet(logger=neptune, model_params=model_params)
    fc_tabnet_model.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        train_params=train_params,
        save_params={
            "model_name": data_params["model_save_name"],
            "save_folder": data_params["model_save_dir"],
        },
    )

    fc_tabnet_model = TabNet(logger=neptune, save_file=fc_tabnet_model.model_save_path)
    y_val_predict_tabnet = fc_tabnet_model.predict(X_val)
    print("TabNet accuracy: {}".format(np.round((y_val_predict_tabnet == y_val).sum() / len(y_val), 3)))