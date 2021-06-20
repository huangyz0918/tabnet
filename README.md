# TabNet 

Yet another implementation of paper: 

Sercan Ö. Arik, Tomas Pfister:
TabNet: Attentive Interpretable Tabular Learning. AAAI 2021: 6679-6687

## Quick start 

Setup environment

```bash 
pip install -r requirements.txt
```

```bash
export PYTHONPATH="${PWD}"
```

Run experiment on the Forest Cover Type dataset. 

```bash
python3 main.py
```

## Example Results

Here is an example output from the console by conducting the self-superised learning for TabNet on Forest Cover Type dataset.

```bash
psutil is not installed. You will not be able to abort this experiment from the UI.
psutil is not installed. Hardware metrics will not be collected.
https://app.neptune.ai/huangyz0918/tabnet/e/TAB-38
Device configuration: Using cuda:0 for training/inference
Starting training...
Training model with predictive objective
Predictive - Epoch: 1, Step: 29, Total train loss: 1.2457, Validation criterion loss: 0.9108, Validation accuracy: 0.6272
Predictive - Epoch: 2, Step: 58, Total train loss: 0.8106, Validation criterion loss: 0.7856, Validation accuracy: 0.6642
Predictive - Epoch: 3, Step: 87, Total train loss: 0.7551, Validation criterion loss: 0.8022, Validation accuracy: 0.6584
Predictive - Epoch: 4, Step: 116, Total train loss: 0.7407, Validation criterion loss: 0.7971, Validation accuracy: 0.6581
Predictive - Epoch: 5, Step: 145, Total train loss: 0.7272, Validation criterion loss: 0.7679, Validation accuracy: 0.6728
Predictive - Epoch: 6, Step: 174, Total train loss: 0.7099, Validation criterion loss: 0.7626, Validation accuracy: 0.6878
Predictive - Epoch: 7, Step: 203, Total train loss: 0.6896, Validation criterion loss: 0.7723, Validation accuracy: 0.6809
Predictive - Epoch: 8, Step: 232, Total train loss: 0.687, Validation criterion loss: 0.8589, Validation accuracy: 0.6572
Predictive - Epoch: 9, Step: 261, Total train loss: 0.673, Validation criterion loss: 0.8155, Validation accuracy: 0.6688
Predictive - Epoch: 10, Step: 290, Total train loss: 0.6671, Validation criterion loss: 0.9569, Validation accuracy: 0.6616
Saving model to: runs/forest_cover/1624188253_forest_cover_predictive_model_final.pt
Device configuration: Using cuda:0 for training/inference
TabNet accuracy: 0.662
```

## Reference 

- [[paper] TabNet: Attentive Interpretable Tabular Learning.](https://arxiv.org/abs/1908.07442)
- [[code] Official Implementation](https://github.com/google-research/google-research/tree/master/tabnet)
- [[code] PyTorch Implementation](https://github.com/dreamquark-ai/tabnet)

## Liscense 

[MIT](./LICENSE)

