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
Starting training...
Training model with predictive objective
Predictive - Epoch: 1, Step: 29, Total train loss: 1.1801, Validation criterion loss: 0.9035, Validation accuracy: 0.6263
Predictive - Epoch: 2, Step: 58, Total train loss: 0.7854, Validation criterion loss: 0.8017, Validation accuracy: 0.6498
Predictive - Epoch: 3, Step: 87, Total train loss: 0.7387, Validation criterion loss: 0.7674, Validation accuracy: 0.6743
Predictive - Epoch: 4, Step: 116, Total train loss: 0.7253, Validation criterion loss: 0.747, Validation accuracy: 0.6913
Predictive - Epoch: 5, Step: 145, Total train loss: 0.711, Validation criterion loss: 0.7692, Validation accuracy: 0.6835
Saving model to: runs/forest_cover/1624193671_forest_cover_predictive_model_final.pt
Device configuration: Using cuda:0 for training/inference
TabNet accuracy: 0.683
```

## Reference 

- [[paper] TabNet: Attentive Interpretable Tabular Learning.](https://arxiv.org/abs/1908.07442)
- [[code] Official Implementation](https://github.com/google-research/google-research/tree/master/tabnet)
- [[code] PyTorch Implementation](https://github.com/dreamquark-ai/tabnet)

## Liscense 

[MIT](./LICENSE)

