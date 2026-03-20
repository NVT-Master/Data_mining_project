# Model Evaluation Report

## Executive Summary

Best Model: **Extra Trees**

- R² Score: 0.9591
- MAE: 7830.0393
- RMSE: 16386.8372
- MAPE: 0.1472%

## Model Comparison

|                        | Model                  |      MAE |    RMSE |         MSE |       R2 |     MAPE |      MASE |     MBE |   Median_AE |
|:-----------------------|:-----------------------|---------:|--------:|------------:|---------:|---------:|----------:|--------:|------------:|
| Extra Trees            | Extra Trees            |  7830.04 | 16386.8 | 2.68528e+08 | 0.959096 | 0.14719  | 0.0992042 | 2727.52 |     3105.46 |
| Hist Gradient Boosting | Hist Gradient Boosting | 13092.2  | 24645.4 | 6.07398e+08 | 0.907476 | 0.236662 | 0.165874  | 4182.8  |     5573.52 |
| Random Forest          | Random Forest          | 12902.3  | 25938.4 | 6.72803e+08 | 0.897513 | 0.22213  | 0.163468  | 7853.73 |     4898.66 |
| Ridge Regression       | Ridge Regression       | 21871.6  | 37609.4 | 1.41447e+09 | 0.784536 | 0.461949 | 0.277107  | 6194.15 |    10477.3  |
| Linear Regression      | Linear Regression      | 21961.6  | 38065.3 | 1.44896e+09 | 0.779282 | 0.456978 | 0.278246  | 5075.56 |    10356.4  |

## Key Findings

1. The Extra Trees model achieved the highest R² score of 0.9591
2. Average prediction error (MAPE): 0.15%
3. Mean Absolute Error: 7830.0393
4. Root Mean Squared Error: 16386.8372

