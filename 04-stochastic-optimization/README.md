# Stochastic Optimization — SGD, Adagrad and Adam for Multinomial Logistic Regression

Course project comparing stochastic first-order methods on the
L2-regularised multinomial logistic regression problem

$$F_\alpha(W)=\frac1N\sum_{i=1}^N \ell_{\mathrm{CE}}(W x_i, y_i)+\frac{\alpha}{2}\|W\|_F^2,\qquad W\in\mathbb{R}^{10\times 784},$$

on MNIST (48 000 training / 12 000 validation images).

## Implementation

`stochastic_opt_code.py` is self-contained:

* the cross-entropy gradient is computed in closed form,
  `∇W = (softmax(XWᵀ) − Y)ᵀ X / B`, instead of autograd;
* SGD, Adagrad and Adam (with bias correction) are implemented from scratch,
  each adding the ridge term `αW` to the stochastic gradient;
* **two-stage hyper-parameter search** for every optimizer — a coarse
  log-grid over (learning rate, α) with 15 epochs and objective-based early
  stopping, then a finer grid around the coarse winner with 30 epochs;
* **final training** (≤ 100 epochs, validation early stopping with patience
  10) with the selected hyper-parameters, followed by the comparison plots.

All random sources (PyTorch, NumPy, data split and loader shuffling) are
seeded with 42.

```bash
python stochastic_opt_code.py   # downloads MNIST to ./data, writes results/
```

The full run takes about 10 minutes on a laptop CPU.

## Results

| Optimizer | learning rate | α | best validation accuracy | best epoch |
|---|---|---|---|---|
| SGD | 0.316 | 0 | 91.69 % | 38 |
| Adam | 1e-3 | 0 | 91.99 % | 19 |
| Adagrad | 0.1 | 0 | 92.08 % | 20 |

Adaptive methods reach the same accuracy plateau in roughly half the epochs
of tuned SGD; with this model capacity the selected ridge parameter is 0 for
all three methods. The JSON files in `results/` contain the full tuning
history and per-epoch training curves; the report is
[`report/stochastic_opt_report.pdf`](report/stochastic_opt_report.pdf).
