# GAN-StockPrediction

A project exploring **Generative Adversarial Networks (GANs)** and **Wasserstein GAN with Gradient Penalty (WGAN-GP)** for stock time-series data. The pipeline covers data loading, preprocessing, and training GAN-based models to generate or model stock price sequences.

## Project structure

| File / folder | Description |
|---------------|-------------|
| `1. data_loading.ipynb` | Fetches historical stock data via **yfinance** (single or multiple tickers, configurable period/interval). Saves to `stock_data.csv`. |
| `2. data_preprocessing.ipynb` | Loads `stock_data.csv`, selects OHLCV + Dividends + Stock Splits, applies **MinMaxScaler** normalization, saves to `normalized_stock_data.csv`. |
| `3. GAN.ipynb` | **Vanilla GAN** (PyTorch): generator vs discriminator on stock sequences. |
| `4. WGAN-GP.ipynb` | **WGAN-GP** (PyTorch): Wasserstein GAN with gradient penalty for more stable training. |
| `stock_data.csv` | Raw stock data (from notebook 1). |
| `normalized_stock_data.csv` | Min-max normalized data (from notebook 2). |

## How to run

1. **Environment**: Python 3.9+ with `pandas`, `numpy`, `yfinance`, `scikit-learn`, `torch`, `matplotlib`.
2. **Order**: Run notebooks in order: **1 → 2 → 3** (or **4**). Notebooks 3 and 4 expect `normalized_stock_data.csv` (or equivalent preprocessed data).
3. **Data**: In `1. data_loading.ipynb`, set `tickers` (e.g. `["NVDA"]` or `["AAPL","MSFT","GOOGL","AMZN"]`) and run to produce `stock_data.csv`.

---

## What works

- **Data pipeline**: End-to-end flow from yfinance → raw CSV → normalized CSV works. Single- and multi-ticker loading, configurable period/interval.
- **Preprocessing**: MinMaxScaler on OHLCV + Dividends + Stock Splits; no-NaN handling; reproducible normalization and CSV export.
- **Model code**: GAN and WGAN-GP are implemented in PyTorch with standard building blocks (e.g. fully connected or simple recurrent/conv layers for sequences). Training loops run and loss curves can be logged.
- **Reproducibility**: Fixed seeds and clear notebook order make runs reproducible for the same data and hyperparameters.
- **WGAN-GP**: Gradient penalty helps with training stability compared to a vanilla GAN (fewer gradient explosions, more stable discriminator/critic).

---

## What doesn’t work (or is limited)

- **Predicting future prices**: These models are **generative** (learn distributions of sequences), not direct “next-day price” predictors. They are not tuned or evaluated as forecasting models (e.g. no MSE/MAE on hold-out future dates).
- **Vanilla GAN instability**: The standard GAN in `3. GAN.ipynb` can suffer from mode collapse, vanishing gradients, or oscillating losses; quality of generated sequences may be inconsistent.
- **Evaluation**: There is no formal metric (e.g. distribution distance, coverage, or forecasting error) to compare real vs generated sequences or to select the best checkpoint.
- **Temporal structure**: If the models use only MLPs on flattened windows, they may not fully capture long-range dependencies; LSTM/Transformer-style generators are not necessarily included or tuned.
- **Single-asset default**: Default setup is one ticker (e.g. NVDA). Multi-ticker and cross-asset behavior are not deeply validated.
- **Production readiness**: No CLI, config files, or unit tests; notebooks are exploratory.

---

## Future improvements

1. **Evaluation**
   - Add metrics: distribution distance (e.g. MMD, Wasserstein), correlation structure, or simple forecasting baselines (e.g. MSE on a held-out horizon).
   - Visual and statistical comparison of real vs generated trajectories (e.g. histograms, ACF, simple t-tests on summary stats).

2. **Architecture**
   - Use **recurrent (LSTM/GRU)** or **temporal convolutional** layers in generator/critic to better capture time dependencies.
   - Try **Conditional GANs** (e.g. conditioning on volatility regime, sector, or past returns) for more controlled generation.

3. **Training**
   - More systematic hyperparameter search (learning rate, latent size, sequence length, batch size).
   - Learning rate scheduling and gradient clipping where helpful.
   - Checkpointing and early stopping based on a validation metric (once defined).

4. **Data**
   - Support multiple tickers and train on pooled or asset-specific data; optional features (e.g. technical indicators, volume).
   - Train/validation/test splits in time (e.g. last N weeks for test) and optional walk-forward evaluation.

5. **Reproducibility and code**
   - Add `requirements.txt` (e.g. `pandas`, `numpy`, `yfinance`, `scikit-learn`, `torch`, `matplotlib`) and a short setup section in the README.
   - Optionally refactor shared code (data loading, sequence creation, training loop) into `.py` modules and call from notebooks.

6. **Downstream use**
   - If the goal is **forecasting**, add a dedicated forecasting head or compare with simple baselines (e.g. persistence, moving average) and report metrics.
   - If the goal is **synthetic data**, document intended use (e.g. backtesting, stress scenarios) and add simple sanity checks (e.g. no negative prices, plausible volatility).

---

## License

See repository license (if any).
