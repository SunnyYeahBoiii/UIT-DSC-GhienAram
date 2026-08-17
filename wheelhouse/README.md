# Offline wheelhouse for Kaggle Python 3.12 / Linux x86_64

This wheelhouse upgrades the Kaggle runtime to Transformers 5.14.1, which recognizes the `qwen3_5` architecture used by `Qwen/Qwen3.5-2B`.

The Kaggle notebook installs every wheel here with `--no-index --no-deps` before starting any model subprocess. The base Kaggle image continues to provide PyTorch, bitsandbytes, datasets and the remaining common dependencies.

Do not install the Linux binary wheels in this directory into the local Windows Python environment.
