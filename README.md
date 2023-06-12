# til-23-finals

Good Luck.

Chassis Documentation: <https://robomaster-dev.readthedocs.io/en/latest/python_sdk/robomaster.html#robomaster-chassis>

## Clone

```sh
git clone --recurse-submodules
git submodule update --init --recursive
```

## Install

```sh
pip install -r requirements.txt
pip install -e ./til-23-cv
pip install -e ./til-23-sdk2
pip install -e ./til-23-asr
pip install -e ./til-23-cv/ultralytics
```

```sh
# OPTIONAL: We do not rely on NeMo's loading code at all.
conda install sox libsndfile ffmpeg -c conda-forge
```

## Run

```sh
# In separate terminals.
til-scoring cfg/scoring_cfg.yml -o logs
til-simulator -c cfg/sim_cfg.yml
python -m til_23_finals --config cfg/sim_autonomy.yml
```

```sh
python -m til_23_finals --config cfg/autonomy_cfg.yml
```
