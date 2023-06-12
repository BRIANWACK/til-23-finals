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

## ToF

ToF documentation: <https://robomaster-dev.readthedocs.io/en/latest/python_sdk/robomaster.html#module-robomaster.sensor>.

Found a cheatsheet that claims the distances are left, right, front and back: <https://github.com/ThePBone/RobomasterCheatsheet#callback>. Cheatsheet also links to example usage.

```py
# Current distances of left, right, front and back.
cur_dist_lrfb = [left, right, front, back]

def update_dist(dists):
    nonlocal cur_dist
    cur_dist = dists

robot.sensor.sub_distance(freq=1, callback=update_dist)
```
