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
pip install -e ./til-23-cv/ultralytics
pip install -e ./til-23-cv
pip install -e ./til-23-sdk2
pip install -e ./til-23-asr
```

```sh
pip install pip-whl/*
pip install sdk-whl/*
pip install whl/*
pip install -e ./til-23-cv/ultralytics --no-build-isolation --no-deps
pip install -e ./til-23-cv --no-build-isolation --no-deps
pip install -e ./til-23-asr --no-build-isolation --no-deps
# If you wish to use the (modified) simulator.
pip install -e ./til-23-sdk2 --no-build-isolation --no-deps
```

```sh
# OPTIONAL: We do not rely on NeMo's loading code at all.
conda install sox libsndfile ffmpeg -c conda-forge
```

## Run

NOTE: There is now a VSCode Debug Launch Config ("Simulator") that launches everything needed for simulation.

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

https://www.youtube.com/watch?v=zkFVO-bCfaY&ab_channel=DJITutorials

```py
# Current distances of left, right, front and back.
cur_dist_lrfb = [left, right, front, back]

def update_dist(dists):
    nonlocal cur_dist
    cur_dist = dists

robot.sensor.sub_distance(freq=1, callback=update_dist)
```

## Calculate Requirements

`til-23-sdk` and its dependencies should be excluded from the requirements calculation. Requirements should be stripped down only to linux platform dependencies.

Comment out this line:

```toml
[tool.poetry.dependencies]
python = ">=3.8,<3.9"
# Comment out below:
# til-23-sdk = {path = "til-23-sdk2", develop = true}
til-23-cv = {path = "til-23-cv", develop = true}
til-23-asr = {path = "til-23-asr", develop = true}
nemo-toolkit = {extras = ["asr"], version = "^1.18.1"}
robomaster = "^0.1.1.68"
imutils = "^0.5.4"
```

Then run (changes to the lockfile should be reverted after):

```sh
poetry lock --no-update
poetry export --without-hashes -f requirements.txt -o requirements-no-sdk.txt
```

Then search for and remove requirements that contain the following:

- `file:`
- `platform_system == "Windows"`
- `platform_system == "Darwin"`
- `sys_platform == "win32"`
- `sys_platform == "darwin"`

Finally:

```sh
pip wheel -r requirements-no-sdk.txt -w whls
```

You may also choose to remove all `; python_version >= "3.8" and python_version < "3.9"`.
