# til-23-finals

Good Luck.

```sh
git clone --recurse-submodules
git submodule update --init --recursive
```

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

```sh
python -m til_23_finals --config cfg/autonomy_cfg.yml
```
