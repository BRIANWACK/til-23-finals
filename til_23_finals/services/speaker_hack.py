"""Overrides for each team member."""

# Tune per team config.
# This can be done by referring to speaker_id.ipynb and seeing when voice extraction
# boosts sigma.
CUSTOM_EXTRACTOR_CFG = {
    "PALMTREE": dict(
        skip_df=True,
        skip_spectral=True,
        noise_removal_limit_db=5,
    ),
    "ANYTHING": dict(
        skip_df=False,
        noise_removal_limit_db=7,
    ),
}

CUSTOM_OTHER_CFG = {
    "PALMTREE": dict(
        use_shifts=True,
        shift=0.0025,
        use_agc=True,
        agc_window=0.4,
        speed=0.985,
    ),
    "IMAGINELOSIN": dict(
        use_shifts=True,
        shift=0.005,
    ),
}
