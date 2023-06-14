"""Overrides for each team member."""

# TODO: Tune per team config.
# This can be done by referring to speaker_id.ipynb and seeing when voice extraction
# boosts sigma.
CUSTOM_EXTRACTOR_CFG = {
    "PALMTREE": dict(
        skip_df=True,
        skip_spectral=True,
        noise_removal_limit_db=10,
    ),
}
