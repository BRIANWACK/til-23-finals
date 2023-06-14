"""Overrides for each team member."""

# TODO: Tune per team config.
CUSTOM_EXTRACTOR_CFG = {
    "PALMTREE": dict(
        skip_df=True,
        skip_spectral=False,
        noise_removal_limit_db=16,
    ),
}
