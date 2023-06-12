"""Utilities."""

import numpy as np

__all__ = [
    "cos_sim",
    "thres_strategy_A",
    "thres_strategy_naive",
    "thres_strategy_softmax",
]


def cos_sim(a, b):
    """Cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def thres_strategy_A(scores: list, accept_thres=0.3, vote_thres=0.0, sd_thres=4.4):
    """Strategy based on standard deviation.

    If any in `scores` is greater than `accept_thres`, return the index of the
    max score. If any in `scores` is greater than `vote_thres`, return the index
    of the max score if it is greater than `sd` standard deviations away from
    the mean of `scores` (excluding the max score). Otherwise, return -1.

    This strategy is not found in any literature and is the author's own.

    Parameters
    ----------
    scores : List[float]
        List of scores.
    accept_thres : float, optional
        Threshold for accepting a prediction, by default 0.7
    vote_thres : float, optional
        Threshold for voting, by default 0.1
    sd_thres : float, optional
        Number of standard deviations away from the mean, by default 5.0

    Returns
    -------
    int
        The index of the max score if it meets the criteria, otherwise -1.
    """
    if np.max(scores) > accept_thres:
        return np.argmax(scores)
    elif np.max(scores) > vote_thres:
        scores = np.array(scores).clip(0.0)  # type: ignore
        mean = np.mean(scores[scores < np.max(scores)])
        std = np.std(scores[scores < np.max(scores)])
        if np.max(scores) - mean > sd_thres * std:
            return np.argmax(scores)
    return -1


def thres_strategy_naive(scores: list, thres=0.3):
    """Naive thresholding strategy."""
    if np.max(scores) > thres:
        return np.argmax(scores)
    return -1


def thres_strategy_softmax(scores: list, temp=0.8, ratio=1.4):
    """Threshold using softmax."""
    x = np.array(scores) / temp  # type: ignore
    ex = np.exp(x - np.max(x))
    ex /= ex.sum() + 1e-12
    # TODO: Figure out proper solution to sensitivity.
    if np.max(ex) > ratio / (len(ex) + 1):
        return np.argmax(ex)
    return -1
