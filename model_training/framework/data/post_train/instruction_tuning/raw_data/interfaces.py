"""Public loaders for instruction-tuning experiment data."""

import logging

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)


def load_bipia_dataset() -> Dataset:
    """Load the Hugging Face BIPIA dataset using the standard local cache.

    The Hub split is named ``train``, but the dataset card describes its
    attacks as BIPIA test attacks. Keep it separate from SFT training data
    when evaluating on those attacks.
    """
    logger.info("load_bipia_dataset | dataset=geodesic-research/bipia split=train")
    try:
        dataset = load_dataset("geodesic-research/bipia", "bipia", split="train")
    except Exception:
        logger.exception("load_bipia_dataset | failed to load BIPIA")
        raise
    logger.info("load_bipia_dataset | rows=%d", len(dataset))
    return dataset
