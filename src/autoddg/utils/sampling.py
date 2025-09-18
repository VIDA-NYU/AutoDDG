from __future__ import annotations

from typing import Tuple

from beartype import beartype
from pandas import DataFrame


@beartype
def get_sample(
    data_pd: DataFrame, sample_size: int, random_state: int = 9
) -> Tuple[DataFrame, str]:
    """
    Return a random sample or whole frame and its CSV text

    Args:
        data_pd: Input DataFrame
        sample_size: Number of rows to sample
        random_state: Seed for reproducibility

    Returns:
        (sample_df, sample_csv)
    """

    if sample_size <= len(data_pd):
        data_sample = data_pd.sample(sample_size, random_state=random_state)
    else:
        data_sample = data_pd
    sample_csv = data_sample.to_csv(index=False)
    return data_sample, sample_csv
