import numpy as np
import pandas as pd

def select_equally_spaced_frames(data, num_frames=30):
    """
    Selects equally spaced frames from a sequence.
    """

    if not isinstance(data, np.ndarray):
        data = np.array(data)

    total_frames = data.shape[0]

    indices = np.linspace(
        0,
        total_frames - 1,
        num_frames
    ).astype(int)

    return data[indices]


def sequence_fixed_c(trimmed_df, C=30):
    """
    Converts a trimmed sequence into exactly C equally spaced frames.
    """

    X = trimmed_df.values.astype(np.float32)

    if len(X) < C:
        raise ValueError(
            f"Sequence too short ({len(X)} frames). "
            f"Need at least {C} frames."
        )

    X_fixed = select_equally_spaced_frames(X, num_frames=C)

    fixed_c_df = pd.DataFrame(
        X_fixed,
        columns=trimmed_df.columns
    )

    return fixed_c_df