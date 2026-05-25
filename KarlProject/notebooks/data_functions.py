import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_video_score(score_path: str = None, lower_bound: float = 0, upper_bound: float = 4, columns: list[str] = ["score"]) -> pd.DataFrame:
    """
    Loads and scales the video score found in the target file into the range [lower_bound, upper_bound]

    Args:
        score_path: path the the csv-file containing the score column
        lower_bound: The lower bound of the output range
        upper_bound: The upper bound of the output range
        columns: The columns to scale

    Returns:
        DataFrame: A pandas DataFrame with the scaled columns fit to the input range using MinMaxScaler from sklearn as the last column.
    """
    if score_path is None:
        raise Exception("Path cannot be of type 'None'")

    df = pd.read_csv(score_path)

    bad_videos = [
        "A1",
        "B2",
        "B3",
        "B4",
        "B5"
        ]

    df = df.loc[~df["file"].isin(bad_videos)]
    for column in columns:
        if column in df.columns:
            if column == "score":
                # Remove faulty scores
                df = df.loc[df["score"] != 0.0]

            scaler = MinMaxScaler(feature_range=(lower_bound, upper_bound))
            df[f"scaled_{column}"] = scaler.fit_transform(df[[column]])
        return df
    else:
        raise Exception(f"File has no column named '{column}'")


def select_equally_spaced_rows(data: pd.DataFrame, num_frames: int = 30):
    """
    Select equally spaced frames/rows from a pandas DataFrame and returns a copy of the original with only the selected frames/rows.

    Args:
        data : Input pandas DataFrame
        num_frames : Number of frames to select
        axis : Axis along which to select frames

    Returns:
        DataFrame with selected number of frames/rows
    """

    df = data.copy()

    total_frames = data.shape[0]
    if total_frames < num_frames:
        raise Exception(f"The specified axis does not contain {num_frames} entries")

    # Compute what frames to include
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

    # Create a mask to index with
    mask = np.zeros((total_frames,), dtype=bool)
    for index in indices:
        mask[index] = True

    return df[mask]


def main():
    path = "MainProject/data/mediapipe_not_trimmed_world/A1_mediapipe.csv"
    df = pd.read_csv(path).drop(columns=["FrameNo"])
    print(df.shape)
    df_sliced = select_equally_spaced_rows(df)
    print(df_sliced.shape)


if __name__ == "__main__":
    main()
