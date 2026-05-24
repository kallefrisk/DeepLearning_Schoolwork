import pandas as pd
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


# Test if the functions work
def main():
    path = "MainProject/data/video_scores.csv"
    print(load_video_score(path))


if __name__ == "__main__":
    main()
