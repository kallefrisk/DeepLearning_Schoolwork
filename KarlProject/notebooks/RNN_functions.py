import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_video_score(score_path: str = None, lower_bound: float = 0, upper_bound: float = 4) -> pd.DataFrame:
    """
    Loads and scales the video score found in the target file into the range [lower_bound, upper_bound]

    Args:
        score_path: path the the csv-file containing the score column
        lower_bound: The lower bound of the output range
        upper_bound: The upper bound of the output range

    Returns:
        DataFrame: A pandas DataFrame with the scaled scores fit to the input range using MinMaxScaler from sklearn as the last column.
    """

    if score_path is None:
        raise Exception("Path cannot be of type 'None'")

    scores = pd.read_csv(score_path)

    if "score" in scores.columns:
        # Remove faulty scores
        scores = scores.loc[scores["score"] != 0.0]

        scaler = MinMaxScaler(feature_range=(lower_bound, upper_bound))
        scores["scaled_score"] = scaler.fit_transform(scores[["score"]])
        return scores
    else:
        raise Exception("File has no column named 'score'")


# Test if the functions work
def main():
    path = "MainProject/data/video_scores.csv"
    print(load_video_score(path))


if __name__ == "__main__":
    main()
