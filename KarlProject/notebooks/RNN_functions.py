import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import augumentations as aug


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


def create_sequence_from_dataframe(data: pd.DataFrame) -> torch.Tensor:

    """
    Converts the input pandas DataFrame into a pytorch Tensor with eventual label to identify

    data_shape: (c, n)

    Args:
        data: The pandas DataFrame to convert
    Returns:
        Tensor: The data in tensor format
    """
    c, n = data.shape
    input_data = torch.tensor(data.to_numpy(), dtype=torch.float32)
    return input_data


def create_TensorDataset(sequences: torch.Tensor, labels: torch.Tensor, batch_size: int = 32) -> TensorDataset:
    """Create a DataLoader using a tensor of sequences with an equally sized label tensor."""
    tensor_data = TensorDataset(sequences, labels)
    dataloader = DataLoader(tensor_data, batch_size=batch_size, shuffle=True, drop_last=False)
    return dataloader


# Test if the functions work
def main():
    path = "MainProject/data/mediapipe_not_trimmed_world/A1_mediapipe.csv"
    df = pd.read_csv(path).drop(columns=["FrameNo"])
    augumented = aug.mirror(df)
    df = pd.concat([df, augumented])
    print(create_sequence_from_dataframe(df).shape)


if __name__ == "__main__":
    main()
