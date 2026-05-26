import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import augumentations as aug


def load_video_score(
        score_path: str = None,
        lower_bound: float = 0,
        upper_bound: float = 4,
        columns: list[str] = ["score"]
        ) -> pd.DataFrame:
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


def select_equally_spaced_rows(
        data: pd.DataFrame,
        num_rows: int = 30
        ) -> pd.DataFrame:
    """
    Select equally spaced frames/rows from a pandas DataFrame and returns a copy of the original with only the selected frames/rows.

    Args:
        data : Input pandas DataFrame
        num_rows : Number of rows to select
        axis : Axis along which to select frames

    Returns:
        DataFrame with selected number of frames/rows
    """

    df = data.copy()

    total_frames = data.shape[0]
    if total_frames < num_rows:
        raise Exception(f"The specified axis does not contain {num_rows} entries")

    # Compute what frames to include
    indices = np.linspace(0, total_frames - 1, num_rows).astype(int)

    # Create a mask to index with
    mask = np.zeros((total_frames,), dtype=bool)
    for index in indices:
        mask[index] = True

    return df[mask]


def create_tensor_from_dataframe(
        data: pd.DataFrame
        ) -> torch.Tensor:

    """
    Converts the input pandas DataFrame into a pytorch Tensor

    data_shape: (c, n)

    Args:
        data: The pandas DataFrame to convert
    Returns:
        Tensor: The data in tensor format
    """
    input_data = torch.tensor(data.to_numpy(), dtype=torch.float32)
    return input_data


def create_TensorDataset(
        sequences: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int = 32
        ) -> TensorDataset:
    """Create a DataLoader using a tensor of sequences with an equally sized label tensor."""
    tensor_data = TensorDataset(sequences, labels)
    dataloader = DataLoader(tensor_data, batch_size=batch_size, shuffle=True, drop_last=False)
    return dataloader


def extend_tensor(
        data: torch.Tensor,
        n: int = 2
        ) -> torch.Tensor:
    """
    Returns a copy of the input tensor with each element in the first dimension copied n times

    Shape (c, ...) -> (c*n, ...)

    Args:
        data: The tensor to be extended
        n: The number of copies of each element to extend by

    Returns:
        Tensor: The extended tensor
    """
    new_data = []
    for entry in data:
        for _ in range(n):
            new_data.append(entry)
    return torch.stack(new_data)


def load_folder_with_split(
        folder_path: str,
        target_path: str = None,
        test_size: float = 0.2,
        random_state: int = 42,
        select_rows: bool = False,
        num_rows: int = 30,
        separate_target_file: bool = True,
        target_column: str = "target"
        ) -> tuple[list, list, list, list]:
    """
    For use with data and target that are in separate files.

    Loads a folder with data and a target file with the target data and returns lists with training/test data and training/test targets.
    The data is loaded into pandas DataFrames and the targets are loaded as values.

    Args:
        folder_path: The path of the data folder
        target_path: The path of the target data file
        test_size: The ratio of test_files
        random_state: the seed to use in the train_test_split
        select_rows: whether to select num_rows of equidistant rows
        num_rows: how many equidistant rows to select

    Returns:
        tuple: lists of DataFrames for each file in folder_path (X_train, X_test, Y_train, Y_test)
    """

    if separate_target_file and target_path is not None:
        scores = load_video_score(target_path)
        files = list(scores["file"])
    elif separate_target_file and target_path is None:
        raise Exception("target_path cannot be None if target path is required")

    all_files = []
    all_targets = []

    for file in os.listdir(folder_path):

        path = os.path.join(folder_path, file)
        file_name = file.split("_")[0]

        if separate_target_file and file_name not in files:
            pass
        else:
            # Select frames from the file
            df = pd.read_csv(path)

            # Drop the FrameNo column if it exists
            if "FrameNo" in df.columns:
                df = df.drop(columns=["FrameNo"])

            if select_rows:
                df = select_equally_spaced_rows(df, num_rows=num_rows)

            all_files.append(df.drop(columns=[target_column]))

            if separate_target_file:
                target = scores["scaled_score"].loc[scores["file"] == file_name].values[0]
            else:
                target = df[target_column]
            all_targets.append(target)

    X_train, X_test, Y_train, Y_test = train_test_split(all_files, all_targets, test_size=test_size, random_state=random_state)

    return X_train, X_test, Y_train, Y_test


def augument_data(
        data: list[pd.DataFrame],
        mirror_axis: list[tuple[bool, bool, bool]],
        rotations: list[float]
        ) -> torch.Tensor:
    """
    Mirrors the data and rotates each mirroring with the angles in the rotation parameter around the z-axis.
    Also standardizes the data.

    Args:
        data: the list of DataFrames to augument
        mirror_axis: list of mirroring setups (x-axis, y-axis, z-axis)
        rotations: list of angles in radians to rotate each node structure around the z-axis
    Returns:
        Tensor: tensor containing the augumented data
    """

    samples = []

    # Augument the data
    for df in data:

        # Mirror
        for setup in mirror_axis:
            mirrored_data = aug.mirror(df, axis=setup)

            # Rotate
            for angle in rotations:
                rotated_data = aug.rotate(mirrored_data, angle, axis=0)

                # Standardize the data
                scaler = StandardScaler()
                sample = torch.tensor(scaler.fit_transform(rotated_data), dtype=torch.float32)

                # Convert to tensors
                samples.append(sample)

    return torch.stack(samples)
