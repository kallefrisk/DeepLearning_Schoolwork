import numpy as np
import pandas as pd

def mirror(df:pd.DataFrame, mirror_x: bool = True, mirror_y: bool = True, mirror_z: bool = True) -> pd.DataFrame:
    '''
    Flips the node coordinates of the input kinect DataFrame.

    OBS: FrameNo column must be removed beforehand.

    Args:
        X: Input DataFrame
        flip_x: If the x-coordinate should be flipped
        flip_y: If the y-coordinate should be flipped
        flip_z: If the z-coordinate should be flipped

    Returns:
        Copy of the original DataFrame with flipped coordinates
    '''
    mirror_pairs = [
        (1, 3),   # shoulders
        (2, 4),   # elbows
        (5, 6),   # hands
        (7, 8),   # hips
        (9, 10),  # knees
        (11, 12), # feet
    ]

    df_copy = df.copy()

    for left, right in mirror_pairs:
        x_idx_left, y_idx_left, z_idx_left = left * 3, left * 3 + 1, left * 3 + 2
        x_idx_right, y_idx_right, z_idx_right = right * 3, right * 3 + 1, right * 3 + 2

        # Swap left with right and vice verse
        if mirror_x:
            df_copy[df_copy.columns[x_idx_left]], df_copy[df_copy.columns[x_idx_right]] = \
                df_copy[df_copy.columns[x_idx_right]], df_copy[df_copy.columns[x_idx_left]]
        if mirror_y:
            df_copy[df_copy.columns[y_idx_left]], df_copy[df_copy.columns[y_idx_right]] = \
                df_copy[df_copy.columns[y_idx_right]], df_copy[df_copy.columns[y_idx_left]]
        if mirror_z:
            df_copy[df_copy.columns[z_idx_left]], df_copy[df_copy.columns[z_idx_right]] = \
                df_copy[df_copy.columns[z_idx_right]], df_copy[df_copy.columns[z_idx_left]]

    return df_copy


def rotate(df: pd.DataFrame, rotation: float = 0.0, axis: int = 0):
    '''
    Rotates the node coordinates of the input kinect DataFrame by "rotation".

    OBS: FrameNo column must be removed beforehand.

    Args:
        df: DataFrame containing nodes to be rotated.
        rotation (0 - 2π): The amount to rotate the nodes by, 0 being no rotation and 2π being a full rotation (counter-clockwise).
        axis: 0 = z_axis, 1 = y_axis, 2 = x_axis

    Returns:
        Copy of the original DataFrame with rotated coordinates
    '''
    cartesian_endings = ["_x", "_y", "_z"]

    used_bodyparts = set()

    df_copy = df.copy()
    columns = df_copy.columns

    for column in columns:
        parts = column.split("_")

        if len(parts) == 2:
            bodypart = parts[0]
            suffix = parts[1]
        else:
            bodypart = f"{parts[0]}_{parts[1]}"
            suffix = parts[2] if len(parts) > 2 else None

        if suffix in cartesian_endings:
            if bodypart not in used_bodyparts:
                x_col = f"{bodypart}_x"
                y_col = f"{bodypart}_y"
                z_col = f"{bodypart}_z"
                
                # Check if all columns exist
                if all(col in df_copy.columns for col in [x_col, y_col, z_col]):
                    # Apply 2D rotation
                    # Rotation around z-axis
                    if axis == 0:
                        r_col = np.sqrt(df_copy[x_col]**2 + df_copy[y_col]**2)
                        t_col = np.arctan(df_copy[y_col]/df_copy[x_col])

                        t_col = t_col + rotation
                        
                        df_copy[x_col] = r_col * np.cos(t_col)
                        df_copy[y_col] = r_col * np.sin(t_col)
                    
                    # Rotation around y-axis
                    elif axis == 1:
                        r_col = np.sqrt(df_copy[x_col]**2 + df_copy[z_col]**2)
                        t_col = np.arctan(df_copy[z_col]/df_copy[x_col])

                        t_col = t_col + rotation
                        
                        df_copy[x_col] = r_col * np.cos(t_col)
                        df_copy[z_col] = r_col * np.sin(t_col)
                    
                    # Rotation around x-axis
                    else:
                        r_col = np.sqrt(df_copy[z_col]**2 + df_copy[y_col]**2)
                        t_col = np.arctan(df_copy[y_col]/df_copy[z_col])

                        t_col = t_col + rotation
                        
                        df_copy[z_col] = r_col * np.cos(t_col)
                        df_copy[y_col] = r_col * np.sin(t_col)

                used_bodyparts.add(bodypart)

    return df_copy


# Testing functions
def main():
    path = "MainProject/data/mediapipe_removed_non_squat/A1_mediapipe.csv"
    df = pd.read_csv(path)
    df.drop(labels=["FrameNo"], axis=1, inplace=True)
    # print(df.head()[["left_foot_x", "right_foot_x"]])
    # print(mirror(df).head()[["left_foot_x", "right_foot_x"]])
    print(rotate(df, 0.5).head())

if __name__ == "__main__":
    main()