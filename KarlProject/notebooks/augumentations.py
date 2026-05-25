import numpy as np
import pandas as pd

def angle(point1, point2, point3):
    """
    Compute the angle between three points (in degrees).
    point2 is the vertex of the angle.

    Parameters:
    point1, point2, point3: tuples or lists of (x, y, z) coordinates

    Returns:
    angle in degrees
    """
    # Convert to numpy arrays
    p1 = np.array([point1[0], point1[1], point1[2]])
    p2 = np.array([point2[0], point2[1], point2[2]])
    p3 = np.array([point3[0], point3[1], point3[2]])

    # Vectors from the vertex
    v1 = p1 - p2
    v2 = p3 - p2

    # Calculate the cosine of the angle using dot product
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    # Clip to avoid numerical issues
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Convert to degrees
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def compute_angles(df):
    """
    Compute angles between different joints and append them to the dataframe.

    Parameters:
    df: pandas DataFrame with joint coordinates

    Returns:
    DataFrame with angle columns appended before the last column
    """
    # Create a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()

    # Dictionary to store computed angles
    angles_dict = {}

    # For each row, compute angles
    for idx, row in df.iterrows():
        # Get coordinates for each joint
        # Format: joint_x, joint_y, joint_z
        joints = {}

        # Extract all joints (matching the column names in your data)
        column_groups = {
            'head': ['head_x', 'head_y', 'head_z'],
            'left_shoulder': ['left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z'],
            'left_elbow': ['left_elbow_x', 'left_elbow_y', 'left_elbow_z'],
            'right_shoulder': ['right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z'],
            'right_elbow': ['right_elbow_x', 'right_elbow_y', 'right_elbow_z'],
            'left_hand': ['left_hand_x', 'left_hand_y', 'left_hand_z'],
            'right_hand': ['right_hand_x', 'right_hand_y', 'right_hand_z'],
            'left_hip': ['left_hip_x', 'left_hip_y', 'left_hip_z'],
            'right_hip': ['right_hip_x', 'right_hip_y', 'right_hip_z'],
            'left_knee': ['left_knee_x', 'left_knee_y', 'left_knee_z'],
            'right_knee': ['right_knee_x', 'right_knee_y', 'right_knee_z'],
            'left_foot': ['left_foot_x', 'left_foot_y', 'left_foot_z'],
            'right_foot': ['right_foot_x', 'right_foot_y', 'right_foot_z']
        }

        for joint_name, coord_cols in column_groups.items():
            if all(col in row.index for col in coord_cols):
                joints[joint_name] = (row[coord_cols[0]], row[coord_cols[1]], row[coord_cols[2]])

        # Define angle definitions (joint1, vertex, joint2)
        angle_definitions = {
            'left_elbow_angle': ('left_shoulder', 'left_elbow', 'left_hand'),
            'right_elbow_angle': ('right_shoulder', 'right_elbow', 'right_hand'),
            'left_shoulder_angle': ('left_hip', 'left_shoulder', 'left_elbow'),
            'right_shoulder_angle': ('right_hip', 'right_shoulder', 'right_elbow'),
            'left_knee_angle': ('left_hip', 'left_knee', 'left_foot'),
            'right_knee_angle': ('right_hip', 'right_knee', 'right_foot'),
            'left_hip_angle': ('left_shoulder', 'left_hip', 'left_knee'),
            'right_hip_angle': ('right_shoulder', 'right_hip', 'right_knee')
        }

        # Compute each angle if all required joints exist
        for angle_name, (joint1, vertex, joint2) in angle_definitions.items():
            if all(j in joints for j in [joint1, vertex, joint2]):
                angles_dict.setdefault(angle_name, []).append(
                    angle(joints[joint1], joints[vertex], joints[joint2])
                )
            else:
                angles_dict.setdefault(angle_name, []).append(np.nan)

    # Add angle columns to the result dataframe
    for angle_name, angle_values in angles_dict.items():
        df_copy[angle_name] = angle_values

    # Reorder columns: keep original columns except move 'running_video' to the end
    original_columns = [col for col in df_copy.columns if col != 'running_video' and col not in angles_dict]
    angle_columns = list(angles_dict.keys())
    final_columns = original_columns + angle_columns + ['running_video']

    df_copy = df_copy[final_columns]

    return df_copy


def mirror(df: pd.DataFrame, mirror_x: bool = True, mirror_y: bool = True, mirror_z: bool = True) -> pd.DataFrame:
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
        (11, 12)  # feet
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
