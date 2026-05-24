import numpy as np 
def select_equally_spaced_frames(data, num_frames=30, axis=0):
    """
    Select equally spaced frames from n-dimensional data along specified axis.
    
    Parameters:
    -----------
    data : numpy array or list
        Input data (2D, 4D, or any dimension)
    num_frames : int
        Number of frames to select (default: 30)
    axis : int
        Axis along which to select frames (default: 0, typically the time/frame dimension)
    
    Returns:
    --------
    numpy array
        Selected frames with same dimensions as input except the selected axis
    """
    # Convert to numpy array if it's a list
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    # Get total number of frames along the specified axis
    total_frames = data.shape[axis]
    
    # Generate equally spaced indices
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

    
    # Select using advanced indexing
    # Build a tuple of slices for indexing
    selector = [slice(None)] * data.ndim
    selector[axis] = indices
    
    return data[tuple(selector)]

