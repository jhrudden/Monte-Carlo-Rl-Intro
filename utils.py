from typing import Callable, Any, List, Dict, Tuple
import os
import pickle

# ChatGPT was heavily used to generate `load_or_compute_and_cache` and `line_cross` functions

def load_or_compute_and_cache(file_path: str, compute_function: Callable, *args, post_process: Callable = None,  **kwargs) -> Any:
    """
    Loads data from a file if it exists and is stable, otherwise computes the data, caches it, and then returns it.

    Args:
        file_path: The path to the file where the data is cached.
        compute_function: A function that computes the data if it's not cached.
        args: Positional arguments to pass to the compute function.
        post_process: A function that post-processes the computed data.
        kwargs: Keyword arguments to pass to the compute function.
    
    Returns:
        The data, either loaded from the file or computed by the compute function.
    """
    # Check if the data is cached and load it
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    # If not cached, compute the data
    data = compute_function(*args, **kwargs)

    if post_process is not None:
        data = post_process(data)

    # Cache the data
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

    return data

def policy_to_dict(policy: Callable, states: List[Any]) -> Dict[Tuple[int, int, bool], int]:
    """
    Convert a policy function to a dictionary mapping state to action.

    Args:
        policy: The policy function to convert.
        states: A list of all possible states.
    
    Returns:
        A dictionary mapping state to action.
    """
    return {state: policy(state) for state in states}

def line_cross(A, B, C, D):
    """
    Check if line segment CD crosses line segment AB.

    Args:
        A, B: Points defining the finish line
        C, D: Points defining the trajectory

    Returns:
        bool: True if CD crosses AB, False otherwise.
    """
    # Function to calculate the determinant, useful for checking the direction
    def det(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    # Check if line segments intersect
    denom = (D[0] - C[0]) * (B[1] - A[1]) - (D[1] - C[1]) * (B[0] - A[0])
    if denom == 0:
        return False  # Lines are parallel or coincident
    
    t_num = (A[0] - C[0]) * (D[1] - C[1]) - (A[1] - C[1]) * (D[0] - C[0])
    u_num = (A[0] - C[0]) * (B[1] - A[1]) - (A[1] - C[1]) * (B[0] - A[0])
    t = t_num / denom
    u = u_num / denom

    # Intersection check
    if 0 <= t <= 1 and 0 <= u <= 1:
        # Check if the trajectory crosses the finish line
        det_C = det(A, B, C)
        det_D = det(A, B, D)
        return det_C * det_D < 0  # The signs are different, indicating a crossing

    return False
