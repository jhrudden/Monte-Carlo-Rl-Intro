from typing import Callable, Any, List, Dict, Tuple
import os
import pickle

# ChatGPT was heavily used to generate `load_or_compute_and_cache` function

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
