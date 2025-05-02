import numpy as np
import time
import random
import string

def compute_quadratic_transform(A, B):
    """
    Computes a 2nd-order polynomial transformation that maps A to B.

    Parameters:
    - A: (N, 2) array of source points
    - B: (N, 2) array of target points

    Returns:
    - coeff_x: Coefficients [a0, a1, a2, a3, a4, a5] for x' equation
    - coeff_y: Coefficients [b0, b1, b2, b3, b4, b5] for y' equation
    """
    A = np.array(A)
    B = np.array(B)
    
    assert A.shape == B.shape, "Point sets must have the same shape"
    assert A.shape[0] >= 6, "Need at least 6 points for quadratic fitting"

    # Construct the matrix for quadratic terms
    X = np.column_stack([
        np.ones(len(A)),   # Bias term (a0, b0)
        A[:, 0],           # x
        A[:, 1],           # y
        A[:, 0] ** 2,      # x^2
        A[:, 0] * A[:, 1], # xy
        A[:, 1] ** 2       # y^2
    ])

    # Solve for least squares coefficients
    coeff_x, _, _, _ = np.linalg.lstsq(X, B[:, 0], rcond=None)
    coeff_y, _, _, _ = np.linalg.lstsq(X, B[:, 1], rcond=None)

    return coeff_x, coeff_y

def apply_quadratic_transform(A, coeff_x, coeff_y):
    """
    Applies the quadratic transformation to a set of points.

    Parameters:
    - A: (N, 2) array of source points
    - coeff_x: Coefficients for x' equation
    - coeff_y: Coefficients for y' equation

    Returns:
    - Transformed points (N, 2)
    """
    A = np.array(A)

    X = np.column_stack([
        np.ones(len(A)),   # Bias term
        A[:, 0],           # x
        A[:, 1],           # y
        A[:, 0] ** 2,      # x^2
        A[:, 0] * A[:, 1], # xy
        A[:, 1] ** 2       # y^2
    ])

    x_new = X @ coeff_x
    y_new = X @ coeff_y

    return np.column_stack([x_new, y_new])

def get_digits_after_last_period(s):
    last_period_index = s.rfind('.')
    if last_period_index == -1:
        return ''  # no period found
    # Get substring after the last period
    after_period = s[last_period_index + 1:]
    # Filter only digits
    digits = ''.join(filter(str.isdigit, after_period))
    return digits

def generate_unique_id(length=8):
    """
    Generate a unique ID of the specified length.
    Combines a timestamp and random characters for uniqueness.
    """
    # Characters to use for random generation
    chars = string.ascii_letters + string.digits
    
    # Add a timestamp for uniqueness (converted to base-36)
    result = base36_encode(int(time.time() * 1000))
    
    # Add random characters to meet the desired length
    while len(result) < length:
        result += random.choice(chars)
    
    return result[:length]

def base36_encode(number):
    """Convert a number to a base-36 string."""
    if number < 0:
        raise ValueError("Base36 encoding only supports positive integers")
    if number == 0:
        return "0"
    
    base36 = []
    while number:
        number, remainder = divmod(number, 36)
        base36.append("0123456789abcdefghijklmnopqrstuvwxyz"[remainder])
    
    return "".join(reversed(base36))