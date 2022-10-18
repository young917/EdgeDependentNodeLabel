import numpy as np
def f_point_5(prec, rec):
    """
    Calculate F_0.5 score
    :param prec: precision values, a vector
    :param rec: recall values, a vector
    :return: the corresponding F_0.5 score
    """
    if hasattr(prec, '__len__') != hasattr(prec, '__len__'):
        return np.array([])
    if hasattr(prec, '__len__'):
        assert len(prec) == len(rec), "The length of prec and recall should be the same!"

    return (1 + 0.25) * prec * rec / (0.25 * prec + rec)

def argmax(v1, v2):
    """
    Find the index of maximum values of v1.
    If multiple elements equals maximum value, then check v2.
    If v2 are also the same, then return the first ond
    Not very efficient, but it is good enough for current usage.
    :param v1: vector 1
    :param v2: vector2
    :return: a single scala
    """
    assert len(v1) == len(v2), "the length of the two vector must be the same!"
    if isinstance(v1, list):
        v1 = np.array(v1)
    if isinstance(v2, list):
        v2 = np.array(v2)

    v1 = np.squeeze(v1)
    v2 = np.squeeze(v2)
    idx = v1 == np.max(v1)
    v2[np.logical_not(idx)] = np.min(v2) - 1
    return np.unravel_index(np.argmax(v2), v2.shape)
