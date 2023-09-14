import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats

def instrument(throughput):
    """
    Instrument throughput/efficiency curve.

    Parameters
    ----------
    throughput : type

    Returns
    -------
    type
        constant value

    Raises
    ------
    ExceptionType
        Description of when this exception is raised.
    """

    constant=0.8

    return throughput*constant
