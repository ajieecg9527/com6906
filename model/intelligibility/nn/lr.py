import numpy as np
from scipy.optimize import curve_fit


class Logistic:
    """Class to represent the mapping from mbstoi parameters to intelligibility scores.
    The mapping uses a simple logistic function scaled between 0 and 100.
    The mapping parameters need to fit first using mbstoi, intelligibility score
    pairs, using fit().
    Once the fit has been made predictions can be made by calling predict()
    """

    params = None  # The model params

    def _logistic_mapping(self, x, x0, k):
        """
        Logistic function
            x0 - x value of the logistic's midpoint
            k - the logistic growth rate or steepness of the curve
        """
        L = 100  # correctness can't be over 100
        return L / (1 + np.exp(-k * (x - x0)))

    def fit(self, pred, intel):
        """Fit a mapping betweeen mbstoi scores and intelligibility scores."""
        initial_guess = [50.0, 1.0]  # Initial guess for parameter values
        self.params, *_remaining_returns = curve_fit(self._logistic_mapping, pred, intel, initial_guess)

    def predict(self, x):
        """Predict intelligilbity scores from mbstoi scores."""
        # Note, fit() must be called before predictions can be made
        assert self.params is not None
        return self._logistic_mapping(x, self.params[0], self.params[1])
