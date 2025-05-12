import numpy as np
from time import time
import pickle

class KohonenLayer:
    def __init__(self, dims_in, dims_out):
        """
        Initializes the Kohonen Layer.

        Args:
            dims_in (int): Number of input dimensions.
            dims_out (tuple): Dimensions of the output grid (e.g., (rows, cols)).
        """
        self.dims_in = dims_in
        self.dims_out = dims_out
        self.grid_rows, self.grid_cols = dims_out  # Unpack for convenience
        self.weights = np.random.rand(self.grid_rows, self.grid_cols, dims_in)  # Initialize weights
        self.learning_rate = 0.1
        self.neighborhood_radius = max(dims_out) // 2  # Initial radius (can decay)

    def _calculate_distance(self, input_vector):
        """
        Calculates the Euclidean distance between the input and all neurons.

        Args:
            input_vector (numpy.ndarray): The input vector.

        Returns:
            numpy.ndarray: A matrix of distances, shape (grid_rows, grid_cols).
        """

        distances = np.linalg.norm(self.weights - input_vector, axis=2)
        return distances

    def _find_best_matching_unit(self, input_vector):
        """
        Finds the Best Matching Unit (BMU) for a given input.

        Args:
            input_vector (numpy.ndarray): The input vector.

        Returns:
            tuple: The row and column indices of the BMU.
        """
        distances = self._calculate_distance(input_vector)
        # np.unravel_index converts a flat index to a tuple of coordinates
        bmu_index_flat = np.argmin(distances)
        return np.unravel_index(bmu_index_flat, (self.grid_rows, self.grid_cols))

    def _calculate_neighborhood(self, bmu_index):
        """
        Calculates the neighborhood indices based on the radius.

        Args:
            bmu_index (tuple): The row and column indices of the BMU.

        Returns:
            list: A list of (row, col) indices within the neighborhood.
        """
        neighborhood = []
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                dist = np.linalg.norm(np.array(bmu_index) - np.array((r, c)))
                if dist <= self.neighborhood_radius:
                    neighborhood.append((r, c))
        return neighborhood

    def update_weights(self, input_vector, bmu_index, neighborhood):
        """
        Updates the weights of the BMU and its neighbors.

        Args:
            input_vector (numpy.ndarray): The input vector.
            bmu_index (tuple): The row and column indices of the BMU.
            neighborhood (list): A list of (row, col) indices in the neighborhood.
        """
        for r, c in neighborhood:
            influence = np.exp(-np.linalg.norm(np.array(bmu_index) - np.array((r, c))) / (2 * self.neighborhood_radius**2))
            self.weights[r, c] += self.learning_rate * influence * (input_vector - self.weights[r, c])

    def decay_parameters(self, epoch, total_epochs):
        """
        Decays the learning rate and neighborhood radius over time.

        Args:
            epoch (int): The current epoch.
            total_epochs (int): The total number of epochs.
        """
        self.learning_rate = 0.1 * np.exp(-epoch / total_epochs)
        self.neighborhood_radius = max(1, max(self.dims_out) * np.exp(-epoch / total_epochs))

    def train(self, data, epochs, decay=True):
        """
        Trains the Kohonen network with the given data.

        Args:
            data (numpy.ndarray): The training data, shape (n_samples, dims_in).
            epochs (int): The number of training epochs.
            decay (bool, optional): Whether to decay learning rate and radius. Defaults to True.
        """

        for epoch in range(epochs):
            t0 = time()
            for input_vector in data:
                bmu_index = self._find_best_matching_unit(input_vector)
                neighborhood = self._calculate_neighborhood(bmu_index)
                self.update_weights(input_vector, bmu_index, neighborhood)

            if decay:
                self.decay_parameters(epoch, epochs)

            t1 = time()
            print(f"Epoch {epoch + 1}/{epochs}, Time: {t1 - t0:.4f}s, Learning Rate: {self.learning_rate:.4f}, Radius: {self.neighborhood_radius:.4f}")

    def predict(self, data):
        """
        Predicts the BMU for each input in the given data.

        Args:
            data (numpy.ndarray): The input data, shape (n_samples, dims_in).

        Returns:
            list: A list of (row, col) tuples representing the BMUs for each input.
        """
        predictions = []
        for input_vector in data:
            predictions.append(self._find_best_matching_unit(input_vector))
        return predictions


class KohonenNetwork:
    def __init__(self, dims_in, dims_out, layer=None):
        """
        Initializes the Kohonen Network.

        Args:
            dims_in (int): Number of input dimensions.
            dims_out (tuple): Dimensions of the output grid (e.g., (rows, cols)).
            layer (KohonenLayer, optional):  A pre-initialized KohonenLayer. If None, 
                                            a new layer is created. Defaults to None.
        """
        self.layer = layer if layer else KohonenLayer(dims_in, dims_out)

    def train(self, data, epochs, decay=True):
        """
        Trains the Kohonen Network.

        Args:
            data (numpy.ndarray): The training data.
            epochs (int): Number of training epochs.
            decay (bool, optional): Whether to decay parameters. Defaults to True.
        """
        self.layer.train(data, epochs, decay)

    def predict(self, data):
        """
        Predicts using the Kohonen Network.

        Args:
            data (numpy.ndarray): The input data.

        Returns:
            list: The predictions (BMUs).
        """
        return self.layer.predict(data)

    def save(self, filename):
        """Saves the trained Kohonen network to a file."""
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """Loads a trained Kohonen network from a file."""
        with open(filename, "rb") as file:
            return pickle.load(file)


