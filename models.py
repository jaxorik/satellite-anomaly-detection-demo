# Function to build and train an autoencoder for anomaly detection (BLACK BOX MODEL)

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.linear_model import Ridge

tf.random.set_seed(42)

def build_black_box(X_train, X_test, y_test, scaler = None):
    """
    Build and train black box autoencoder for anomaly detection.
    Returns the model, predictions, and reconstruction errors.
    """
    # Scale the data
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    # Define the input dimension
    input_dim = X_train.shape[1]

    # Define the encoder
    encoding_dim = 2 # Compressed representation size

    # Build the autoencoder black box model
    input_layer = Input(shape = (input_dim,))

    # Encoder layers
    encoded = Dense(8, activation = 'relu')(input_layer) # ReLU is a traditionally used activation function
    decoded = Dense(input_dim, activation = 'sigmoid')(encoded) # Sigmoid is traditionally used activation function

    # Autoencoder model
    autoencoder = Model(inputs = input_layer, outputs = decoded)

    # Separate encoder model for visualization
    encoder = Model(inputs=input_layer, outputs=encoded)

    # Compile the model
    autoencoder.compile(optimizer = 'adam', loss = 'mse') # Adam optimization and mean squard error

    # Train the model
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights=True)
    history = autoencoder.fit(
        X_train_scaled, X_train_scaled,
        epochs = 100,
        batch_size = 32,
        shuffle = True,
        validation_split = 0.2,
        callbacks = [early_stopping],
        verbose = 0
    )

    # Get reconstruction error on test set
    X_test_pred = autoencoder.predict(X_test_scaled)
    mse = np.mean(np.square(X_test_scaled - X_test_pred), axis = 1)

    # Find threshold for anomaly detection
    # Here we will use percentile-based approach, but could use other methods.
    threshold = np.percentile(mse, 95) # Assume top 5% are anomalous

    # Make predictions based on reconstruction error
    y_pred = (mse > threshold).astype(int)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)

    return autoencoder, encoder, y_pred, mse, threshold, accuracy, scaler, history    

from sklearn.tree import DecisionTreeClassifier, plot_tree

# Function to train and evaluate transparent model (Decision Tree)
def train_transparent_model(X_train, y_train, X_test, y_test, feature_names):
    """
    Train a transparent decision tree model.
    Returns the model, predictions on test data, and tree visualization.
    """
    # Create a decision tree classifier
    model = DecisionTreeClassifier(
        max_depth=4, # Limit tree depth for interpretability
        random_state = 42
    )

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = model.score(X_test, y_test)

    return model, y_pred, accuracy

# Function to explain the transparent model prediction for a sample
def explain_transparent_prediction(model, X_sample, feature_names):
    """
    Trace the decision path for a sample through the decision tree
    Returns a list of decision steps
    """
    # Get the decision path
    decision_path = model.decision_path(X_sample.reshape(1, -1))

    # Get the node feature indices, thresholds, and children
    n_nodes = model.tree_.node_count
    feature = model.tree_.feature
    threshold = model.tree_.threshold

    # Get leaf value (prediction)
    leaf_id = model.apply(X_sample.reshape(1, -1))[0]
    leaf_value = model.tree_.value[leaf_id][0]
    prediction = np.argmax(leaf_value)

    # Trace the path
    node_index = 0
    path = []

    while feature[node_index] != -2: # -2 indicates a leaf node
        if X_sample[feature[node_index]] <= threshold[node_index]:
            path.append(f"{feature_names[feature[node_index]]} <= {threshold[node_index]:.2f} ✓")
            node_index = model.tree_.children_left[node_index]
        else:
            path.append(f"{feature_names[feature[node_index]]} > {threshold[node_index]:.2f} ✓")
            node_index = model.tree_.children_right[node_index]

    path.append(f"Prediction: {'ANOMALY' if prediction == 1 else 'NORMAL'}")

    return path

# Reservoir computing implementation

class ReservoirComputing:
    def __init__(self, n_inputs, n_reservoir=200, spectral_radius=0.95, sparsity=0.1,
                 leak_rate = 0.7, noise=0.001, random_state=42):
        """
        Initialize a Reservoir Computing model.

        Parameters:
        - n_inputs: Numner of input features
        - n_reservoir: Number of nodes (size) of the reservoir
        - spectral_radius: Radius of the reservoir weights (usually less than 1)
        - sparsity: Sparsity of the reservoir weights
        - leak_rate: Leak rate of the reservoir (usually between 0.3 and 0.9)
        - noise: Amount of noise added to the reservoir state
        - random_state: Random seed for reproducibility
        """
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leak_rate = leak_rate
        self.noise = noise

        if random_state is not None:
            np.random.seed(random_state)

        # Initialize input weights (W_in)
        self.W_in = np.random.randn(n_reservoir, n_inputs) * 0.1

        # Initialize reservoir weights (W)
        # Create sparse random matrix
        W = np.random.rand(n_reservoir, n_reservoir) - 0.5
        # Make it sparse
        W[np.random.rand(*W.shape) > sparsity] = 0
        # Compute the spectral radius
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        # Rescale to desired spectral radius
        self.W = W * (spectral_radius / radius) if radius > 0 else W

        # Initialize readout with Ridge regeression
        self.readout = Ridge(alpha=1e-6)

        # For visualization and interpretation
        self.feature_weights = None
        self.reservoir_weights = None

    def _update_reservoir(self, x, r):
        """
        Updates the reservoir state.

        Parameters:
        - x: Input data point
        - r: Current reservoir state

        Returns: update reservoir state
        """
        # Compute new reservoir state
        r_new = np.tanh(np.dot(self.W_in, x) + np.dot(self.W, r)) # tanh this time because why not use all the typical functions?
        # Add noise
        r_new += self.noise * np.random.randn(self.n_reservoir)
        r_new = (1 - self.leak_rate) * r + self.leak_rate * r_new
        return r_new

    def _compute_reservoir_states(self, X):
        """
        Compute reservoir states for all input data.

        Parameters:
        - X: Input data of shape (n_samples, n_features)

        Returns: Reservoir state of all samples
        """
        n_samples = X.shape[0]

        # Initialize reservoir states
        r = np.zeros((n_samples, self.n_reservoir))

        # Compute reservoir states for each sample
        r_t = np.zeros(self.n_reservoir)
        for t in range(n_samples):
            r_t = self._update_reservoir(X[t], r_t)
            r[t] = r_t

        return r

    def fit(self, X, y):
        """
        Train the reservoir computer

        Parameters:
        - X: Input data of shape (n_samples, n_features)
        - y: Target labels

        Returns: self
        """
        # Compute reservoir states
        r = self._compute_reservoir_states(X)

        # Train readout using ridge regression
        self.readout.fit(r, y)

        # Save weights for interpretation
        self.reservoir_weights = self.readout.coef_

        # Calculate direct connection weights from features to output
        # This helps with interpretation
        self.feature_weights = np.zeros(self.n_inputs)
        for i in range(self.n_inputs):
            # Calculate the weighted sum of reservoir responses to each input feature
            feature_influence = np.dot(self.W_in[:, i], self.reservoir_weights)
            self.feature_weights[i] = np.abs(feature_influence).mean()

        # Normalize feature weights for easier interpretation
        if np.sum(self.feature_weights) > 0:
            self.feature_weights = self.feature_weights / np.sum(self.feature_weights)

        return self

    def predict(self, X):
        """
        Make predictions using the trained reservoir model.

        Parameters:
        - X: Input data of shape (n_samples, n_features)

        Returns:
        - Predicted labels
        """
        # Compute reservoir states
        r = self._compute_reservoir_states(X)

        # Make predictions using the readout
        y_pred = self.readout.predict(r)

        # Convert to binary predictions for classification
        return (y_pred > 0.5).astype(int)

    def predict_proba(self, X):
        """
        Predict probability estimates.

        Parameters:
        - X: Input data

        Returns:
        - Probability estimates
        """
        # Compute reservoir states
        r = self._compute_reservoir_states(X)

        # Get raw predictions
        y_pred = self.readout.predict(r)

        # Clip to [0, 1] range and reshape for scikit-learn compatibility
        y_pred = np.clip(y_pred, 0, 1)

        # Return probabilities for both classes
        return np.column_stack([1 - y_pred, y_pred])

    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance based on the reservoir's response to inputs.

        Parameters:
        - feature_names: Names of the input features

        Returns:
        - Dictionary mapping feature names to importance scores
        """
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(self.n_inputs)]

        return dict(zip(feature_names, self.feature_weights))

    def explain_prediction(self, x, feature_names=None):
        """
        Explain a prediction by showing the contribution of each feature.

        Parameters:
        - x: Input data point
        - feature_names: Names of the input features

        Returns:
        - Dictionary with explanation details
        """
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(self.n_inputs)]

        # Get reservoir state for this input
        r_t = np.zeros(self.n_reservoir)
        r_t = self._update_reservoir(x, r_t)

        # Get individual reservoir neuron activations
        neuron_activations = r_t

        # Get prediction
        pred = self.readout.predict(r_t.reshape(1, -1))[0]

        # Calculate feature contributions
        feature_contributions = {}
        for i, name in enumerate(feature_names):
            # For each feature, calculate its contribution through the reservoir
            direct_effect = self.W_in[:, i] * x[i]
            weighted_effect = np.dot(direct_effect, self.reservoir_weights)
            feature_contributions[name] = weighted_effect

        # Normalize contributions
        total = sum(abs(v) for v in feature_contributions.values())
        if total > 0:
            normalized_contributions = {k: v/total for k, v in feature_contributions.items()}
        else:
            normalized_contributions = feature_contributions

        return {
            'prediction': pred,
            'binary_prediction': 1 if pred > 0.5 else 0,
            'reservoir_state': neuron_activations,
            'feature_contributions': normalized_contributions
        }

  # Train reservoir computing model for anomaly detection
def train_reservoir_model(X_train, y_train, X_test, y_test, feature_names, scaler=None):
    """
    Train a transparent reservoir computing model.

    Returns the model, predictions, and feature importances.
    """
    # Scale the data
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    # Create and train the reservoir model
    model = ReservoirComputing(
        n_inputs=X_train.shape[1],
        n_reservoir=100,  # Number of reservoir neurons
        spectral_radius=0.9,
        sparsity=0.1,
        noise=0.001,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)

    # Get feature importance
    feature_importance = model.get_feature_importance(feature_names)

    return model, y_pred, accuracy, scaler, feature_importance
