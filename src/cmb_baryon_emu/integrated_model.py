"""TensorFlow/Keras-based integrated emulator models for cosmological predictions.

This module provides the IntegratedModel class which wraps TensorFlow/Keras
neural network models for fast cosmological predictions. It handles input/output
scaling, PCA compression, log preprocessing, Chebyshev preprocessing, and A_s rescaling.

The module includes custom activation functions and loss functions designed
for cosmological emulation tasks.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout

from cmb_baryon_emu.module import *  # noqa

import pickle

from tensorflow.keras.models import load_model
from tqdm import tqdm

from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback


def create_model(input_dim, hidden_layers=[64, 32], output_dim=1):
    """Create a neural network model with custom activation.

    Args:
        input_dim (int): Number of input features (cosmological parameters).
        hidden_layers (list, optional): List of integers specifying the number
            of units in each hidden layer. Defaults to [64, 32].
        output_dim (int, optional): Number of output dimensions (e.g., number
            of ℓ modes in power spectrum). Defaults to 1.

    Returns:
        Model: Compiled Keras model with custom activation functions.
    """
    inputs = Input(shape=(input_dim,))
    x = inputs

    # Add hidden layers with custom activation
    for units in hidden_layers:
        x = Dense(units)(x)
        x = CustomActivation()(x)

    outputs = Dense(output_dim)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam", loss="mse"
    )  # Using Mean Squared Error for simplicity
    return model


def create_model_with_dropout(input_dim, hidden_layers, output_dim, dropout_rate=0.1):
    """Create a neural network model with custom activation and dropout.

    Args:
        input_dim (int): Number of input features (cosmological parameters).
        hidden_layers (list, optional): List of integers specifying the number
            of units in each hidden layer. Defaults to [64, 32].
        output_dim (int, optional): Number of output dimensions (e.g., number
            of ℓ modes in power spectrum). Defaults to 1.
        dropout_rate (float, optional): Dropout rate for each hidden layer.
            Defaults to 0.1.

    Returns:
        Model: Compiled Keras model with custom activation functions and dropout.
    """
    inputs = Input(shape=(input_dim,))
    x = inputs
    for units in hidden_layers:
        x = Dense(units)(x)
        x = CustomActivation()(x)
        x = Dropout(dropout_rate)(x)  # Add this!
    outputs = Dense(output_dim)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


class CustomActivation(Layer):
    """Custom activation function matching arXiv:2106.03846.

    This activation function uses trainable parameters alpha and beta to
    create a learnable activation: (β + sigmoid(αx) * (1 - β)) * x

    This is designed for cosmological emulation tasks where the activation
    shape can significantly impact model performance.
    """

    def build(self, input_shape):
        """Build the layer by creating trainable weights.

        Args:
            input_shape: Shape of the input tensor.
        """
        # Trainable weight variables for alpha and beta initialized with random normal distribution
        self.alpha = self.add_weight(
            name="alpha",
            shape=input_shape[1:],
            initializer="random_normal",
            trainable=True,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=input_shape[1:],
            initializer="random_normal",
            trainable=True,
        )
        super(CustomActivation, self).build(input_shape)

    def call(self, inputs):
        """Apply the custom activation function.

        Args:
            inputs: Input tensor.

        Returns:
            Tensor: Transformed input tensor.
        """
        return (self.beta + tf.sigmoid(self.alpha * inputs) * (1 - self.beta)) * inputs


class CustomLoss(Layer):
    """Custom loss function with element-wise weighting.

    This loss function allows for element-wise weighting of the squared error,
    which is useful when different parts of the output spectrum have different
    importance or uncertainty.

    Attributes:
        element_weights (Tensor): Weights for each element of the output.
    """

    def __init__(self, element_weights=None, **kwargs):
        """Initialize the custom loss function.

        Args:
            element_weights (array-like, optional): Weights for each output
                element. If None, all elements are weighted equally.
                Defaults to None.
            **kwargs: Additional keyword arguments passed to parent class.
        """
        super(CustomLoss, self).__init__(**kwargs)
        self.element_weights = (
            tf.convert_to_tensor(element_weights, dtype=tf.float32)
            if element_weights is not None
            else tf.constant(1.0, dtype=tf.float32)
        )

    def call(self, y_true, y_pred):
        """Compute the weighted mean squared error.

        Args:
            y_true: True values.
            y_pred: Predicted values.

        Returns:
            Tensor: Scalar loss value.
        """
        # Compute element-wise squared errors
        squared_errors = tf.square(y_true - y_pred)

        if self.element_weights is not None:
            # Apply element-wise weights
            squared_errors = squared_errors / self.element_weights

        # Compute the mean of squared errors
        return tf.reduce_mean(squared_errors)

    def get_config(self):
        """Get configuration dictionary for serialization.

        Returns:
            dict: Configuration dictionary.
        """
        config = super(CustomLoss, self).get_config()
        config.update({"element_weights": self.element_weights.numpy()})
        return config


class IntegratedModel:
    """Integrated emulator model wrapper for TensorFlow/Keras neural networks.

    This class wraps a trained Keras model with preprocessing/postprocessing
    steps including input/output scaling, PCA compression, log preprocessing,
    Chebyshev preprocessing, and A_s rescaling.
    It provides a unified interface for making cosmological predictions from
    parameter dictionaries.

    Attributes:
        model (Model): Trained Keras model.
        input_scaler (StandardScaler): Scaler for input parameters.
        output_scaler (StandardScaler): Scaler for output predictions.
        parameter_names (list): List of parameter names in order.
        pca (PCA, optional): PCA transformer for output compression.
        pca_scaler (StandardScaler, optional): Scaler for PCA space.
        log_preprocess (bool): Whether to apply log preprocessing.
        cheby_preprocess (bool): Whether to apply Chebyshev preprocessing.
        cheby_meta (dict, optional): Chebyshev metadata for inverse transform.
        rescale_by_as (bool): Whether to rescale by A_s.
        offset (float, optional): Offset for log preprocessing.
        rescaling_factor (float, optional): Rescaling factor.
        temp_file (str, optional): Temporary file path.
        train_losses (list): Training loss history.
        val_losses (list): Validation loss history.
    """

    def __init__(
        self,
        keras_model,
        input_scaler,
        output_scaler,
        parameter_names,
        temp_file=None,
        offset=None,
        log_preprocess=False,
        cheby_preprocess=False,
        cheby_meta=None,
        rescaling_factor=None,
        pca=None,
        pca_scaler=None,
        rescale_by_as=False,
        galaxy_bias_config=None,
    ):
        """Initialize the integrated model.

        Args:
            keras_model (Model): Trained Keras model.
            input_scaler (StandardScaler): Scaler fitted on input parameters.
            output_scaler (StandardScaler): Scaler fitted on output predictions.
            parameter_names (list): List of parameter names in the order
                expected by the model.
            temp_file (str, optional): Path to temporary file. Defaults to None.
            offset (float, optional): Offset for log preprocessing. Defaults to None.
            log_preprocess (bool, optional): Whether to apply log preprocessing
                to inputs. Defaults to False.
            cheby_preprocess (bool, optional): Whether to apply Chebyshev
                preprocessing for oscillatory signals like TE spectra. Defaults to False.
            cheby_meta (dict, optional): Chebyshev metadata containing ell_factor,
                D_mean, D_std, V, c_min, c_denom for inverse transform. Required if
                cheby_preprocess is True. Defaults to None.
            rescaling_factor (float, optional): Rescaling factor for outputs.
                Defaults to None.
            pca (PCA, optional): PCA transformer for output compression.
                Defaults to None.
            pca_scaler (StandardScaler, optional): Scaler for PCA-transformed
                outputs. Defaults to None.
            rescale_by_as (bool, optional): Whether to rescale outputs by
                A_s parameter. Defaults to False.
        """
        self.model = keras_model
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
        self.offset = offset
        self.rescaling_factor = rescaling_factor
        self.temp_file = temp_file
        self.train_losses = []
        self.val_losses = []
        self.log_preprocess = log_preprocess
        self.cheby_preprocess = cheby_preprocess
        self.cheby_meta = cheby_meta
        self.parameter_names = parameter_names
        self.pca = pca
        self.pca_scaler = pca_scaler
        self.rescale_by_as = rescale_by_as
        
        # Galaxy bias rescaling configuration (metadata-driven)
        # Format: {
        #   'enabled': bool,
        #   'linear': [(start_idx, end_idx, bias_param_name), ...],  # cross-spectra scale as b_i
        #   'quadratic': [(start_idx, end_idx, bias_param_name), ...],  # auto-spectra scale as b_i^2
        # }
        # Set to None to disable, or provide dict with 'enabled': True and index ranges
        self.galaxy_bias_config = galaxy_bias_config

    def predict(self, data_dict):
        """Make predictions from cosmological parameters.

        Args:
            data_dict (dict): Dictionary mapping parameter names to arrays
                of parameter values. All arrays must have the same length
                (number of samples).

        Returns:
            ndarray: Predicted outputs. Shape is (n_samples, n_output_features).

        Raises:
            ValueError: If required parameter keys are missing from data_dict.
        """
        if self.parameter_names is not None:
            required_keys = set(sorted(data_dict.keys()))
            expected_keys = set(self.parameter_names)
            if not expected_keys.issubset(required_keys):
                raise ValueError(
                    f"Input dictionary keys {required_keys} do not match expected keys {expected_keys}"
                )

        # Map parameter values to the correct positions in the input matrix
        num_samples = len(
            next(iter(data_dict.values()))
        )  # Assuming all values in data_dict have the same length
        num_features = len(self.parameter_names)
        data = np.empty((num_samples, num_features))
        for i, key in enumerate(self.parameter_names):
            data[:, i] = data_dict[key]

        # Store A_s values if we need to rescale
        amplitudes = None
        if self.rescale_by_as and "ln10^{10}A_s" in self.parameter_names:
            # Find the index of the A_s parameter
            as_index = list(self.parameter_names).index("ln10^{10}A_s")
            ln_10_10_as = data[:, as_index]
            amplitudes = exp(ln_10_10_as) * 1e-10

        scaled_data = (data - self.scaler_mean_in) / self.scaler_scale_in
        prediction = self.model(tf.convert_to_tensor(scaled_data, dtype=tf.float32))

        # Inverse transforms applied in reverse order of training
        
        # Apply inverse PCA transform if PCA is used
        if self.pca is not None:
            prediction = prediction * self.scaler_scale_out + self.scaler_mean_out
            prediction = self.pca.inverse_transform(prediction)
            prediction = self.pca_scaler.inverse_transform(prediction)
        elif self.cheby_preprocess and self.cheby_meta is not None:
            # Inverse Chebyshev transform
            # 1. Unscale from output scaler
            prediction = prediction * self.scaler_scale_out + self.scaler_mean_out
            
            # 2. Undo min-max scaling
            coeffs = prediction * self.cheby_meta['c_denom'] + self.cheby_meta['c_min']
            
            # 3. Reconstruct: coeffs -> standardized D_ell -> D_ell -> C_ell
            y_std = coeffs @ self.cheby_meta['V'].T
            y_D = y_std * self.cheby_meta['D_std'] + self.cheby_meta['D_mean']
            prediction = y_D / self.cheby_meta['ell_factor']
        else:
            prediction = prediction * self.scaler_scale_out + self.scaler_mean_out

        # Inverse offset and log
        if self.log_preprocess:
            prediction = np.exp(prediction) + 2 * self.offset

        # Rescale by A_s if needed
        if self.rescale_by_as and amplitudes is not None:
            prediction = prediction * amplitudes[:, newaxis]

        # Apply galaxy bias rescaling if configured
        prediction = self._apply_galaxy_bias_rescaling(prediction, data_dict)

        return prediction

    def _apply_galaxy_bias_rescaling(self, prediction, data_dict):
        """Apply galaxy bias rescaling to prediction based on configuration.
        
        Auto-spectra (dd_ii, gg_ii) scale as b_i^2 (quadratic).
        Cross-spectra (kad_i, td_i, kg_i, tg_i) scale as b_i (linear).
        
        Args:
            prediction: Array of shape (n_samples, n_output).
            data_dict: Dictionary with bias parameter values.
        
        Returns:
            Rescaled prediction array.
        """
        if self.galaxy_bias_config is None:
            return prediction
        if not self.galaxy_bias_config.get('enabled', False):
            return prediction
        
        # Quadratic rescaling for auto-spectra
        for start_idx, end_idx, bias_name in self.galaxy_bias_config.get('quadratic', []):
            if bias_name in data_dict:
                bias_vals = np.array(data_dict[bias_name])
                prediction[:, start_idx:end_idx] *= (bias_vals ** 2)[:, newaxis]
        
        # Linear rescaling for cross-spectra
        for start_idx, end_idx, bias_name in self.galaxy_bias_config.get('linear', []):
            if bias_name in data_dict:
                bias_vals = np.array(data_dict[bias_name])
                prediction[:, start_idx:end_idx] *= bias_vals[:, newaxis]
        
        return prediction

    def compute_gradient(self, data_dict):
        """Compute gradient of predictions with respect to input parameters.

        Args:
            data_dict (dict): Dictionary mapping parameter names to arrays
                of parameter values.

        Returns:
            ndarray: Gradient of outputs with respect to inputs. Shape is
                (n_samples, n_output_features, n_input_features).

        Raises:
            ValueError: If parameter keys don't match expected keys exactly.
        """
        if self.parameter_names is not None:
            required_keys = set(sorted(data_dict.keys()))
            expected_keys = set(self.parameter_names)
            if required_keys != expected_keys:
                raise ValueError(
                    f"Input dictionary keys {required_keys} do not match expected keys {expected_keys}"
                )

        # Map parameter values to the correct positions in the input matrix
        # data = np.stack([data_dict[key] for key in self.parameter_names], axis=1)
        num_samples = len(
            next(iter(data_dict.values()))
        )  # Assuming all values in data_dict have the same length
        num_features = len(self.parameter_names)
        data = np.empty((num_samples, num_features))
        for i, key in enumerate(self.parameter_names):
            data[:, i] = data_dict[key]

        # Convert the dictionary to an input matrix and scale it
        scaled_data = self.input_scaler.transform(data)

        # Convert to TensorFlow tensor
        input_tensor = tf.convert_to_tensor(scaled_data, dtype=tf.float32)

        # Use GradientTape to compute the gradient
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            # Compute model prediction first
            prediction = self.model(input_tensor)
            # Apply PCA transform if PCA is used
            if self.pca is not None:
                prediction = self.pca.transform(prediction)
                prediction = self.output_scaler.inverse_transform(prediction)
            else:
                prediction = self.output_scaler.inverse_transform(prediction)

        # Compute the gradient of the output with respect to the inputs
        gradient = tape.gradient(prediction, input_tensor)

        # Convert gradient tensor to numpy array and return
        if gradient is None:
            raise ValueError("Gradient computation failed - gradient is None")
        # Handle both TensorFlow tensors and numpy arrays
        if hasattr(gradient, "numpy"):
            return gradient.numpy()
        else:
            # Already a numpy array
            return np.asarray(gradient)

    def train(self, X, y, learning_rates=[1e-3, 5e-4, 1e-4, 5e-5], *args, **kwargs):
        """Train the model with learning rate scheduling.

        Args:
            X (ndarray): Training input data. Shape (n_samples, n_features).
            y (ndarray): Training target data. Shape (n_samples, n_outputs).
            learning_rates (list, optional): List of learning rates to use
                sequentially. Defaults to [1e-3, 5e-4, 1e-4, 5e-5].
            *args: Additional positional arguments passed to model.fit().
            **kwargs: Additional keyword arguments passed to model.fit(),
                including 'epochs', 'validation_split', 'batch_size', etc.
        """
        # Scale the training data
        X_scaled = self.input_scaler.transform(X)
        y_scaled = self.output_scaler.transform(y)

        batch_size = kwargs.pop("batch_size", 32)
        validation_split = kwargs.pop("validation_split", 0.0)
        total_epochs = kwargs.pop("epochs", 200)

        # Mirrors Keras's own validation_split semantics: an unshuffled trailing
        # slice, taken before any shuffling of the training portion.
        n_val = int(round(len(X_scaled) * validation_split))
        if n_val > 0:
            X_train_arr, X_val_arr = X_scaled[:-n_val], X_scaled[-n_val:]
            y_train_arr, y_val_arr = y_scaled[:-n_val], y_scaled[-n_val:]
        else:
            X_train_arr, y_train_arr = X_scaled, y_scaled
            X_val_arr = y_val_arr = None

        # Building the Dataset under tf.device("/cpu:0") keeps the full arrays
        # resident in host RAM; only per-batch slices are staged onto the GPU
        # during iteration. Passing the raw numpy arrays straight to
        # model.fit(X, y, ...) instead lets Keras eagerly place the whole
        # array as a GPU constant, which OOMs once the dataset no longer fits
        # in GPU VRAM (observed with EE_unlensed at ntrain=900k on a 22GB GPU).
        with tf.device("/cpu:0"):
            train_ds = (
                tf.data.Dataset.from_tensor_slices((X_train_arr, y_train_arr))
                .shuffle(len(X_train_arr), reshuffle_each_iteration=True)
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )
            val_ds = None
            if X_val_arr is not None:
                val_ds = (
                    tf.data.Dataset.from_tensor_slices((X_val_arr, y_val_arr))
                    .batch(batch_size)
                    .prefetch(tf.data.AUTOTUNE)
                )

        early_stopping = EarlyStopping(
            monitor="val_loss" if val_ds is not None else "loss",
            patience=100, verbose=1, restore_best_weights=True,
        )

        for lr in learning_rates:
            print(f"\nTraining with learning rate: {lr}")

            # Compile the model with the current learning rate
            custom_loss = CustomLoss(element_weights=self.rescaling_factor)
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=custom_loss
            )

            # Wrap the training in a tqdm progress bar
            with tqdm(total=total_epochs, unit="epoch", postfix={}) as pbar:
                # Custom callback to update tqdm progress bar after each epoch and display validation loss (in scientific notation)
                def on_epoch_end(epoch, logs):
                    train_loss = logs.get(
                        "loss", 0.0
                    )  # Get the training loss from logs
                    val_loss = logs.get(
                        "val_loss", 0.0
                    )  # Get the validation loss from logs
                    pbar.set_postfix(
                        {
                            "train_loss": f"{train_loss:.2e}",
                            "val_loss": f"{val_loss:.2e}",
                        }
                    )  # Show losses in scientific notation
                    self.train_losses.append(train_loss)
                    self.val_losses.append(val_loss)
                    pbar.update(1)

                # Train the model with early stopping and tqdm callback
                self.model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=total_epochs,
                    callbacks=[
                        early_stopping,
                        LambdaCallback(on_epoch_end=on_epoch_end),
                    ],
                    verbose=0,  # Turn off default verbose
                    *args,
                    **kwargs,
                )

            # Optionally save the model at the end of the training cycle
            if self.temp_file is not None:
                self.save(self.temp_file)

    def save(self, filename):
        """Save IntegratedModel to disk.

        Saves the Keras model as an HDF5 file and other attributes (scalers,
        PCA, etc.) as a pickle file.

        Args:
            filename (str): Base filename (without suffix) where model will
                be saved. Files will be saved as:
                - {filename}_model.h5 (Keras model)
                - {filename}.pkl (other attributes)
        """
        # Save the Keras model in the .h5 format to handle custom objects like CustomActivation
        self.model.save(filename + "_model.h5")

        # attributes, excluding the model which we've saved separately
        attributes = [
            self.input_scaler,
            self.output_scaler,
            self.offset,
            self.log_preprocess,
            self.pca,
            self.pca_scaler,
            self.parameter_names,
            self.rescaling_factor,
            self.rescale_by_as,
            self.cheby_preprocess,
            self.cheby_meta,
            self.galaxy_bias_config,
        ]

        # Save the remaining attributes to file
        with open(filename + ".pkl", "wb") as f:
            pickle.dump(attributes, f)

        print(f"Model saved to {filename}_model.h5")

    def restore(self, filename):
        """Load IntegratedModel from disk.

        Loads the Keras model from HDF5 file and other attributes from
        pickle file. Supports backward compatibility with older model formats.

        Args:
            filename (str): Base filename (without suffix) where model was
                saved. Files should be:
                - {filename}_model.h5 (Keras model)
                - {filename}.pkl (other attributes)
        """
        # Load the Keras model from the .h5 format
        self.model = load_model(
            filename + "_model.h5",
            custom_objects={
                "CustomActivation": CustomActivation,
                "CustomLoss": CustomLoss,
            },
            compile=False,
        )

        # Load the remaining attributes
        with open(filename + ".pkl", "rb") as f:
            attributes = pickle.load(f)

            # Default values for backward compatibility
            self.cheby_preprocess = False
            self.cheby_meta = None
            self.rescale_by_as = False
            self.galaxy_bias_config = None

            try:
                # Newest format with galaxy bias config
                (
                    self.input_scaler,
                    self.output_scaler,
                    self.offset,
                    self.log_preprocess,
                    self.pca,
                    self.pca_scaler,
                    self.parameter_names,
                    self.rescaling_factor,
                    self.rescale_by_as,
                    self.cheby_preprocess,
                    self.cheby_meta,
                    self.galaxy_bias_config,
                ) = attributes
            except ValueError:
                try:
                    # Format with Chebyshev support but no galaxy bias config
                    (
                        self.input_scaler,
                        self.output_scaler,
                        self.offset,
                        self.log_preprocess,
                        self.pca,
                        self.pca_scaler,
                        self.parameter_names,
                        self.rescaling_factor,
                        self.rescale_by_as,
                        self.cheby_preprocess,
                        self.cheby_meta,
                    ) = attributes
                    self.galaxy_bias_config = None
                except ValueError:
                    try:
                        # Format with A_s rescaling but no cheby
                        (
                            self.input_scaler,
                            self.output_scaler,
                            self.offset,
                            self.log_preprocess,
                            self.pca,
                            self.pca_scaler,
                            self.parameter_names,
                            self.rescaling_factor,
                            self.rescale_by_as,
                        ) = attributes
                        self.galaxy_bias_config = None
                    except ValueError:
                        try:
                            # For backward compatibility with older models without A_s rescaling
                            (
                                self.input_scaler,
                                self.output_scaler,
                                self.offset,
                                self.log_preprocess,
                                self.pca,
                                self.pca_scaler,
                                self.parameter_names,
                                self.rescaling_factor,
                            ) = attributes
                            self.rescale_by_as = False
                            self.galaxy_bias_config = None
                        except:
                            print("Very old model format detected")
                            (
                                self.input_scaler,
                                self.output_scaler,
                                self.offset,
                                self.log_preprocess,
                                self.pca,
                                self.parameter_names,
                                self.rescaling_factor,
                            ) = attributes
                            self.rescale_by_as = False
                            self.galaxy_bias_config = None

        self.scaler_mean_in = self.input_scaler.mean_  # Mean of the features
        self.scaler_scale_in = (
            self.input_scaler.scale_
        )  # Standard deviation of the features

        self.scaler_scale_out = (
            self.output_scaler.scale_
        )  # Standard deviation of the features
        self.scaler_mean_out = (
            self.output_scaler.mean_
        )  # Standard deviation of the features

        print("restore successful")
