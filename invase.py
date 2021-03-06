"""
Tensorflow implementation of INVASE method from Yoon et al. (2019)
https://openreview.net/forum?id=BJg_roAcK7
"""
import tensorflow as tf

class Invase(object):

    def __init__(self, predictor_model, selector_model, error_fn, norm_coeff=1.):
        """Initialise

        Args:
            predictor_model: keras model that the predictor and baseline are constructed from
            selector_model: keras model that the selector is constructed from
            error_fn: a function that takes prediction and truth as argument and returns model error
            norm_coeff: regularisation coefficient from INVASE paper
        """
        self.predictor = predictor_model()
        self.baseline = predictor_model()
        self.selector = selector_model()
        self.error_fn = error_fn
        self.norm_coeff = norm_coeff

        self.predictor_train_loss = tf.keras.metrics.Mean(name='predictor_train_loss')
        self.baseline_train_loss = tf.keras.metrics.Mean(name='baseline_train_loss')
        self.selector_train_loss = tf.keras.metrics.Mean(name='selector_train_loss')

        self.predictor_test_loss = tf.keras.metrics.Mean(name='predictor_test_loss')
        self.baseline_test_loss = tf.keras.metrics.Mean(name='baseline_test_loss')

    def _selector_dropout_vector(self, x, rates=0.5):
        """Returns random binary vector with the same size as x"""
        uniform_random = tf.random.uniform(shape=tf.shape(x), dtype=x.dtype)
        mask = (uniform_random < tf.cast(rates, dtype=x.dtype))
        
        return tf.cast(mask, dtype=x.dtype)

    def _selector_loss(self, predictor_error, baseline_error, selection_vectors, selection_probs):
        """Calculates the loss of the selector model"""
        # Obtain log(pi(s))
        sel_log_probs = tf.math.reduce_sum(selection_vectors * tf.math.log(selection_probs) +
            (1 - selection_vectors) * tf.math.log(1 - selection_probs), axis=1)
        # Obtain ||s||
        sel_norm = tf.math.reduce_sum(selection_vectors, axis=1)

        error = predictor_error - baseline_error
        # Loss function from the original paper:
        # sel_error = (error + self._norm_coeff * sel_norm) * sel_log_probs
        # Loss function used in published code:
        sel_error = error * sel_log_probs + self.norm_coeff * sel_norm
        sel_loss = tf.math.reduce_mean(sel_error, axis=0)

        return sel_loss

    @tf.function
    def train_step(self, features, labels, optimizer):
        """Performs INVERSE train step

        Args:
            features: tensor containing batch features
            labels: tensor containing ground truth response of batch
            optimizer: keras optimizer

        Returns:
            Predictor model predictions on features for custom evaluation
        """
        with tf.GradientTape() as baseline_tape:
            # Get train loss for baseline
            baseline_predictions = self.baseline(features, training=True)
            baseline_error = self.error_fn(labels, baseline_predictions)
            baseline_loss = tf.math.reduce_mean(baseline_error)

        with tf.GradientTape() as selector_tape:
            # Get selective dropout vector
            selection_probs = self.selector(features)
            selection_vectors = self._selector_dropout_vector(features, selection_probs)

            # Apply dropout to obtain feature selection
            selected_features = features * selection_vectors

            with tf.GradientTape() as predictor_tape:
                # Obtain train loss for predictor
                predictor_predictions = self.predictor(selected_features, training=True)
                predictor_error = self.error_fn(labels, predictor_predictions)
                predictor_loss = tf.math.reduce_mean(predictor_error)
            
            # Calculate selector error
            selector_loss = self._selector_loss(predictor_error, baseline_error, selection_vectors, selection_probs)

        # Update gradients for predictor and selector
        predictor_gradients = predictor_tape.gradient(predictor_loss, self.predictor.trainable_variables)
        baseline_gradients = baseline_tape.gradient(baseline_loss, self.baseline.trainable_variables)
        selector_gradients = selector_tape.gradient(selector_loss, self.selector.trainable_variables)
        gradients = predictor_gradients + baseline_gradients + selector_gradients
        variables = self.predictor.trainable_variables + self.baseline.trainable_variables + self.selector.trainable_variables
        optimizer.apply_gradients(zip(gradients, variables))

        self.predictor_train_loss(predictor_loss)
        self.baseline_train_loss(baseline_loss)
        self.selector_train_loss(selector_loss)

        return predictor_predictions

    @tf.function
    def test_step(self, features, labels):
        """Performs INVERSE test step

        Args:
            features: tensor containing batch features
            labels: tensor containing ground truth response of batch

        Returns:
            Predictor model predictions on features for custom evaluation
        """
        predictor_predictions = self.predictor(features, training=False)
        predictor_loss = self.error_fn(labels, predictor_predictions)

        baseline_predictions = self.baseline(features, training=False)
        baseline_loss = self.error_fn(labels, baseline_predictions)

        # Update metrics
        self.baseline_test_loss(predictor_loss)
        self.predictor_test_loss(predictor_loss)

        return predictor_predictions

    def reset_metrics(self):
        """Resets train and test metrics"""
        # Reset train metrics
        self.predictor_train_loss.reset_states()
        self.baseline_train_loss.reset_states()
        self.selector_train_loss.reset_states()

        # Reset test metrics
        self.predictor_test_loss.reset_states()