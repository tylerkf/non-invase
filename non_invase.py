"""
Tensorflow implementation of NON-INVASE method
"""
import tensorflow as tf

class NonInvase(object):

    def __init__(self, predictor_model, selector_model, error_fn, prior_coeff=0.01):
        """Initialise

        Args:
            predictor_model: keras model that the predictor and baseline are constructed from
            selector_model: keras model that the selector is constructed from
            error_fn: a function that takes prediction and truth as argument and returns model error
            prior_coeff: regularisation coefficient for prior divergence
        """
        self.predictor = predictor_model()
        self.selector = selector_model()
        self.error_fn = error_fn
        self.prior_coeff = prior_coeff

        self.predictor_train_loss = tf.keras.metrics.Mean(name='predictor_train_loss')
        self.selector_train_loss = tf.keras.metrics.Mean(name='selector_train_loss')

        self.predictor_test_loss = tf.keras.metrics.Mean(name='predictor_test_loss')

    def _selector_dropout_vector(self, x, rates=0.5):
        """Returns random binary vector with the same size as x"""
        uniform_random = tf.random.uniform(shape=tf.shape(x), dtype=x.dtype)
        mask = (uniform_random < tf.cast(rates, dtype=x.dtype))
        
        return tf.cast(mask, dtype=x.dtype)

    def _selector_loss(self, error, selection_probs, selection_vectors):
        """Calculates the loss of the selector model"""
        entropy = tf.math.reduce_sum(
            selection_probs * tf.math.log(selection_probs) + (1 - selection_probs) * tf.math.log(1 - selection_probs), axis=1)

        selection_log_probs = tf.math.reduce_sum(selection_vectors * tf.math.log(selection_probs) +
                (1 - selection_vectors) * tf.math.log(1 - selection_probs), axis=1)
        weighted_error =  selection_log_probs * error

        loss = tf.math.reduce_mean(weighted_error) + self.prior_coeff * tf.math.reduce_mean(entropy)
        return loss

    def _param_reg(self, selection_probs, first_weights):
        """Calculates the regularization term"""
        rank = len(selection_probs.shape)
        weighted_l2 = tf.math.reduce_sum(
                tf.tensordot(selection_probs, tf.math.square(first_weights), [[rank - 1], [0]]), axis=1)
        l2_reg = self.prior_coeff * 0.5 * tf.math.reduce_mean(weighted_l2, axis=0)

        return l2_reg

    @tf.function
    def train_step(self, features, labels, optimizer):
        """Performs NON-INVERSE train step

        Args:
            features: tensor containing batch features
            labels: tensor containing ground truth response of batch
            optimizer: keras optimizer

        Returns:
            Predictor model predictions on features for custom evaluation
        """
        with tf.GradientTape() as selector_tape:
            # Get selective dropout vector
            selection_probs = self.selector(features)
            selection_vectors = self._selector_dropout_vector(features, selection_probs)

            # Apply dropout to obtain feature selection
            selected_features = features * selection_vectors

            with tf.GradientTape() as predictor_tape:
                # Obtain train error for predictor
                predictor_predictions = self.predictor(selected_features, training=True)
                predictor_error = self.error_fn(labels, predictor_predictions)

                # Calcualte regularization term
                first_weights = self.predictor.d1.kernel
                l2_reg = self._param_reg(selection_probs, first_weights)

                # Calculate predictor loss
                predictor_loss = tf.math.reduce_mean(predictor_error)
                predictor_loss_reg = predictor_loss + l2_reg
            
            # Calculate selector error
            selector_loss = self._selector_loss(predictor_error, selection_probs, selection_vectors) + l2_reg

        # Update gradients for predictor and selector
        predictor_gradients = predictor_tape.gradient(predictor_loss_reg, self.predictor.trainable_variables)
        selector_gradients = selector_tape.gradient(selector_loss, self.selector.trainable_variables)
        gradients = predictor_gradients + selector_gradients
        variables = self.predictor.trainable_variables + self.selector.trainable_variables
        optimizer.apply_gradients(zip(gradients, variables))

        # Update metrics
        self.predictor_train_loss(predictor_loss)
        self.selector_train_loss(selector_loss)

        return predictor_predictions

    @tf.function
    def test_step(self, features, labels):
        """Performs NON-INVERSE test step

        Args:
            features: tensor containing batch features
            labels: tensor containing ground truth response of batch

        Returns:
            Predictor model predictions on features for custom evaluation
        """
        predictor_predictions = self.predictor(features, training=False)
        predictor_loss = self.error_fn(labels, predictor_predictions)

        # Update metrics
        self.predictor_test_loss(predictor_loss)

        return predictor_predictions

    def reset_metrics(self):
        """Resets train and test metrics"""
        # Reset train metrics
        self.predictor_train_loss.reset_states()
        self.selector_train_loss.reset_states()

        # Reset test metrics
        self.predictor_test_loss.reset_states()