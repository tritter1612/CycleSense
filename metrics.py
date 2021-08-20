import numpy as np
import tensorflow as tf
from keras import backend
from keras.utils import metrics_utils
from keras.utils.generic_utils import to_list


class TSS(tf.keras.metrics.Metric):

    def __init__(self,
                 thresholds=None,
                 top_k=None,
                 class_id=None,
                 name='tss',
                 dtype=None):
        super(TSS, self).__init__(name=name, dtype=dtype)
        self.init_thresholds = thresholds
        self.top_k = top_k
        self.class_id = class_id

        default_threshold = 0.5 if top_k is None else tf.keras.metrics_utils.NEG_INF
        self.thresholds = metrics_utils.parse_init_thresholds(
            thresholds, default_threshold=default_threshold)
        self.true_positives = self.add_weight(
            'true_positives',
            shape=(len(self.thresholds),),
            initializer=tf.zeros_initializer())
        self.false_positives = self.add_weight(
            'false_positives',
            shape=(len(self.thresholds),),
            initializer=tf.zeros_initializer())
        self.true_negatives = self.add_weight(
            'true_negatives',
            shape=(len(self.thresholds),),
            initializer=tf.zeros_initializer())
        self.false_negatives = self.add_weight(
            'false_negatives',
            shape=(len(self.thresholds),),
            initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred, sample_weight=None):
        '''Accumulates true and false positive and true and false negative statistics.
        Args:
          y_true: The ground truth values, with the same dimensions as `y_pred`.
            Will be cast to `bool`.
          y_pred: The predicted values. Each element must be in the range `[0, 1]`.
          sample_weight: Optional weighting of each example. Defaults to 1. Can be a
            `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
            be broadcastable to `y_true`.
        Returns:
          Update op.
        '''
        return metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
                metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives,
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives
            },
            y_true,
            y_pred,
            thresholds=self.thresholds,
            top_k=self.top_k,
            class_id=self.class_id,
            sample_weight=sample_weight)

    def result(self):
        result = tf.math.subtract(
            tf.math.divide_no_nan(self.true_positives,
                                  tf.math.add(self.true_positives, self.false_negatives)),
            tf.math.divide_no_nan(self.false_positives,
                                  tf.math.add(self.false_positives, self.true_negatives)))
        return result[0] if len(self.thresholds) == 1 else result

    def reset_state(self):
        num_thresholds = len(to_list(self.thresholds))
        backend.batch_set_value([(v, np.zeros((num_thresholds,)))
                                          for v in (self.true_positives,
                                                    self.false_positives,
                                                    self.true_negatives,
                                                    self.false_negatives)])

    def get_config(self):
        config = {
            'thresholds': self.init_thresholds,
            'top_k': self.top_k,
            'class_id': self.class_id
        }
        base_config = super(TSS, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
