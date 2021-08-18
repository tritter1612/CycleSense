import tensorflow as tf


class TSS(tf.keras.metrics.Metric):

    def __init__(self, **kwargs):
        super(TSS, self).__init__(name='tss', **kwargs)
        self.total_cm = self.add_weight('total_cm_tss', shape=(2, 2), initializer='zeros')

    def reset_state(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.total_cm.assign_add(self.confusion_matrix(y_true, y_pred))
        return self.total_cm

    def result(self):
        cm = self.total_cm

        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tp = cm[1, 1]

        tss = tp / (tp + fn) - fp / (fp + tn)
        return tss

    def confusion_matrix(self, y_true, y_pred):
        cm = tf.math.confusion_matrix(tf.squeeze(y_true, 1), tf.squeeze(y_pred, 1), dtype=tf.float32, num_classes=2)
        return cm
