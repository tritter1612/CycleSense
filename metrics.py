import tensorflow as tf


class ConfusionMatrixMetric(tf.keras.metrics.Metric):

    def __init__(self, num_classes, **kwargs):
        super(ConfusionMatrixMetric, self).__init__(name='confusion_matrix_metric',
                                                    **kwargs)  # handles base args (e.g., dtype)
        self.num_classes = num_classes
        self.total_cm = self.add_weight("total", shape=(num_classes, num_classes), initializer="zeros")

    def reset_states(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.total_cm.assign_add(self.confusion_matrix(y_true, y_pred))
        return self.total_cm

    def result(self):
        return self.process_confusion_matrix()

    def confusion_matrix(self, y_true, y_pred):

        if self.num_classes <= 2:
            cm = tf.math.confusion_matrix(tf.squeeze(y_true, 1), tf.squeeze(y_pred, 1), dtype=tf.float32, num_classes=self.num_classes)
        else:
            y_pred = tf.argmax(y_pred, 1)
            cm = tf.math.confusion_matrix(y_true, y_pred, dtype=tf.float32, num_classes=self.num_classes)

        return cm

    def process_confusion_matrix(self):
        cm = self.total_cm
        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tp = cm[1, 1]
        return tn, fp, fn, tp

    def fill_output(self, output):
        results = self.result()
        output['tn'] = results[0]
        output['fp'] = results[1]
        output['fn'] = results[2]
        output['tp'] = results[3]
