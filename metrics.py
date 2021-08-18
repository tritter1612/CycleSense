
import tensorflow as tf

class Specificity(tf.keras.metrics.Metric):

    def __init__(self, name='specificity', **kwargs):
        super(Specificity, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        # true positives
        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)
        self.tp.assign_add(tf.reduce_sum(values))

        # false negatives
        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)
        self.fn.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.tp / (self.tp + self.fn)


class Sensitivity(tf.keras.metrics.Metric):

    def __init__(self, name='sensitivity', **kwargs):
        super(Sensitivity, self).__init__(name=name, **kwargs)
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.tn = self.add_weight(name='tn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        # false positives
        values = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)
        self.fp.assign_add(tf.reduce_sum(values))

        # true negatives
        values = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, False))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)
        self.tn.assign_add(tf.reduce_sum(values))


    def result(self):
        return self.tn / (self.tn + self.fp)


class TSS(tf.keras.metrics.Metric):

    def __init__(self, name='tss', **kwargs):
        super(TSS, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.tn = self.add_weight(name='tn', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        # true positives
        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)
        self.tp.assign_add(tf.reduce_sum(values))

        # false positives
        values = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)
        self.fp.assign_add(tf.reduce_sum(values))

        # true negatives
        values = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, False))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)
        self.tn.assign_add(tf.reduce_sum(values))

        # false negatives
        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)
        self.fn.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.tp / (self.tp + self.fn) - self.fp/(self.fp + self.tn)