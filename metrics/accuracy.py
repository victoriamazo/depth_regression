import numpy as np
from metrics.metric_builder import Metric


class accuracy_onehot(Metric):
    def __init__(self, x, y, FLAGS):
        super(accuracy_onehot, self).__init__(x, y, FLAGS)
        self.logits = x
        self.y = y


    def build(self):
        correct_pred = np.sum(np.equal(np.argmax(self.logits, 1), np.argmax(self.y, 1)))
        acc = correct_pred * 100.0 / len(self.y)

        return acc




class accuracy_categ(Metric):
    def __init__(self, x, y, FLAGS):
        super(accuracy_categ, self).__init__(x, y, FLAGS)
        self.logits = x
        self.y = y


    def build(self):
        correct_pred = np.sum(np.equal(np.argmax(self.logits, 1), self.y))
        acc = correct_pred * 100.0 / len(self.y)
        return acc