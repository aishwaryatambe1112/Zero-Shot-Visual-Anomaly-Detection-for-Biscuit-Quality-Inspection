from collections import defaultdict

class Metrics:
    def __init__(self, classes):
        self.classes = classes
        self.matrix = defaultdict(lambda: defaultdict(int))

    def update(self, gt, pred):
        self.matrix[gt][pred] += 1

    def precision(self, cls):
        tp = self.matrix[cls][cls]
        fp = sum(self.matrix[c][cls] for c in self.classes if c != cls)
        return tp / (tp + fp + 1e-6)

    def recall(self, cls):
        tp = self.matrix[cls][cls]
        fn = sum(self.matrix[cls][c] for c in self.classes if c != cls)
        return tp / (tp + fn + 1e-6)

    def f1(self, cls):
        p = self.precision(cls)
        r = self.recall(cls)
        return 2 * p * r / (p + r + 1e-6)

