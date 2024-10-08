class MetricFunctions:
    @staticmethod
    def recall(tp, fn):
        return tp / (tp + fn)

    @staticmethod
    def precision(tp, fp):
        return tp / (tp + fp)

    @staticmethod
    def accuracy(tp, fp, fn, tn):
        return (tp + tn) / (tp + tn + fn + fp)
