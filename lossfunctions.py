import numpy as np

def mse(targets, preds):
    n = len(targets)
    return sum([(ti - pi)**2 for ti, pi in zip(targets, preds)]) / n

def dmse(targets, preds):
    n = len(targets)
    return [2 * (pi - ti) / n for ti, pi in zip(targets, preds)]

class CrossEntropyLoss():
    def loss(self, logits, targs: '[batch_size][num_classes]'):
        """
        logits: shape (batch_size, num_classes)
        targets: shape (batch_size,) integer class indices
        returns: float
        """
        logits = np.array(logits)
        targs = np.array(targs)

        batch_range = range(targs.size)

        exp_logits_sum = np.sum(np.exp(logits), axis = 1) 
        exp_correct_logits = np.exp([logits[i][targs[i]] for i in batch_range])

        ln = -np.log(exp_correct_logits/exp_logits_sum)

        out = np.sum(ln)

        return out.tolist()
    
    def theta(self, logits, targets):
        """
        logits: shape (batch_size, num_classes)
        targets: shape (batch_size,) integer class indices
        returns: shape (batch_size, num_classes)
        """
        logits = np.array(logits)
        targs = np.array(targs)

        # Compute softmax
        exp_logits = np.exp(logits)
        softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Create one-hot targets
        one_hot = np.zeros_like(logits)
        one_hot[np.arange(len(targets)), targets] = 1

        # Derivative
        return (softmax - one_hot) / len(targets)