def mse(targets, preds):
    n = len(targets)
    return sum([(ti - pi)**2 for ti, pi in zip(targets, preds)]) / n

def dmse(targets, preds):
    n = len(targets)
    return [2 * (pi - ti) / n for ti, pi in zip(targets, preds)]
