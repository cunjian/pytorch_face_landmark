from __future__ import print_function, absolute_import
import numpy as np

__all__ = ['accuracy','normalizedME']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def normalizedME(output,target,w,h):
    batch_size = target.size(0)
    diff = output - target
    diff = np.sqrt(diff.T*diff)/(w*h)
    return diff/batch_size