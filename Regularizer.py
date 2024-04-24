import numpy as np
import math
from abc import ABC, abstractmethod

class Regulizer(ABC):
    def __init__(self,alpha):
        self.alpha=alpha
    @abstractmethod
    def __call__(self, w):
        pass
    @abstractmethod
    def _grad(self,w):
        pass

class No_Regulizer(Regulizer):
    def __init__(self,alpha=0):
        super(No_Regulizer, self).__init__(alpha=0)
    def __call__(self, w):
        return 0
    def _grad(self,w):
        return 0


class Ridge(Regulizer):
    def __init__(self,alpha):
        super(Ridge, self).__init__(alpha=alpha)

    def __call__(self, w):
        return self.alpha * 0.5 *  w.T.dot(w)

    def _grad(self,w):
        return self.alpha * w


class Lasso(Regulizer):
    def __init__(self,alpha):
        super(Lasso, self).__init__(alpha=alpha)

    def __call__(self, w):
        return self.alpha * np.linalg.norm(w)
    def _grad(self,w):
        return self.alpha * np.sign(w)


class ElasticNet(Regulizer):
    def __init__(self,alpha,l1_ratio):
        super(ElasticNet, self).__init__(alpha=alpha)
        self.l1_ratio = l1_ratio

    def __call__(self, w):
        l1_contr = self.l1_ratio * np.linalg.norm(w)
        l2_contr = (1 - self.l1_ratio) * 0.5 * w.T.dot(w) 
        return self.alpha * (l1_contr + l2_contr)
        
    def _grad(self,w):
        l1_contr = self.l1_ratio * np.sign(w)
        l2_contr = (1 - self.l1_ratio) * w
        return self.alpha * (l1_contr + l2_contr) 
