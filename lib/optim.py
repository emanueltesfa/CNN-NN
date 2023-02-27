from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


""" Super Class """


class Optimizer(object):
    """
    This is a template for implementing the classes of optimizers
    """

    def __init__(self, net, lr=1e-4):
        self.net = net  # the model
        self.lr = lr    # learning rate

    """ Make a step and update all parameters """

    def step(self):
        #### FOR RNN / LSTM ####
        if hasattr(self.net, "preprocess") and self.net.preprocess is not None:
            self.update(self.net.preprocess)
        if hasattr(self.net, "rnn") and self.net.rnn is not None:
            self.update(self.net.rnn)
        if hasattr(self.net, "postprocess") and self.net.postprocess is not None:
            self.update(self.net.postprocess)

        #### MLP ####
        if not hasattr(self.net, "preprocess") and \
           not hasattr(self.net, "rnn") and \
           not hasattr(self.net, "postprocess"):
            for layer in self.net.layers:
                self.update(layer)


""" Classes """


class SGD(Optimizer):
    """ Some comments """

    def __init__(self, net, lr=1e-4, weight_decay=0.0):
        self.net = net
        self.lr = lr
        self.weight_decay = weight_decay

    def update(self, layer):
        for n, dv in layer.grads.items():
            #############################################################################
            # TODO: Implement the SGD with (optional) Weight Decay                      #
            #############################################################################
            #print("SGD: ", layer.grads.items())

            temp = 0.0
            if self.weight_decay != 0:
                temp += self.weight_decay * layer.params[n]
            layer.params[n] -= (self.lr * dv) + temp
            #############################################################################
            #                             END OF YOUR CODE                              #
            #############################################################################


class Adam(Optimizer):
    """ Some comments """

    def __init__(self, net, lr=1e-3, beta1=0.9, beta2=0.999, t=0, eps=1e-8, weight_decay=0.0):
        self.net = net
        self.lr = lr
        self.beta1, self.beta2 = beta1, beta2
        self.eps = eps
        self.mt = {}
        self.vt = {}
        self.t = t
        self.weight_decay = weight_decay

    def update(self, layer):
        #############################################################################
        # TODO: Implement the Adam with [optinal] Weight Decay                      #
        #############################################################################
        #print("beta: ", self.beta1, "eps: ",  self.eps, "mt: ", self.mt['adam_fc_w'].shape, "vt: ", self.vt['adam_fc_w'].shape , "t: ", self.t)
        self.t += 1 
        #print(layer.grads['adam_fc_w']) # is this var the same as theta = 
        
        # 
        # if self.mt == None:
        #     self.mt = 0

        for n, dv in layer.grads.items():
            # print("layrer parmams: ", layer.grads[n])
            # print(layer.grads['adam_fc_w'])
            #print("self.mt[n]: ", self.mt[n])
            if n not in self.mt.keys(): 
                self.mt[n] = 0
            if n not in self.vt.keys():
                self.vt[n] = 0 
            self.mt[n] = (self.mt[n] * self.beta1) + (1 - self.beta1) * dv
            self.vt[n] = (self.vt[n] * self.beta2) + (1 - self.beta2) * np.power(dv, 2)
            m_hat = self.mt[n] / ( 1 - np.power(self.beta1, self.t) ) 
            v_hat = self.vt[n] / ( 1 - np.power(self.beta2, self.t) )

            layer.params[n] -= ((self.lr * m_hat) / (np.sqrt(v_hat) + self.eps)) 
            layer.params[n] -= (self.weight_decay * layer.params[n])
            #     top_term = (self.lr) * (1 - self.beta1) * layer.grads['adam_fc_w']
            # #print(top_term)
            #     bottom_term = (self.eps + (1-self.beta2) * layer.grads['adam_fc_w']) / (1 - np.power(self.beta2, self.t))
            #     layer.grads[n] -= ( top_term / (1 - np.power(self.beta2, self.t) )) / bottom_term
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
