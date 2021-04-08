# import numpy as np
# import tensorflow as tf
# from ares.attack.base import BatchAttack
# from ares.attack.utils import get_xs_ph, get_ys_ph
# from ares.loss import CrossEntropyLoss


# class Attacker(BatchAttack):
#     def __init__(self, model, batch_size, dataset, session):
#         """ Based on ares.attack.bim.BIM, numpy version. """
#         self.model, self.batch_size, self._session = model, batch_size, session
#         # dataset == "imagenet" or "cifar10"
#         loss = CrossEntropyLoss(self.model)
#         # placeholder for batch_attack's input
#         self.xs_ph = get_xs_ph(model, batch_size)
#         labels_op = model.labels(self.xs_ph)
#         self.logits, _ = model.logits_and_labels(self.xs_ph)
#         self.ys_ph = get_ys_ph(model, batch_size)
#         self.loss = loss(self.xs_ph, self.ys_ph)
#         self.grad = tf.gradients(self.loss, self.xs_ph)[0]
#         self.logits_softmax = tf.nn.softmax(self.logits)
#         self.iteration = 5

#     def config(self, **kwargs):
#         if 'magnitude' in kwargs:
#             self.eps = kwargs['magnitude'] - 1e-8
#             self.alpha = self.eps / 6

#     def batch_attack(self, xs, ys=None, ys_target=None):
#         xs_lo, xs_hi = xs - self.eps, xs + self.eps
#         a = np.shape(xs)
#         np.random.seed(2021327)
#         xs_adv = xs + 7*self.alpha*np.random.random_sample(a)
#         # xs_adv = xs
#         # grad = self._session.run(self.grad, feed_dict={self.xs_ph: xs_adv, self.ys_ph: ys})
#         # grad = grad.reshape(self.batch_size, *self.model.x_shape)
#         # grad_sign = np.sign(grad)
#         # xs_adv = np.clip(xs_adv + 6*self.alpha * grad_sign, xs_lo, xs_hi)
#         xs_adv = np.clip(xs_adv, self.model.x_min, self.model.x_max)
        
#         # logits_list = []
#         # label_list = []

#         for i in range(self.iteration): 
             
#             grad = self._session.run(self.grad, feed_dict={self.xs_ph: xs_adv, self.ys_ph: ys})
#             # logits = self._session.run(self.logits, feed_dict={self.xs_ph: xs_adv, self.ys_ph: ys})
#             # logits_list.append(logits)
#             # label_list.append(ys)
#             grad = grad.reshape(self.batch_size, *self.model.x_shape)
#             grad_sign = np.sign(grad)
#             # losslist = self._session.run(self.loss, feed_dict={self.xs_ph: xs_adv, self.ys_ph: ys})
#             logits_list = self._session.run(self.logits, feed_dict={self.xs_ph: xs_adv, self.ys_ph: ys})
#             logits_softmax_data = self._session.run(self.logits_softmax, feed_dict={self.xs_ph: xs_adv, self.ys_ph: ys})
            
#             print(type(logits_softmax_data))
#             # print('................................')
#             # print(logits_list)
#             # print('********************************')
#             # print(logits_list)
#             # if i< 0.4*self.iteration:
#             #     xs_adv = xs_adv + self.alpha * grad_sign
#             # else:
#             #     xs_adv = np.clip(xs_adv + self.alpha * grad_sign, xs_lo, xs_hi)
#             #     xs_adv = np.clip(xs_adv, self.model.x_min, self.model.x_max)
#             xs_adv = np.clip(xs_adv + self.alpha * grad_sign, xs_lo, xs_hi)
#             xs_adv = np.clip(xs_adv, self.model.x_min, self.model.x_max)
#         # return xs_adv,logits_list,label_list
#         return xs_adv


import numpy as np
import copy
import tensorflow as tf
from ares.attack.base import BatchAttack
from ares.attack.utils import get_xs_ph, get_ys_ph
from ares.loss import CrossEntropyLoss


class Attacker(BatchAttack):
    def __init__(self, model, batch_size, dataset, session):
        """ Based on ares.attack.bim.BIM, numpy version. """
        self.model, self.batch_size, self._session = model, batch_size, session
        # dataset == "imagenet" or "cifar10"
        loss = CrossEntropyLoss(self.model)
        # placeholder for batch_attack's input
        self.xs_ph = get_xs_ph(model, batch_size)
        self.logits, self.labels = model.logits_and_labels(self.xs_ph)
        self.ys_ph = get_ys_ph(model, batch_size)
        self.loss = loss(self.xs_ph, self.ys_ph)
        self.grad = tf.gradients(self.loss, self.xs_ph)[0]
        self.logits_softmax = tf.nn.softmax(self.logits)
        self.iteration = 50

    def config(self, **kwargs):
        if 'magnitude' in kwargs:
            self.eps = kwargs['magnitude'] - 1e-8
            self.alpha = self.eps / 6

    def batch_attack(self, xs, ys=None, ys_target=None):
        xs_lo, xs_hi = xs - self.eps, xs + self.eps
        a = np.shape(xs)
        # xs_adv = xs + 7/255*np.random.random_sample(a)
        xs_adv = xs
        np.random.seed(2021327)
        xs_adv = xs + 4*self.alpha*np.random.random_sample(a)

        #grad = self._session.run(self.grad, feed_dict={self.xs_ph: xs_adv, self.ys_ph: ys})
        #grad = grad.reshape(self.batch_size, *self.model.x_shape)
        #grad_sign = np.sign(grad)
        xs_adv = np.clip(xs_adv, xs_lo, xs_hi)
        xs_adv = np.clip(xs_adv, self.model.x_min, self.model.x_max)

        #r = 0
        #rho = 0.2
        #sigma = 0.0000001
        np.zeros(50)
        pr_copy = []
        for i in range(self.iteration):            
            #logits,_ = self._session.run((self.logits, self.labels), feed_dict={self.xs_ph: xs, self.ys_ph: ys})            
            logits_softmax_data = self._session.run(self.logits_softmax, feed_dict={self.xs_ph: xs_adv, self.ys_ph: ys})
            grad = self._session.run(self.grad, feed_dict={self.xs_ph: xs_adv, self.ys_ph: ys})
            grad = grad.reshape(self.batch_size, *self.model.x_shape)
            grad_sign = np.sign(grad)
            pr = [(max(p)) for p in logits_softmax_data.tolist()]            
            b = [p.index(max(p)) for p in logits_softmax_data.tolist()]
            for j in range(len(b)):
                if b[j] == ys[j]:
                    if len(pr_copy) is 0:
                        weight = 1
                    else:
                        
                        weight = (abs(pr[j] - pr_copy[j]) + 0.001) / pr[j]
                        #print(np.clip((0.01 / weight), 1, 60))

                    xs_adv[j] = np.clip(xs_adv[j] + np.clip((0.05 / weight), 1, 60) * self.alpha * grad_sign[j], xs_lo[j], xs_hi[j])
                    xs_adv[j] = np.clip(xs_adv[j], self.model.x_min, self.model.x_max)
                #else:
                    #print(j)
            pr_copy = pr.copy()
            #print('pr_copy',pr_copy)
            #print('**********************')
            #print('pr',pr)




            

            #r += rho * r + (1-rho) * grad * grad
            #delta_w = ((sigma + r) ** 0.5 / self.alpha) * grad
            #if i< 0.1*self.iteration:
            #    xs_adv = xs_adv + delta_w
                #xs_adv = xs_adv + self.alpha * grad_sign
                
            #else:
            #xs_adv = np.clip(xs_adv + delta_w, xs_lo, xs_hi)
            #xs_adv = np.clip(xs_adv, self.model.x_min, self.model.x_max)
            
            
            #with tf.Session() as sess:  
            #  print(self.loss.eval())
            # configure = tf.ConfigProto()
            # configure.gpu_options.allow_growth = True
            # mysession = tf.Session(config=configure)
            #myLoss = self.loss.eval(session=self._session)
            #print(myLoss)
            # print(type(self.loss))
            # .eval(session=your_session)
            #print(self.loss.weight.data.cpu().numpy())
            
            #print(np.shape(logits))
            
            #num_nonzero = np.count_nonzero(np.array(b) - np.array(ys))
            #if num_nonzero > 20:
            #  print(num_nonzero, i)
        return xs_adv
