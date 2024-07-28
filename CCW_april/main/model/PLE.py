import tensorflow as tf
from tensorflow import keras as k
from mmoe import MLP

class CGC_Layer(k.layers.Layer):
    def __init__(self
                 , expert_layer
                 , expert_layer_drop
                 , n_expert_share
                 , n_expert_specific
                 , n_task = 2
                 , l2 = 0.0
                 ):
        super(CGC_Layer, self).__init__()
        self.share_expert_num = n_expert_share
        self.specific_expert_num = n_expert_specific
        self.n_task = n_task
        self.share_expert_layer = [
            MLP(expert_layer, expert_layer_drop, prefix = "share_expert{}".format(i), l2=l2)
            for i in range(n_expert_share)
        ]
        self.specific_expert_layer = [
            [
                MLP(expert_layer, expert_layer_drop, prefix = "specific_expert{}-{}".format(k, i), l2=l2)
                for i in range(n_expert_specific)
            ]
            for k in range(n_task)
        ]
        self.gate_layer = [
            k.layers.Dense(n_expert_share + n_expert_specific, activation = "softmax")
            for i in range(n_task)
        ]

    def call(self, x):
        share_experts = [expert(x) for expert in self.share_expert_layer]
        towers = []
        for i in range(self.n_task):
            E_net = share_experts + [expert(x) for expert in self.specific_expert_layer[i]]
            E_net = k.layers.Concatenate(axis = 1)([e[:, tf.newaxis, :] for e in E_net])
            gate_net =  self.gate_layer[i](x)
            g = tf.expand_dims(gate_net, axis = -1)
            _tower = tf.matmul(E_net, g, transpose_a = True)
            towers.append(k.layers.Flatten()(_tower))
        return towers

    def get_config(self):
        base_config = super(CGC_Layer, self).get_config()
        config = {
            "n_task": self.n_task,
            "n_expert_share": self.share_expert_num,
            "n_expert_specific": self.specific_expert_num
        }
        return dict( list(config.items()) + list(base_config.items()) )
class PLE_Layer(k.layers.Layer):
    def __init__(self
                 , expert_layer
                 , expert_layer_drop
                 , n_expert_share
                 , n_expert_specific
                 , n_task = 2
                 , l2 = 0.0
                 , n_cgc = 2):
        super(PLE_Layer, self).__init__()
        self.share_expert_num = n_expert_share
        self.specific_expert_num = n_expert_specific
        self.n_task = n_task
        self.n_cgc = n_cgc
        self.share_expert_layer = {}
        self.specific_expert_layer = {}
        self.gate_layer = {}
        for j in range(n_cgc):
            self.share_expert_layer[j] = [
                MLP(expert_layer, expert_layer_drop, prefix = "layer{}_share_expert{}".format(j, i), l2 = l2)
                for i in range(n_expert_share)
            ]
            self.specific_expert_layer[j] = [
                [
                    MLP(expert_layer, expert_layer_drop, prefix = "layer{}_specific_expert{}-{}".format(j, k, i), l2 =l2)
                    for i in range(n_expert_specific)
                ]
                for k in range(n_task)
            ]
            self.gate_layer[j] = [
                k.layers.Dense(n_expert_share + n_expert_specific, activation = "softmax")
                for i in range(n_task)
            ]

    def call(self, x):
        for j in range(self.n_cgc):
            share_experts = [expert(x) for expert in self.share_expert_layer[j]]
            towers = []
            for i in range(self.n_task):
                E_net = share_experts + [expert(x) for expert in self.specific_expert_layer[j][i]]
                E_net = k.layers.Concatenate(axis = 1)([e[:, tf.newaxis, :] for e in E_net])
                gate_net =  self.gate_layer[j][i](x)
                g = tf.expand_dims(gate_net, axis = -1)
                _tower = tf.matmul(E_net, g, transpose_a = True)
                towers.append(k.layers.Flatten()(_tower))
            if j < self.n_cgc - 1:
                x = k.layers.Concatenate(axis = 1)(towers)
        return towers

    def get_config(self):
        base_config = super(CGC_Layer, self).get_config()
        config = {
            "n_task": self.n_task,
            "n_expert_share": self.share_expert_num,
            "n_expert_specific": self.specific_expert_num,
            "n_cgc": self.n_cgc
        }
        return dict(list(config.items()) + list(base_config.items()))