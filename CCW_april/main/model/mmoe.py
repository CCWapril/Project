import tensorflow as tf
from tensorflow import keras as k
from ..layers.core import DNN


class MMoE_Layer(k.layers.Layer):
    def __init__(self, expert_layer, expert_layer_drop, n_expert, activation="softmax", n_task=2, l2=0.0, use_bn=False):
        super(MMoE_Layer, self).__init__()  # 子类中调用父类的方法
        self.n_expert = n_expert
        self.n_task = n_task
        self.expert_layer = [
            DNN(expert_layer, dropout_rate=expert_layer_drop, l2_reg=l2, use_bn=use_bn)
            for i in range(n_expert)]
        self.gate_layers = [
            k.layers.Dense(n_expert, activation=activation) for i in range(n_task)]

    def call(self, x):
        # 构建多个专家网络
        E_net = [expert(x) for expert in self.expert_layer]
        E_net = k.layers.Concatenate(axis=1)([e[:, tf.newaxis, :] for e in E_net])  # 维度 (bs,n_expert,n_dims)
        # 构建多个门网络
        gate_net = [gate(x) for gate in self.gate_layers]  # 维度 n_task个(bs,n_expert)
        # towers计算：对应的门网络乘上所有的专家网络
        towers = []
        for i in range(self.n_task):
            # tf.expand_dims(,-1)最后一维上增加一维
            g = tf.expand_dims(gate_net[i], axis=-1)  # 维度(bs,n_expert,1)
            # tf.matmul()矩阵乘法
            _tower = tf.matmul(E_net, g, transpose_a=True)
            towers.append(k.layers.Flatten()(_tower))  # 维度(bs,n_dims)
        return towers

    def get_config(self):
        config = {"n_task": self.n_task, "n_expert": self.n_expert}
        base_config = super(MMoE_Layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def MLP(layer, layer_dropout, activation="relu", prefix='', l2=0.0, user_bn=False):
    model = k.models.Sequential()

    # Adding layers to the model
    for i, (units, dropout_rate) in enumerate(zip(layer, layer_dropout)):
        # Adding Dense layer
        model.add(k.layers.Dense(units, activation=None, kernel_regularizer=tf.keras.regularizers.l2(l2)
                                 , name=f'{prefix}dense_{i + 1}'))

        # Adding BatchNormalization layer if specified
        if user_bn:
            model.add(k.layers.BatchNormalization(name=f'{prefix}batchnorm_{i + 1}'))

        # Adding activation function
        model.add(k.layers.Activation(activation, name=f'{prefix}{activation}_{i + 1}'))

        # Adding Dropout layer
        if dropout_rate > 0.0:
            model.add(k.layers.Dropout(dropout_rate, name=f'{prefix}dropout_{i + 1}'))

    return model
