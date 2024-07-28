import tensorflow as tf
from mmoe import MLP  # Multilayer Perceptron


class ESMM(tf.keras.layers.Layer):
    def __init__(self,
                 layer_info,
                 dropout_info,
                 activation = "relu",
                 l2 = 0.0,
                 dnn_use_bn = False,
                 name='esmm',
                 **kwargs):
        super(ESMM, self).__init__(name=name, **kwargs)
        # 构建CVR和CTR双塔tower
        self.cvr_dnn = MLP(layer_info, dropout_info, activation=activation, l2=l2, user_bn = dnn_use_bn)
        self.ctr_dnn = MLP(layer_info, dropout_info, activation=activation, l2=l2, user_bn = dnn_use_bn)
        self.cvr_output = tf.keras.layers.Dense(1, activation='sigmoid', name='cvr')
        self.ctr_output = tf.keras.layers.Dense(1, activation='sigmoid', name='ctr')

        self.multply_layer = tf.keras.layers.Multiply(name="ctcvr")
    def call(self, inputs):
        ctr_output = self.ctr_dnn(inputs)
        cvr_output = self.cvr_dnn(inputs)
        ctr_pred = self.ctr_output(ctr_output)
        cvr_pred = self.cvr_output(cvr_output)
        # 计算pCTCVR=pCTR*pCVR
        ctcvr_pred = self.multply_layer([ctr_pred, cvr_pred])
        return ctr_pred, ctcvr_pred, cvr_pred