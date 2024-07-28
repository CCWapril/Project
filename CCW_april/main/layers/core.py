import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2


def DNN(expert_layer, dropout_rate=0.0, l2_reg=0.0, use_bn=True):
    model = Sequential()

    # Adding the first hidden layer
    model.add(Dense(units=expert_layer, activation='relu', kernel_regularizer=l2(l2_reg)))
    if use_bn:
        model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Adding subsequent hidden layers
    for _ in range(expert_layer - 1):
        model.add(Dense(units=expert_layer, activation='relu', kernel_regularizer=l2(l2_reg)))
        if use_bn:
            model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    # Adding the output layer
    model.add(Dense(units=1, activation='softmax'))  # Assuming it's a classification task

    return model
