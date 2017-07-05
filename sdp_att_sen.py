from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.layers import Convolution2D, MaxPooling2D, Masking, Embedding
from keras.layers import merge, Convolution1D, MaxPooling1D, Input, Flatten, LSTM, GRU, AveragePooling1D, Reshape
from keras.models import Model, model_from_json, model_from_config
from keras.layers import Dense, Dropout, Activation, Lambda, TimeDistributed, RepeatVector, Permute
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.constraints import maxnorm
from keras.engine.topology import Layer
from keras import initializations
import keras
import numpy as np
from theano import tensor as T
import theano
from keras.utils import np_utils
from keras.callbacks import *
import os
import time
import sys

def myloss(y_true, y_pred):
    A = y_pred
    B = K.T.batched_dot(A, A.transpose(0,2,1))
    I = K.T.eye(n=A.shape[1],m=A.shape[1])
    return K.sqrt(K.sum(K.square(K.reshape(B-I, [B.shape[0],-1])), axis=-1))

def my_metric(y_true, y_pred):
    return K.mean(K.cast(K.equal(K.cast(K.max(y_true, axis=-1), K.floatx()), K.cast(K.argmax(y_pred, axis=-1), K.floatx())), K.floatx()))

def get_R(X):
    Y, alpha = X[0], X[1]
    ans = K.T.batched_dot(Y, alpha)
    return ans


def get_sent(X):
    return X[:, :103, :]


def get_e1(X):
    return X[:, 103, :]


def get_e2(X):
    return X[:, 104, :]


def get_sum(X):
    return K.sum(X, axis=1, keepdims=False)


class AttLayer(Layer):

    def __init__(self, **kwargs):
        self.init = initializations.get('normal')
        # self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        # self.W = self.init((input_shape[-1],1))
        self.W = self.init((input_shape[-1],))
        # self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W]
        # be sure you call this somewhere!
        super(AttLayer, self).build(input_shape)

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))

        ai = K.exp(eij)
        weights = ai/K.sum(ai, axis=1).dimshuffle(0, 'x')

        weighted_input = x*weights.dimshuffle(0, 1, 'x')
        return weighted_input.sum(axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])



class Multi_AttLayer(Layer):

    def __init__(self, **kwargs):
        self.init = initializations.get('normal')
        # self.input_spec = [InputSpec(ndim=3)]
        super(Multi_AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        # self.W = self.init((input_shape[-1],1))
        self.W_s1 = self.init((input_shape[-1], 10))
        self.W_s2 = self.init((10,))
        # self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W_s1, self.W_s2]
        # be sure you call this somewhere!
        super(Multi_AttLayer, self).build(input_shape)

    def call(self, x, mask=None):
        eij = K.dot(K.tanh(K.dot(x, self.W_s1)), self.W_s2)
        ai = K.exp(eij)
        weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')
        return weights.dimshuffle(0,1,'x')

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], 1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 1)


def build_multi_att_model(batch_size, sent_size, vec_size, embedding_weights, nb_class):
    pf_dim = 25
    input_sentence = Input(shape=(sent_size,), dtype='int32', name='word')
    embedding = Embedding(input_dim=embedding_weights.shape[
                          0], output_dim=vec_size, weights=[embedding_weights], name="embed")
    input_ori = embedding(input_sentence)
    input = Dropout(0.7)(input_ori)

    input_pf1 = Input(shape=(sent_size,), dtype='int32', name='position1')
    pf1_embedding = Embedding(
        input_dim=2 * sent_size, output_dim=pf_dim, init='normal')
    pf_1 = pf1_embedding(input_pf1)

    input_pf2 = Input(shape=(sent_size,), dtype='int32', name='position2')
    pf2_embedding = Embedding(
        input_dim=2 * sent_size, output_dim=pf_dim, init='normal')
    pf_2 = pf2_embedding(input_pf2)

    input_vector = Input(shape=(sent_size,), dtype='float32', name='vector')
    vector_weight_R = RepeatVector(350)(input_vector)
    vector_weight = Permute((2, 1))(vector_weight_R)
    sentence_ori = merge([input, pf_1, pf_2], mode='concat')
    sentence = merge([sentence_ori, vector_weight], mode='mul')

# MODEL
    forwards = GRU(100, consume_less='gpu', return_sequences=True)(sentence)
    backwards = GRU(
        100, consume_less='gpu', return_sequences=True, go_backwards=True)(sentence)
    backwards = Lambda(lambda x: x[:, ::-1, :])(backwards)
    combine = merge([forwards, backwards], mode='concat')
    combine_trans = Permute((2, 1))(combine)
    print 'combine_shape', combine._keras_shape

# multi random attention
    attention = Multi_AttLayer(name='attention')(combine)
    print 'attention_shape', attention._keras_shape

    hidden_attention = merge([combine_trans, attention],
                             output_shape=(200, 1), mode=get_R)
    hidden_attention = Reshape((200,))(hidden_attention)

    out = Dense(100, activation='tanh', W_constraint=maxnorm(3))(
        hidden_attention)
    out = Dropout(0.2)(out)
    out = Dense(nb_class, activation='softmax', W_constraint=maxnorm(3), name='out')(out)
    model = Model(
        [input_sentence, input_pf1, input_pf2, input_vector], [out, attention])

    return model

def build_sdp_att_model(batch_size, sent_size, sdp_size, vec_size, embedding_weights, nb_class):
    pf_dim = 25
    input_sentence = Input(shape=(sent_size,), dtype='int32', name='word')
    embedding = Embedding(input_dim=embedding_weights.shape[
                          0], output_dim=vec_size, weights=[embedding_weights], name="embed")
    input_ori = embedding(input_sentence)
    input = Dropout(0.7)(input_ori)

    input_sdp = Input(shape=(sdp_size,), dtype='int32', name='sdp')
    sdp = embedding(input_sdp)
    sdp = Dropout(0.7)(sdp)

    input_pf1 = Input(shape=(sent_size,), dtype='int32', name='position1')
    pf1_embedding = Embedding(
        input_dim=2 * sent_size, output_dim=pf_dim, init='normal')
    pf_1 = pf1_embedding(input_pf1)

    input_pf2 = Input(shape=(sent_size,), dtype='int32', name='position2')
    pf2_embedding = Embedding(
        input_dim=2 * sent_size, output_dim=pf_dim, init='normal')
    pf_2 = pf2_embedding(input_pf2)

    input_vector = Input(shape=(sent_size,), dtype='float32', name='vector')
    vector_weight_R = RepeatVector(350)(input_vector)
    vector_weight = Permute((2, 1))(vector_weight_R)
    sentence_ori = merge([input, pf_1, pf_2], mode='concat')
    sentence = merge([sentence_ori, vector_weight], mode='mul')

# MODEL
    forwards = GRU(100, consume_less='gpu', return_sequences=True)(sentence)
    backwards = GRU(
        100, consume_less='gpu', return_sequences=True, go_backwards=True)(sentence)
    backwards = Lambda(lambda x: x[:, ::-1, :])(backwards)
    combine = merge([forwards, backwards], mode='concat')

    e1_e2_1 = Convolution1D(100, 3, input_shape=(sdp_size, vec_size), W_constraint=maxnorm(3),
                            activation='relu', border_mode='same')(sdp)
    e1_e2_1 = MaxPooling1D(sdp_size)(e1_e2_1)
    e1_e2_1 = Flatten()(e1_e2_1)

    e1_e2_2 = Convolution1D(100, 4, input_shape=(sdp_size, vec_size), W_constraint=maxnorm(3),
                            activation='relu', border_mode='same')(sdp)
    e1_e2_2 = MaxPooling1D(sdp_size)(e1_e2_2)
    e1_e2_2 = Flatten()(e1_e2_2)

    e1_e2_3 = Convolution1D(100, 5, input_shape=(sdp_size, vec_size), W_constraint=maxnorm(3),
                            activation='relu', border_mode='same')(sdp)
    e1_e2_3 = MaxPooling1D(sdp_size)(e1_e2_3)
    e1_e2_3 = Flatten()(e1_e2_3)

    e1e2_merge = merge(
        [e1_e2_1, e1_e2_2, e1_e2_3], mode='concat', concat_axis=1)
    e1_e2_dense = Dense(
        200, activation='tanh', W_constraint=maxnorm(3))(e1e2_merge)
    print 'e1_e2_dense', e1_e2_dense._keras_shape
    e1_e2_dense = Dropout(0.6)(e1_e2_dense)

    combine_trans = Permute((2, 1))(combine)
    print "combine_trans", combine_trans._keras_shape

    a_merge = merge(
        [combine, e1_e2_dense], output_shape=(sent_size, 1), mode=get_R)

    a_softmax = Activation('softmax')(a_merge)
    print "a_softmax", a_softmax._keras_shape

    hidden_attention = merge(
        [combine_trans, a_softmax], output_shape=(200, 1), mode=get_R)

    print "hidden_attention", hidden_attention._keras_shape

    hidden_attention = Reshape((200,))(hidden_attention)

    out = Dense(100, activation='tanh', W_constraint=maxnorm(3))(
        hidden_attention)

    out = Dropout(0.2)(out)
    out = Dense(nb_class, activation='softmax', W_constraint=maxnorm(3))(out)
    model = Model(
        [input_sentence, input_sdp, input_pf1, input_pf2, input_vector], out)

    return model


def build_random_att_model(batch_size, sent_size, vec_size, embedding_weights, nb_class):
    pf_dim = 25
    input_sentence = Input(shape=(sent_size,), dtype='int32', name='word')
    embedding = Embedding(input_dim=embedding_weights.shape[
                          0], output_dim=vec_size, weights=[embedding_weights], name="embed")
    input_ori = embedding(input_sentence)
    input = Dropout(0.7)(input_ori)

    input_pf1 = Input(shape=(sent_size,), dtype='int32', name='position1')
    pf1_embedding = Embedding(
        input_dim=2 * sent_size, output_dim=pf_dim, init='normal')
    pf_1 = pf1_embedding(input_pf1)

    input_pf2 = Input(shape=(sent_size,), dtype='int32', name='position2')
    pf2_embedding = Embedding(
        input_dim=2 * sent_size, output_dim=pf_dim, init='normal')
    pf_2 = pf2_embedding(input_pf2)

    input_vector = Input(shape=(sent_size,), dtype='float32', name='vector')
    vector_weight_R = RepeatVector(350)(input_vector)
    vector_weight = Permute((2, 1))(vector_weight_R)
    sentence_ori = merge([input, pf_1, pf_2], mode='concat')
    sentence = merge([sentence_ori, vector_weight], mode='mul')

# MODEL
    forwards = GRU(100, consume_less='gpu', return_sequences=True)(sentence)
    backwards = GRU(
        100, consume_less='gpu', return_sequences=True, go_backwards=True)(sentence)
    backwards = Lambda(lambda x: x[:, ::-1, :])(backwards)
    combine = merge([forwards, backwards], mode='concat')

    # random attention
    hidden_attention = AttLayer()(combine)
    print 'hiddden_att_shape', hidden_attention._keras_shape
    out = Dense(100, activation='tanh', W_constraint=maxnorm(3))(
        hidden_attention)
    out = Dropout(0.2)(out)
    out = Dense(nb_class, activation='softmax', W_constraint=maxnorm(3))(out)
    model = Model(
        [input_sentence, input_pf1, input_pf2, input_vector], out)

    return model

def build_model(ml_choose, embedding_weights, nb_class):
    if ml_choose == 's':
        model = build_sdp_att_model(50, 101, 14, 300, embedding_weights, nb_class)
    elif ml_choose == 'm':
        model = build_multi_att_model(50, 101, 300, embedding_weights, nb_class)
    elif ml_choose == 'r':
        model = build_random_att_model(50, 101, 300, embedding_weights, nb_class)
    else:
        raise IOError('invalid model input')
    return model

def train_model(model, ml_choose, x_train, y_train, pf1_train, pf2_train, train_e1e2_phrase, vector_train, embedding_weights, split, nb_class):
    if ml_choose == 'm':
        model.compile(loss=["sparse_categorical_crossentropy", myloss], loss_weights=[1,0.1], optimizer='adadelta', metrics=[my_metric,])
        monitor = 'val_out_my_metric'
    else:
        model.compile(loss='sparse_categorical_crossentropy',optimizer='adadelta', metrics=['accuracy'])
        monitor = 'val_acc'

    checkpointer = ModelCheckpoint(filepath="model_data/weights_%s.hdf5"%ml_choose, verbose=1, save_best_only=True, monitor=monitor)  # ,save_weights_only=True)

    if ml_choose == 'm':
        attention_tmp = np.zeros(shape=[x_train.shape[0], x_train.shape[1], 1])
        v = model.fit([x_train, pf1_train, pf2_train, vector_train], [y_train, attention_tmp], nb_epoch=60, batch_size=20,
                  shuffle=True, validation_split=split, callbacks=[checkpointer, History(), ProgbarLogger()])
    elif ml_choose == 'r':
        v = model.fit([x_train, pf1_train, pf2_train, vector_train], [y_train], nb_epoch=60, batch_size=20,
                  shuffle=True, validation_split=split, callbacks=[checkpointer, History(), ProgbarLogger()])
    else:
        v = model.fit([x_train, train_e1e2_phrase, pf1_train, pf2_train, vector_train], [y_train], nb_epoch=60, batch_size=20,
                  shuffle=True, validation_split=split, callbacks=[checkpointer, History(),ProgbarLogger()])

    print v.history
    return model  # json_string


def test_model(ml_choose, model, x_test, y_test, pf1_test, pf2_test, test_e1e2_phrase, vector_test):
    model.load_weights("model_data/weights_%s.hdf5"%ml_choose)
    if ml_choose == 'm':
        model.compile(loss=["sparse_categorical_crossentropy", myloss], loss_weights=[1, 0.1], optimizer='adadelta', metrics=[my_metric,])
    else:
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    if ml_choose == 'm':
        att_tmp = np.zeros(shape=[x_test.shape[0], x_test.shape[1],1])
        score = model.evaluate([x_test, pf1_test, pf2_test, vector_test], [y_test,att_tmp], batch_size=32)
        prediction = model.predict([x_test, pf1_test, pf2_test, vector_test], batch_size=32)[0]

    elif ml_choose == 'r':
        score = model.evaluate([x_test, pf1_test, pf2_test, vector_test], y_test, batch_size=32)
        prediction = model.predict([x_test, pf1_test, pf2_test, vector_test], batch_size=32)

    else:
        score = model.evaluate([x_test, test_e1e2_phrase, pf1_test, pf2_test, vector_test], y_test, batch_size=32)
        prediction = model.predict([x_test, test_e1e2_phrase, pf1_test, pf2_test, vector_test], batch_size=32)

    print prediction.shape

    return score, prediction


def visual_layer(train1, json_string, layer_idx, weight_path):
    model = model_from_json(json_string)
    model.load_weights(weight_path)  # ('./model_data/qpdmouth.hdf5')
    model.compile(
        loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print len(model.layers)
    layer_visual = theano.function([model.layers[0].input], model.layers[
                                   layer_idx].output, allow_input_downcast=True)
    #layer_weight = theano.function([pair_wise_cnn.layers[0].input], pair_wise_cnn.get_weights(),allow_input_downcast=True)
    layer_out = layer_visual(train1)
    #layer_weight_out = layer_weight(train1)
    return layer_out
    # for layer in pair_wise_cnn.get_weights():
    #    print layer.shape

if __name__ == '__main__':
    class_dict = {'1': 'Component-Whole(e1,e2)', '0': 'Other', '2': 'Cause-Effect(e1,e2)', '3': 'Entity-Destination(e1,e2)', '4': 'Content-Container(e1,e2)', '5': 'Member-Collection(e1,e2)', '6': 'Message-Topic(e1,e2)', '7': 'Instrument-Agency(e1,e2)', '8': 'Product-Producer(e1,e2)', '9': 'Entity-Origin(e1,e2)',
                  '10': 'Component-Whole(e2,e1)', '11': 'Cause-Effect(e2,e1)', '12': 'Entity-Destination(e2,e1)', '13': 'Content-Container(e2,e1)', '14': 'Member-Collection(e2,e1)', '15': 'Message-Topic(e2,e1)', '16': 'Instrument-Agency(e2,e1)', '17': 'Product-Producer(e2,e1)', '18': 'Entity-Origin(e2,e1)'}
    np.random.seed(123)
    nb_class = 19
    split = 0.253522  # stan 2
    data_dic = "data_sdp_2label/gru"

    model_dic = 'model_data'
    if not os.path.exists(model_dic):
        os.mkdir(model_dic)

    pre_dic = 'predictions'
    if not os.path.exists(pre_dic):
        os.mkdir(pre_dic)

    embedding_weights = np.load(
        'data_sdp_2label/embedding_weights.npz', 'rb')['embedding_weights']
    x_train = np.load('%s/train_x.npz' % data_dic, 'rb')['arr_0'][:100]
    y_train = np.load('%s/train_y.npz' % data_dic)['arr_0'][:100]
    x_test = np.load('%s/test_x.npz' % data_dic, 'rb')['arr_0']
    y_test = np.load('%s/test_y.npz' % data_dic)['arr_0']
    pf1_train = np.load('%s/train_pf1.npz' % data_dic)['arr_0'][:100]
    pf2_train = np.load('%s/train_pf2.npz' % data_dic)['arr_0'][:100]
    pf1_test = np.load('%s/test_pf1.npz' % data_dic)['arr_0']
    pf2_test = np.load('%s/test_pf2.npz' % data_dic)['arr_0']

    vector_train = np.load('%s/train_mword.npz' % data_dic)['arr_0'][:100]
    vector_test = np.load('%s/test_mword.npz' % data_dic)['arr_0']
    train_e1e2_phrase = np.load('%s/train_sdp.npz' % data_dic)['arr_0'][:100]
    test_e1e2_phrase = np.load('%s/test_sdp.npz' % data_dic)['arr_0']

    print x_train.shape
    print y_train.shape
    print x_test.shape
    print y_test.shape
    print train_e1e2_phrase.shape
    print test_e1e2_phrase.shape
    print pf1_train.shape

    if len(sys.argv) != 3:
        raise IOError('arg num error')

    ml_choose = sys.argv[1]
    mode = sys.argv[2]

    model = build_model(ml_choose, embedding_weights, nb_class)
    time = str(int(time.time()))

    if mode == 'tr':
        os.system("mv model_data/weights_%s.hdf5 model_data/weights_%s_%s.hdf5"%(ml_choose, ml_choose, time))
        model = train_model(model, ml_choose, x_train, y_train, pf1_train, pf2_train, train_e1e2_phrase, vector_train, embedding_weights, split, nb_class)
    elif mode == 'co':
        model.load_weights("model_data/weights_%s.hdf5"%ml_choose)
        model = train_model(model, ml_choose, x_train, y_train, pf1_train, pf2_train, train_e1e2_phrase, vector_train, embedding_weights, split, nb_class)
    elif mode == 'te':
        pass
    else:
        raise IOError('invalid mode')

    score, prediction = test_model(ml_choose, model, x_test, y_test, pf1_test, pf2_test, test_e1e2_phrase, vector_test)
    print 'score:', score
    with open("{pre_dic}/prediction_{ml_choose}_{time}".format(pre_dic=pre_dic, ml_choose=ml_choose, time=time),"w") as f_pre:
        for num in range(len(prediction)):
            f_pre.write(str(np.argmax(prediction[num])) + "\t" + str(y_test[num][0]) + "\n")
