# coding:utf-8
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.layers import Convolution2D, MaxPooling2D, Masking, Embedding
from keras.layers import merge, Convolution1D, MaxPooling1D, Input, Flatten, LSTM, GRU, AveragePooling1D, Reshape
from keras.models import Model, model_from_json, model_from_config
from keras.layers import Dense, Dropout, Activation, Lambda, TimeDistributed, RepeatVector, Permute
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD
from keras.regularizers import l2, activity_l2
from keras.constraints import maxnorm
import keras
import numpy as np
from theano import tensor as T
import theano
from keras.utils import np_utils
from keras.callbacks import *
import os


def get_R(X):
    Y, alpha = X[0], X[1]
    ans = K.T.batched_dot(Y, alpha)
    return ans


def get_sent(X):  # 输入数据的末尾是两个实体，把其去掉拿到句子
    return X[:, :103, :]


def get_e1(X):
    return X[:, 103, :]


def get_e2(X):
    return X[:, 104, :]


def get_sum(X):
    return K.sum(X, axis=1, keepdims=False)


def build_model(batch_size, sent_size, e1e2_phrase_size, vec_size, embedding_weights, nb_class):
    pf_dim = 25
    input_sentence = Input(shape=(sent_size + 2,), dtype='int32', name='word')
    input_e1e2_phrase = Input(
        shape=(e1e2_phrase_size,), dtype='int32', name='e1e2_phrase')

    embedding = Embedding(input_dim=embedding_weights.shape[0], output_dim=vec_size,
                          weights=[embedding_weights], name="embed")

    input_ori = embedding(input_sentence)
    input_s = Dropout(0.7)(input_ori)
    input_e1e2_ori = embedding(input_e1e2_phrase)
    input_e1e2 = Dropout(0.7)(input_e1e2_ori)

    word_ori = Lambda(get_sent, output_shape=(sent_size, vec_size))(input_s)

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
    sentence_ori = merge([word_ori, pf_1, pf_2], mode='concat')
    sentence = merge([sentence_ori, vector_weight],
                     mode='mul')  # 把padding的位置都置0

    # 双向GRU
    forwards = GRU(100, consume_less='gpu', return_sequences=True)(sentence)
    backwards = GRU(100, consume_less='gpu',
                    return_sequences=True, go_backwards=True)(sentence)
    backwards = Lambda(lambda x: x[:, ::-1, :])(backwards)
    combine = merge([forwards, backwards], mode='concat')

    # CNNs 对e1_to_e2处理
    e1_e2_1 = Convolution1D(100, 3, input_shape=(e1e2_phrase_size, vec_size), W_constraint=maxnorm(3),
                            activation='relu', border_mode='same')(input_e1e2)
    e1_e2_1 = MaxPooling1D(e1e2_phrase_size)(e1_e2_1)
    e1_e2_1 = Flatten()(e1_e2_1)

    e1_e2_2 = Convolution1D(100, 4, input_shape=(e1e2_phrase_size, vec_size), W_constraint=maxnorm(3),
                            activation='relu', border_mode='same')(input_e1e2)
    e1_e2_2 = MaxPooling1D(e1e2_phrase_size)(e1_e2_2)
    e1_e2_2 = Flatten()(e1_e2_2)

    e1_e2_3 = Convolution1D(100, 5, input_shape=(e1e2_phrase_size, vec_size), W_constraint=maxnorm(3),
                            activation='relu', border_mode='same')(input_e1e2)
    e1_e2_3 = MaxPooling1D(e1e2_phrase_size)(e1_e2_3)
    e1_e2_3 = Flatten()(e1_e2_3)

    e1_e2_merge = merge([e1_e2_1, e1_e2_2, e1_e2_3],
                        mode='concat', concat_axis=1)
    e1_e2_dense = Dense(200, activation='tanh',
                        W_constraint=maxnorm(3))(e1_e2_merge)
    print 'e1_e2_dense', e1_e2_dense._keras_shape
    e1_e2_dense = Dropout(0.6)(e1_e2_dense)

    combine_trans = Permute((2, 1))(combine)

    # 计算weight
    a_merge = merge([combine, e1_e2_dense],
                    output_shape=(sent_size, 1), mode=get_R)
    a_softmax = Activation('softmax')(a_merge)
    print "a_softmax", a_softmax._keras_shape

    # 对隐层输出加权求和
    hidden_attention = merge(
        [combine_trans, a_softmax], output_shape=(200,), mode=get_R)

    # 分类
    out = Dense(100, activation='tanh',
                W_constraint=maxnorm(3))(hidden_attention)
    out = Dropout(0.2)(out)
    out = Dense(nb_class, activation='softmax', W_constraint=maxnorm(3))(out)
    model = Model([input_sentence, input_pf1, input_pf2,
                   input_e1e2_phrase, input_vector], out)

    return model


def train_model(x1_train, y_train, pf1_train, pf2_train, train_e1e2_phrase, vector_train, embedding_weights, save_to, split, nb_class, sw):
    model = build_model(50, 103, 45, 300, embedding_weights, nb_class)
    json_string = model.to_json()
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adadelta', metrics=['accuracy'])
    checkpointer = ModelCheckpoint(
        filepath="model_data/weights_%.2f.hdf5" % sw, verbose=1, save_best_only=True, monitor='val_acc')  # ,save_weights_only=True)

    v = model.fit([x1_train, pf1_train, pf2_train, train_e1e2_phrase, vector_train], [y_train],
                  nb_epoch=20, batch_size=20,
                  shuffle=True,
                  validation_split=split, callbacks=[checkpointer, History(), ProgbarLogger()])
    print v.history
    open('model_data/ly_model_architecture.json', 'w').write(json_string)

    return model


def test_model(pair_wise_cnn, x1_test, y_test, pf1_test, pf2_test, test_e1e2_phrase, vector_test, json_string, weight_path):
    pair_wise_cnn.load_weights(weight_path)
    pair_wise_cnn.compile(
        loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    score = pair_wise_cnn.evaluate(
        [x1_test, pf1_test, pf2_test, test_e1e2_phrase, vector_test], y_test, batch_size=32)
    prediction = pair_wise_cnn.predict(
        [x1_test, pf1_test, pf2_test, test_e1e2_phrase, vector_test], batch_size=32)
    return score, prediction


def visual_layer(train1, json_string, layer_idx, weight_path):
    pair_wise_cnn = model_from_json(json_string)
    pair_wise_cnn.load_weights(weight_path)  # ('./model_data/qpdmouth.hdf5')
    pair_wise_cnn.compile(
        loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print len(pair_wise_cnn.layers)
    layer_visual = theano.function([pair_wise_cnn.layers[0].input], pair_wise_cnn.layers[
                                   layer_idx].output, allow_input_downcast=True)
    layer_out = layer_visual(train1)
    return layer_out

if __name__ == '__main__':
    class_dict = {'1': 'Component-Whole(e1,e2)', '0': 'Other', '2': 'Cause-Effect(e1,e2)', '3': 'Entity-Destination(e1,e2)', '4': 'Content-Container(e1,e2)', '5': 'Member-Collection(e1,e2)', '6': 'Message-Topic(e1,e2)', '7': 'Instrument-Agency(e1,e2)', '8': 'Product-Producer(e1,e2)', '9': 'Entity-Origin(e1,e2)',
                  '10': 'Component-Whole(e2,e1)', '11': 'Cause-Effect(e2,e1)', '12': 'Entity-Destination(e2,e1)', '13': 'Content-Container(e2,e1)', '14': 'Member-Collection(e2,e1)', '15': 'Message-Topic(e2,e1)', '16': 'Instrument-Agency(e2,e1)', '17': 'Product-Producer(e2,e1)', '18': 'Entity-Origin(e2,e1)'}

    save_to = 'model/test_6_20.npz'
    nb_class = 19
    split = 0.253522
    data_dic = "data_vector_luyao_2label"
    model_dic = 'model_data'

    pre_dic = 'predictions'

    embedding_weights = np.load(
        '%s/embedding_weights.npz' % data_dic, 'rb')['embedding_weights']
    x_train = np.load('%s/train_x.npz' % data_dic, 'rb')['arr_0']
    y_train = np.load('%s/train_y.npz' % data_dic)['arr_0']
    x_test = np.load('%s/test_x.npz' % data_dic, 'rb')['arr_0']
    y_test = np.load('%s/test_y.npz' % data_dic)['arr_0']
    pf1_train = np.load('%s/train_pf1.npz' % data_dic)['arr_0']
    pf2_train = np.load('%s/train_pf2.npz' % data_dic)['arr_0']
    pf1_test = np.load('%s/test_pf1.npz' % data_dic)['arr_0']
    pf2_test = np.load('%s/test_pf2.npz' % data_dic)['arr_0']
    # vector_train = np.load('./data_vector/train_vector.npz')['arr_0']
    # vector_test = np.load('./data_vector/test_vector.npz')['arr_0']
    vector_train = np.load('%s/train_mword.npz' % data_dic)['arr_0']
    vector_test = np.load('%s/test_mword.npz' % data_dic)['arr_0']
    train_e1e2_phrase = np.load('%s/train_e1e2_phrase.npz' % data_dic)['arr_0']
    test_e1e2_phrase = np.load('%s/test_e1e2_phrase.npz' % data_dic)['arr_0']

    print x_train.shape
    print y_train.shape
    print x_test.shape
    print y_test.shape
    print train_e1e2_phrase.shape
    print test_e1e2_phrase.shape

    for sw in [1112]:
        model = train_model(x_train, y_train, pf1_train, pf2_train, train_e1e2_phrase,
                            vector_train, embedding_weights, save_to, split, nb_class, sw)
        json_string = model.to_json()

        for root, dirs, files in os.walk(model_dic):
            for filename in files:
                if '%.2f.hdf5' % sw in filename:
                    weight_path = os.path.join(root, filename)
                    score, prediction = test_model(model,
                                                   x_test, y_test, pf1_test, pf2_test, test_e1e2_phrase, vector_test, json_string, weight_path)
                    print filename + ' score:', score

                    with open("{pre_dic}/prediction_{name}".format(pre_dic=pre_dic, name=sw),
                              "w") as f_pre:
                        for num in range(len(prediction)):
                            f_pre.write(
                                str(np.argmax(prediction[num])) + "\t" + str(y_test[num][0]) + "\n")
