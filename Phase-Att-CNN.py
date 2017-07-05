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


def get_sent(X):
    return X[:, :105, :]


def get_e1(X):
    return X[:, 105, :]


def get_e2(X):
    return X[:, 106, :]


def get_sum(X):
    return K.sum(X, axis=2, keepdims=False)


def build_model(batch_size, sent_size, e1e2_phrase_size, vec_size, embedding_weights, nb_class):
    pf_dim = 10
    embedding = Embedding(input_dim=embedding_weights.shape[0], output_dim=vec_size,
                          weights=[embedding_weights], name="embed")

    input_sentence = Input(shape=(sent_size + 2,), dtype='int32', name='word')
    input_ori = embedding(input_sentence)
    input_drop = Dropout(0.7)(input_ori)
    word_ori = Lambda(get_sent, output_shape=(sent_size, vec_size))(input_drop)

    input_e1e2_phrase = Input(
        shape=(e1e2_phrase_size,), dtype='int32', name='e1e2_phrase')
    input_e1e2_ori = embedding(input_e1e2_phrase)
    input_e1e2 = Dropout(0.7)(input_e1e2_ori)

    # #word = Dropout(0.7)(word_ori)
    # entity1 = Lambda(get_e1, output_shape=(vec_size, ))(input)
    # #entity1 = Dropout(0.6)(entity1_ori)
    # entity2 = Lambda(get_e2, output_shape=(vec_size, ))(input)
    # #entity2 = Dropout(0.6)(entity2_ori)

    input_pf1 = Input(shape=(sent_size,), dtype='int32', name='position1')
    pf1_embedding = Embedding(
        input_dim=2 * sent_size, output_dim=pf_dim, init='normal')
    pf_1 = pf1_embedding(input_pf1)

    input_pf2 = Input(shape=(sent_size,), dtype='int32', name='position2')
    pf2_embedding = Embedding(
        input_dim=2 * sent_size, output_dim=pf_dim, init='normal')
    pf_2 = pf2_embedding(input_pf2)

    input_vector = Input(shape=(sent_size,), dtype='float32', name='vector')
    vector_weight_R = RepeatVector(320)(input_vector)
    vector_weight = Permute((2, 1))(vector_weight_R)

    sentence_ori = merge([word_ori, pf_1, pf_2], mode='concat')
    sentence = merge([sentence_ori, vector_weight], mode='mul')

    # sentence = Dropout(0.6)(sentence_com)

    # forwards = GRU(100, consume_less='gpu', return_sequences=True)(sentence)
    # backwards = GRU(
    #     100, consume_less='gpu', return_sequences=True, go_backwards=True)(sentence)
    # backwards = Lambda(lambda x: x[:, ::-1, :])(backwards)
    # combine = merge([forwards, backwards], mode='concat')
    # print 'combine_1', combine._keras_shape

    combine = Convolution1D(1000, 3, input_shape=(sent_size, vec_size + 2 * pf_dim),
                            activation='relu', border_mode='same')(sentence)
    print 'combine_2', combine._keras_shape
    # hidden_attention_0 = MaxPooling1D(sent_size)(combine)
    # hidden_attention_reshape = Flatten()(hidden_attention_0)

    # entity1_dense = Dense(100, activation='tanh')(entity1)
    # entity2_dense = Dense(100, activation='tanh')(entity2)

    # e1_e2 = merge([entity1, entity2], mode='concat')
    # e1_e2_dense = Dense(200, activation='tanh')(e1_e2)

    e1_e2_1 = Convolution1D(400, 3, input_shape=(e1e2_phrase_size, vec_size),
                            activation='relu', border_mode='same')(input_e1e2)
    e1_e2_1 = MaxPooling1D(e1e2_phrase_size - 2)(e1_e2_1)
    e1_e2_1 = Flatten()(e1_e2_1)

    e1_e2_2 = Convolution1D(400, 4, input_shape=(e1e2_phrase_size, vec_size),
                            activation='relu', border_mode='same')(input_e1e2)
    e1_e2_2 = MaxPooling1D(e1e2_phrase_size - 3)(e1_e2_2)
    e1_e2_2 = Flatten()(e1_e2_2)

    e1_e2_3 = Convolution1D(400, 5, input_shape=(e1e2_phrase_size, vec_size),
                            activation='relu', border_mode='same')(input_e1e2)
    e1_e2_3 = MaxPooling1D(e1e2_phrase_size - 4)(e1_e2_3)
    e1_e2_3 = Flatten()(e1_e2_3)

    e1_e2_merge = merge(
        [e1_e2_1, e1_e2_2, e1_e2_3], mode='concat', concat_axis=1)

    e1_e2_dense = Dense(1000, activation='tanh')(e1_e2_merge)
    e1_e2_drop = Dropout(0.6)(e1_e2_dense)
    e1_e2_rep = RepeatVector(sent_size)(e1_e2_drop)
    print 'e1_e2_rep', e1_e2_rep._keras_shape

    a_merge_ori = merge([combine, e1_e2_rep], mode='mul')
    a_merge = Lambda(get_sum, output_shape=(sent_size,))(a_merge_ori)
    # a_merge = merge(
    #     [combine, e1_e2_drop], output_shape=(sent_size, ), mode=get_R)
    a_softmax = Activation('softmax')(a_merge)
    print "a_softmax", a_softmax._keras_shape

    a_rep = RepeatVector(1000)(a_softmax)
    a_trans = Permute((2, 1))(a_rep)
    hidden_attention = merge([combine, a_trans], mode='mul')
    print 'hidden_attention', hidden_attention._keras_shape

    # hidden_attention_1 = MaxPooling1D(sent_size-3)(hidden_attention)
    # hidden_attention_2 = Flatten()(hidden_attention_1)
    hidden_attention_2 = Lambda(
        get_sum, output_shape=(sent_size,))(hidden_attention)

    # combine_trans = Permute((2, 1))(combine)
    # print "combine_trans", combine_trans._keras_shape

    # hidden_attention = merge(
    #     [combine_trans, a_softmax], output_shape=(200, 1), mode=get_R)
    # print "hidden_attention", hidden_attention._keras_shape

    # hidden_attention_reshape = Reshape((200,))(hidden_attention)
    # print "hidden_attention_reshape", hidden_attention_reshape._keras_shape

    out = Dense(100, activation='tanh')(hidden_attention_2)
    out = Dense(nb_class, activation='softmax')(out)
    model = Model([input_sentence, input_e1e2_phrase, input_pf1, input_pf2,
                   input_vector], out)

    return model


def train_model(x1_train, y_train, pf1_train, pf2_train, train_e1e2_phrase, vector_train, embedding_weights, save_to, split, nb_class):
    pair_wise_cnn = build_model(50, 105, 47, 300, embedding_weights, nb_class)
    json_string = pair_wise_cnn.to_json()
    # sparse_categorical_crossentropy
    # RMSprop
    pair_wise_cnn.compile(loss='sparse_categorical_crossentropy',
                          optimizer='adadelta', metrics=['accuracy'])
    # checkpointer = ModelCheckpoint(
    # filepath="model_data/lymouth.hdf5", verbose=1, save_best_only=True,
    # monitor='val_acc', save_weights_only=True)
    checkpointer = ModelCheckpoint(
        filepath="model_data/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
        verbose=1, save_best_only=True, monitor='val_acc',
        save_weights_only=True)

    v = pair_wise_cnn.fit([x1_train, train_e1e2_phrase, pf1_train, pf2_train, vector_train], [y_train],
                          nb_epoch=70, batch_size=20,
                          shuffle=True,
                          validation_split=split, callbacks=[checkpointer, History(), ProgbarLogger()])
    print v.history
    open('model_data/ly_model_architecture.json', 'w').write(json_string)

    return pair_wise_cnn  # json_string


def test_model(x1_test, y_test, pf1_test, pf2_test, test_e1e2_phrase, vector_test, json_string, weight_path, model):
    pair_wise_cnn = model  # model_from_json(json_string)
    pair_wise_cnn.load_weights(weight_path)  # ('./model_data/qpdmouth.hdf5')
    pair_wise_cnn.compile(
        loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    score = pair_wise_cnn.evaluate(
        [x1_test, test_e1e2_phrase, pf1_test, pf2_test, vector_test], y_test, batch_size=32)
    prediction = pair_wise_cnn.predict(
        [x1_test, test_e1e2_phrase, pf1_test, pf2_test, vector_test], batch_size=32)
    return score, prediction


def visual_layer(train1, json_string, layer_idx, weight_path):
    pair_wise_cnn = model_from_json(json_string)
    pair_wise_cnn.load_weights(weight_path)  # ('./model_data/qpdmouth.hdf5')
    pair_wise_cnn.compile(
        loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print len(pair_wise_cnn.layers)
    layer_visual = theano.function([pair_wise_cnn.layers[0].input], pair_wise_cnn.layers[
                                   layer_idx].output, allow_input_downcast=True)
    # layer_weight = theano.function([pair_wise_cnn.layers[0].input],
    # pair_wise_cnn.get_weights(),allow_input_downcast=True)
    layer_out = layer_visual(train1)
    # layer_weight_out = layer_weight(train1)
    return layer_out
    # for layer in pair_wise_cnn.get_weights():
    #    print layer.shape

if __name__ == '__main__':
    class_dict = {'1': 'Component-Whole(e1,e2)', '0': 'Other', '2': 'Cause-Effect(e1,e2)', '3': 'Entity-Destination(e1,e2)', '4': 'Content-Container(e1,e2)', '5': 'Member-Collection(e1,e2)', '6': 'Message-Topic(e1,e2)', '7': 'Instrument-Agency(e1,e2)', '8': 'Product-Producer(e1,e2)', '9': 'Entity-Origin(e1,e2)',
                  '10': 'Component-Whole(e2,e1)', '11': 'Cause-Effect(e2,e1)', '12': 'Entity-Destination(e2,e1)', '13': 'Content-Container(e2,e1)', '14': 'Member-Collection(e2,e1)', '15': 'Message-Topic(e2,e1)', '16': 'Instrument-Agency(e2,e1)', '17': 'Product-Producer(e2,e1)', '18': 'Entity-Origin(e2,e1)'}

    save_to = 'model/test_6_20.npz'
    nb_class = 19
    split = 0.1
    data_dic = "data_vector_luyao_cnn_2label/"
    os.system('rm -r model_data/')
    os.mkdir('model_data')
    os.system('rm -r predictions/')
    os.mkdir('predictions')

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

    model = train_model(x_train, y_train, pf1_train, pf2_train, train_e1e2_phrase,
                        vector_train, embedding_weights, save_to, split, nb_class)
    json_string = ''

    # json_string = open('model_data/ly_model_architecture.json', 'r').read()
    # model = build_model(50, 103, 45, 300, embedding_weights, nb_class)
    # json_string = model.to_json()
    # open('model_data/ly_model_architecture.json', 'w').write(json_string)
    # pair_wise_cnn = model_from_json(json_string)

    for root, dirs, files in os.walk('model_data'):
        for filename in files:
            if '.hdf5' in filename:
                weight_path = os.path.join(root, filename)
                score, prediction = test_model(
                    x_test, y_test, pf1_test, pf2_test, test_e1e2_phrase, vector_test, json_string, weight_path, model)
    # score,prediction =
    # test_model(x_train,y_train,pf1_train,pf2_train,vector_train,json_string)
                print filename + ' score:', score
                accuracy = filename.replace('.hdf5', '').split('-')[1]
                with open("predictions/prediction_%s" % accuracy, "w") as f_pre:
                    for num in range(len(prediction)):
                        f_pre.write(
                            str(np.argmax(prediction[num])) + "\t" + str(y_test[num][0]) + "\n")


# pre_num = 8001
# pre_num = 1
# for num in range(len(prediction)):
#     f_pre.write(
#         str(pre_num) + '\t' + class_dict[str(np.argmax(prediction[num]))] + "\n")
#     pre_num += 1

# for num in range(len(prediction)):
#     f_pre.write(
# str(np.argmax(prediction[num])) + "\t" + str(y_test[num][0]) + "\n")
