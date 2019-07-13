## set the deep learning network structure
from keras.layers import *
# from __future__ import print_function
from keras.activations import softmax

from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model
import numpy as np
# def compute_euclidean_match_score(l_r):

#     l, r = l_r

#     return 1. / (1. +

#         K.sqrt(

#             -2 * K.batch_dot(l, r, axes=[2, 2]) +

#             K.expand_dims(K.sum(K.square(l), axis=2), 2) +

#             K.expand_dims(K.sum(K.square(r), axis=2), 1)

#         )

#     )

#

#
import tensorflow as tf
def compute_cos_match_score(l_r):

    l, r = l_r

    return K.batch_dot(

        K.l2_normalize(l, axis=-1),

        K.l2_normalize(r, axis=-1),

        axes=[2, 2]

    )

def compute_euclidean_match_score(l_r):

    l, r = l_r

    denominator = 1. + K.sqrt(

        -2 * K.batch_dot(l, r, axes=[2, 2]) +

        K.expand_dims(K.sum(K.square(l), axis=2), 2) +

        K.expand_dims(K.sum(K.square(r), axis=2), 1)

    )

    denominator = K.maximum(denominator, K.epsilon())

    return 1. / denominator

# def compute_cos_match_score(l_r):

#     # K.batch_dot(

#     #     K.l2_normalize(l, axis=-1),

#     #     K.l2_normalize(r, axis=-1),

#     #     axes=[2, 2]

#     # )

#

#     l, r = l_r

#     denominator = K.sqrt(K.batch_dot(l, l, axes=[2, 2]) *

#                          K.batch_dot(r, r, axes=[2, 2]))

#     denominator = K.maximum(denominator, K.epsilon())

#     output = K.batch_dot(l, r, axes=[2, 2]) / denominator

#     # output = K.expand_dims(output, 1)

#     # denominator = K.maximum(denominator, K.epsilon())

#     return output


# def MatchScore(l, r, mode="euclidean"):
#
#     if mode == "euclidean":
#
#         return merge(
#
#             [l, r],
#
#             mode=compute_euclidean_match_score,
#
#             output_shape=lambda shapes: (None, shapes[0][1], shapes[1][1])
#
#         )
#
#     elif mode == "cos":
#
#         return merge(
#
#             [l, r],
#
#             mode=compute_cos_match_score,
#
#             output_shape=lambda shapes: (None, shapes[0][1], shapes[1][1])
#
#         )
#
#     elif mode == "dot":
#
#         return merge([l, r], mode="dot")
#
#     else:
#
#         raise ValueError("Unknown match score mode %s" % mode)

MAX_LEN =20
def create_pretrained_embedding(pretrained_weights_path, trainable=False, **kwargs):
    pretrained_weights = np.load(pretrained_weights_path)
    in_dim, out_dim = pretrained_weights.shape
    embedding = Embedding(in_dim, out_dim, weights=[pretrained_weights], trainable=trainable, **kwargs)
    return embedding
def unchanged_shape(input_shape):
    return input_shape
def substract(input_1, input_2):
    neg_input_2 = Lambda(lambda x: -x, output_shape=unchanged_shape)(input_2)
    out_ = Add()([input_1, neg_input_2])
    return out_
def submult(input_1, input_2):
    mult = Multiply()([input_1, input_2])
    sub = substract(input_1, input_2)
    out_ = Concatenate()([sub, mult])
    return out_
def apply_multiple(input_, layers):
    if not len(layers) > 1:
        raise ValueError('Layers list should contain more than 1 layer')
    else:
        agg_ = []
        for layer in layers:
            agg_.append(layer(input_))
        out_ = Concatenate()(agg_)
    return out_
def time_distributed(input_, layers):
    out_ = []
    node_ = input_
    for layer_ in layers:
        node_ = TimeDistributed(layer_)(node_)
    out_ = node_
    return out_
def soft_attention_alignment(input_1, input_2):
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1),   ##soft max to each column
                     output_shape=unchanged_shape)(attention)
    w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2), ## axis =2 soft max to each row
                                     output_shape=unchanged_shape)(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned
def repeat_matrix(x,rep,axis):
    x=K.repeat_elements(x,rep=rep,axis=axis)
    return x
def decomposable_attention(pretrained_embedding='word_vec/word_enc.npy',
                           projection_dim=200, projection_hidden=0, projection_dropout=0.3,
                           compare_dim=600, compare_dropout=0.2,
                           dense_dim=300, dense_dropout=0.2,
                           lr=1e-3, activation='elu', maxlen=MAX_LEN):
    q1 = Input(name='q1', shape=(maxlen,))
    q2 = Input(name='q2', shape=(maxlen,))
    # Embedding
    embedding = create_pretrained_embedding(pretrained_embedding,
                                            mask_zero=False)
    q1_embed = embedding(q1)
    q2_embed = embedding(q2)

    projection_layers = []
    if projection_hidden > 0:
        projection_layers.extend([
            Dense(projection_hidden, activation=activation),
            Dropout(rate=projection_dropout),
        ])
    projection_layers.extend([
        Dense(projection_dim, activation=None),
        Dropout(rate=projection_dropout),
    ])
    q1_encoded = time_distributed(q1_embed, projection_layers)
    q2_encoded = time_distributed(q2_embed, projection_layers)
    # Attention
    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)
    q1_aligned_1, q2_aligned_1 = soft_attention_alignment(q1_embed, q2_embed)


    # Compare
    q1_combined = Concatenate()([q1_embed, q1_encoded,submult(q1_embed,q2_aligned_1),
                                 submult(q1_encoded, q2_aligned)])
    q2_combined = Concatenate()([ q2_embed,q2_encoded, submult(q2_embed,q1_aligned_1),
                                  submult(q2_encoded, q1_aligned)])
    compare_layers = [
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
    ]
    q1_compare = time_distributed(q1_combined, compare_layers)
    q2_compare = time_distributed(q2_combined, compare_layers)


    # Aggregate
    q1_rep = apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    q2_rep = apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    # Classifier
    merged = Concatenate()([q1_rep, q2_rep])
    dense = BatchNormalization()(merged)
    dense = Dense(dense_dim, activation=activation)(dense)
    dense = Dropout(dense_dropout)(dense)
    dense = BatchNormalization()(dense)
    #
    # dense = Dense(dense_dim, activation=activation)(dense)
    # dense = Dropout(dense_dropout)(dense)


    out_= Dense(1, activation='sigmoid')(dense)




   ###################### ##add a loss

    def myloss(y_true, y_pred):
        loss1=(tf.reduce_mean(y_pred, 0)-0.3)**2
        myloss_=K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)+loss1

        return myloss_

   ######################################################### ##add a loss













    model = Model(inputs=[q1, q2], outputs=out_)
    model.compile(optimizer=Adam(lr=lr), loss=myloss,
                  metrics=['binary_crossentropy', 'accuracy'])
    return model
def esim(pretrained_embedding='word_vec/word_enc.npy',
         maxlen=MAX_LEN,
         lstm_dim=300,
         dense_dim=300,
         dense_dropout=0.3):
    q1 = Input(name='q1', shape=(maxlen,))
    q2 = Input(name='q2', shape=(maxlen,))
    # Embedding
    embedding = create_pretrained_embedding(pretrained_embedding, mask_zero=False)
    bn = BatchNormalization(axis=2)
    q1_embed = bn(embedding(q1))
    q2_embed = bn(embedding(q2))
    # Encode
    encode = Bidirectional(LSTM(lstm_dim, return_sequences=True))
    q1_encoded = encode(q1_embed)
    q2_encoded = encode(q2_embed)
    # Attention
    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)

    # Compose
    q1_combined = Concatenate()([q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned)])
    q2_combined = Concatenate()([q2_encoded, q1_aligned, submult(q2_encoded, q1_aligned)])
    compose = Bidirectional(LSTM(lstm_dim, return_sequences=True))
    q1_compare = compose(q1_combined)
    q2_compare = compose(q2_combined)
    # Aggregate
    q1_rep = apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    q2_rep = apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    # Classifier
    merged = Concatenate()([q1_rep, q2_rep])
    dense = BatchNormalization()(merged)
    dense = Dense(dense_dim, activation='elu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(dense_dropout)(dense)
    dense = Dense(dense_dim, activation='elu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(dense_dropout)(dense)
    out_ = Dense(1, activation='sigmoid')(dense)
    model = Model(inputs=[q1, q2], outputs=out_)
    model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['binary_crossentropy', 'accuracy'])
    return model














# def combine_(pretrained_embedding='word_vec/word_enc.npy',
#                            projection_dim=60, projection_hidden=0, projection_dropout=0.3,
#                            compare_dim=100, compare_dropout=0.2,
#                            dense_dim=300, dense_dropout=0.2,
#                            lr=1e-3, activation='elu', maxlen=MAX_LEN):
#     q1 = Input(name='q1', shape=(maxlen,))
#     q2 = Input(name='q2', shape=(maxlen,))
#     # Embedding
#     embedding = create_pretrained_embedding(pretrained_embedding,
#                                             mask_zero=False)
#     q1_embed = embedding(q1)
#     q2_embed = embedding(q2)
#     projection_layers = []
#     if projection_hidden > 0:
#         projection_layers.extend([
#             Dense(projection_hidden, activation=activation),
#             Dropout(rate=projection_dropout),
#         ])
#     projection_layers.extend([
#         Dense(projection_dim, activation=None),
#         Dropout(rate=projection_dropout),
#     ])
#     q1_encoded = time_distributed(q1_embed, projection_layers)
#     q2_encoded = time_distributed(q2_embed, projection_layers)
#
#
# ## add encoded by lstm
#     encode = Bidirectional(LSTM(60, return_sequences=True))
#     q1_encoded_lstm = encode(q1_embed)
#     q2_encoded_lstm = encode(q2_embed)
#     q1_encoded=Concatenate()([q1_encoded,q1_encoded_lstm])
#     q2_encoded=Concatenate()([q2_encoded,q2_encoded_lstm])
# ## add encoded by lstm
#     # Attention
#     q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)
#     # Compare
#     q1_combined = Concatenate()([q1_embed,q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned)])
#     q2_combined = Concatenate()([q2_embed,q2_encoded, q1_aligned, submult(q2_encoded, q1_aligned)])
#
#
#     compare_layers = [
#         Dense(compare_dim, activation=activation),
#         Dropout(compare_dropout),
#         Dense(compare_dim, activation=activation),
#         Dropout(compare_dropout),
#     ]
#     q1_compare = time_distributed(q1_combined, compare_layers)
#     q2_compare = time_distributed(q2_combined, compare_layers)
#
#     # Aggregate
#     q1_rep = apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
#     q2_rep = apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
#
#
# ### add the feature by lstm
#     # compose = Bidirectional(LSTM(60, return_sequences=True))
#     # q1_compare = compose(q1_combined)
#     # q2_compare = compose(q2_combined)
#     # # Aggregate
#     # q1_rep_lstm = apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
#     # q2_rep_lstm = apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
# ### add the feature by lstm
#
#
#     # Classifier
#     merged = Concatenate()([q1_rep, q2_rep])
#     dense = BatchNormalization()(merged)
#     dense = Dense(dense_dim, activation=activation)(dense)
#     dense = Dropout(dense_dropout)(dense)
#     dense = BatchNormalization()(dense)
#     #
#     # dense = Dense(dense_dim, activation=activation)(dense)
#     # dense = Dropout(dense_dropout)(dense)
#
#
#     out_ = Dense(1, activation='sigmoid')(dense)
#     model = Model(inputs=[q1, q2], outputs=out_)
#     model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy',
#                   metrics=['binary_crossentropy', 'accuracy'])
#     return model




# def combine_(
#         left_seq_len=MAX_LEN, right_seq_len=MAX_LEN, embed_dimensions=300, nb_filter=32, filter_widths=20,
#         depth=2, dropout=0.4, abcnn_1=True, abcnn_2=True, collect_sentence_representations=False, mode="euclidean",
#         batch_normalize=True):
#     assert depth >= 1, "Need at least one layer to build ABCNN"
#     assert not (depth == 1 and abcnn_2), "Cannot build ABCNN-2 with only one layer!"
#     if type(filter_widths) == int:
#         filter_widths = [filter_widths] * depth
#     assert len(filter_widths) == depth
#     print("Using %s match score" % mode)
#     left_sentence_representations = []
#     right_sentence_representations = []
#     left_input = Input(shape=(left_seq_len,))
#     right_input = Input(shape=(right_seq_len,))
#     embedding = create_pretrained_embedding(pretrained_weights_path='word_vec/word_enc.npy',
#                                             mask_zero=False)
#     left_embed = embedding(left_input)
#     right_embed = embedding(right_input)
#     # if batch_normalize:
#     #     left_embed = BatchNormalization()(left_embed)
#     #     right_embed = BatchNormalization()(right_embed)
#     filter_width = filter_widths.pop(0)
#
#     if abcnn_1:
#         match_score = MatchScore(left_embed, right_embed, mode=mode)
#         # compute attention
#         attention_left = TimeDistributed(
#             Dense(embed_dimensions, activation="relu"), input_shape=(left_seq_len, right_seq_len))(match_score)
#         match_score_t = Permute((2, 1))(match_score)
#
#         attention_right = TimeDistributed(
#
#             Dense(embed_dimensions, activation="relu"), input_shape=(right_seq_len, left_seq_len))(match_score_t)
#
#         left_reshape = Reshape((1, attention_left._keras_shape[1], attention_left._keras_shape[2]))
#
#         right_reshape = Reshape((1, attention_right._keras_shape[1], attention_right._keras_shape[2]))
#         attention_left = left_reshape(attention_left)
#         left_embed = left_reshape(left_embed)
#         attention_right = right_reshape(attention_right)
#         right_embed = right_reshape(right_embed)
#         # concat attention
#         # (samples, channels, rows, cols)
#         left_embed = merge([left_embed, attention_left], mode="concat", concat_axis=1)
#         right_embed = merge([right_embed, attention_right], mode="concat", concat_axis=1)
#         # Padding so we have wide convolution
#         left_embed_padded = ZeroPadding2D((filter_width - 1, 0))(left_embed)
#         right_embed_padded = ZeroPadding2D((filter_width - 1, 0))(right_embed)
#         # 2D convolutions so we have the ability to treat channels. Effectively, we are still doing 1-D convolutions.
#         conv_left = Convolution2D(
#             nb_filter=nb_filter, nb_row=filter_width, nb_col=embed_dimensions, activation="tanh",
#             border_mode="valid",
#             dim_ordering="th"
#         )(left_embed_padded)
#         # Reshape and Permute to get back to 1-D
#         conv_left = (Reshape((conv_left._keras_shape[1], conv_left._keras_shape[2])))(conv_left)
#         conv_left = Permute((2, 1))(conv_left)
#         conv_right = Convolution2D(
#             nb_filter=nb_filter, nb_row=filter_width, nb_col=embed_dimensions, activation="tanh",
#             border_mode="valid",
#             dim_ordering="th"
#         )(right_embed_padded)
#         # Reshape and Permute to get back to 1-D
#         conv_right = (Reshape((conv_right._keras_shape[1], conv_right._keras_shape[2])))(conv_right)
#         conv_right = Permute((2, 1))(conv_right)
#
#
#
#     else:
#         # Padding so we have wide convolution
#         left_embed_padded = ZeroPadding1D(filter_width - 1)(left_embed)
#         right_embed_padded = ZeroPadding1D(filter_width - 1)(right_embed)
#         conv_left = Convolution1D(nb_filter, filter_width, activation="tanh", border_mode="valid")(
#             left_embed_padded)
#         conv_right = Convolution1D(nb_filter, filter_width, activation="tanh", border_mode="valid")(
#             right_embed_padded)
#     # if batch_normalize:
#     #     conv_left = BatchNormalization()(conv_left)
#     #     conv_right = BatchNormalization()(conv_right)
#     conv_left = Dropout(dropout)(conv_left)
#     conv_right = Dropout(dropout)(conv_right)
#     pool_left = AveragePooling1D(pool_length=filter_width, stride=1, border_mode="valid")(conv_left)
#     pool_right = AveragePooling1D(pool_length=filter_width, stride=1, border_mode="valid")(conv_right)
#     assert pool_left._keras_shape[1] == left_seq_len, "%s != %s" % (pool_left._keras_shape[1], left_seq_len)
#
#     assert pool_right._keras_shape[1] == right_seq_len, "%s != %s" % (pool_right._keras_shape[1], right_seq_len)
#
#     if collect_sentence_representations or depth == 1:  # always collect last layers global representation
#
#         left_sentence_representations.append(GlobalAveragePooling1D()(conv_left))
#
#         right_sentence_representations.append(GlobalAveragePooling1D()(conv_right))
#
#     # ###################### #
#
#     # ### END OF ABCNN-1 ### #
#
#     # ###################### #
#
#
#
#     for i in range(depth - 1):
#
#         filter_width = filter_widths.pop(0)
#
#         pool_left = ZeroPadding1D(filter_width - 1)(pool_left)
#
#         pool_right = ZeroPadding1D(filter_width - 1)(pool_right)
#
#         # Wide convolution
#
#         conv_left = Convolution1D(nb_filter, filter_width, activation="tanh", border_mode="valid")(pool_left)
#
#         conv_right = Convolution1D(nb_filter, filter_width, activation="tanh", border_mode="valid")(pool_right)
#
#         if abcnn_2:
#             conv_match_score = MatchScore(conv_left, conv_right, mode=mode)
#
#             # compute attention
#
#             conv_attention_left = Lambda(lambda match: K.sum(match, axis=-1),
#                                          output_shape=(conv_match_score._keras_shape[1],))(conv_match_score)
#
#             conv_attention_right = Lambda(lambda match: K.sum(match, axis=-2),
#                                           output_shape=(conv_match_score._keras_shape[2],))(conv_match_score)
#
#             conv_attention_left = Permute((2, 1))(RepeatVector(nb_filter)(conv_attention_left))
#
#             conv_attention_right = Permute((2, 1))(RepeatVector(nb_filter)(conv_attention_right))
#
#             # apply attention  TODO is "multiply each value by the sum of it's respective attention row/column" correct?
#
#             conv_left = merge([conv_left, conv_attention_left], mode="mul")
#
#             conv_right = merge([conv_right, conv_attention_right], mode="mul")
#
#         # if batch_normalize:
#
#         #     conv_left = BatchNormalization()(conv_left)
#
#         #     conv_right = BatchNormalization()(conv_right)
#
#
#
#         conv_left = Dropout(dropout)(conv_left)
#
#         conv_right = Dropout(dropout)(conv_right)
#
#         pool_left = AveragePooling1D(pool_length=filter_width, stride=1, border_mode="valid")(conv_left)
#
#         pool_right = AveragePooling1D(pool_length=filter_width, stride=1, border_mode="valid")(conv_right)
#
#         assert pool_left._keras_shape[1] == left_seq_len
#
#         assert pool_right._keras_shape[1] == right_seq_len
#
#         if collect_sentence_representations or (
#             i == (depth - 2)):  # always collect last layers global representation
#
#             left_sentence_representations.append(GlobalAveragePooling1D()(conv_left))
#
#             right_sentence_representations.append(GlobalAveragePooling1D()(conv_right))
#     # Merge collected sentence representations if necessary
#
#     left_sentence_rep = left_sentence_representations.pop(-1)
#
#     if left_sentence_representations:
#         left_sentence_rep = merge([left_sentence_rep] + left_sentence_representations, mode="concat")
#
#     right_sentence_rep = right_sentence_representations.pop(-1)
#
#     if right_sentence_representations:
#         right_sentence_rep = merge([right_sentence_rep] + right_sentence_representations, mode="concat")
#     global_representation = merge([left_sentence_rep, right_sentence_rep], mode="concat")
#     global_representation = Dropout(dropout)(global_representation)
#     classify = Dense(1, activation="sigmoid")(global_representation)
#     model = Model(inputs=[left_input, right_input], outputs=classify)
#     model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy',
#                   metrics=['binary_crossentropy', 'accuracy'])
#     return model




































































































































