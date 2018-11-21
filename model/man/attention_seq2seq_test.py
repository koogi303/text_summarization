from keras import backend as K
from keras.layers import Input, Embedding, Bidirectional
from keras.layers import concatenate
from keras.layers import RNN, LSTM, LSTMCell, GRUCell
from keras.layers import TimeDistributed, Dense, Concatenate, Lambda
from keras.models import Model
from model.custom_layers import DenseAnnotationAttention


class seq2seq_attention:
    def __init__(self, num_encoder_tokens, embedding_dim,
                 hidden_dim, num_decoder_tokens):
        self.num_encoder_tokens = num_encoder_tokens
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_decoder_tokens = num_decoder_tokens

    def dense_maxout(self, x_):
        """Implements a dense maxout layer where max is taken
        over _two_ units"""
        x_ = Dense(self.hidden_dim)(x_)
        x_1 = x_[:, :self.hidden_dim // 2]
        x_2 = x_[:, self.hidden_dim // 2:]
        return K.max(K.stack([x_1, x_2], axis=-1), axis=-1, keepdims=False)

    def get_model(self):
        # Input text
        encoder_inputs = Input(shape=(None,))
        # Input summary
        decoder_inputs = Input(shape=(None,))

        # word embedding layer for text
        encoder_inputs_emb = Embedding(input_dim=self.num_encoder_tokens + 1,
                                       output_dim=self.embedding_dim,
                                       mask_zero=True)(encoder_inputs)
        # word embedding layer for summary
        decoder_inputs_emb = Embedding(input_dim=self.num_decoder_tokens + 1,
                                       output_dim=self.embedding_dim,
                                       mask_zero=True)(decoder_inputs)

        # Bidirectional LSTM encoder
        encoder_out = Bidirectional(LSTM(self.hidden_dim // 2,
                                        return_sequences=True,
                                        return_state=True),
                                    merge_mode='concat')(encoder_inputs_emb)

        encoder_o = encoder_out[0]
        initial_h_lstm = concatenate([encoder_out[1], encoder_out[2]])
        initial_c_lstm = concatenate([encoder_out[3], encoder_out[4]])
        initial_decoder_state = Dense(self.hidden_dim, activation='tanh')(concatenate([initial_h_lstm, initial_c_lstm]))

        # LSTM decoder + attention
        initial_attention_h = Lambda(lambda x: K.zeros_like(x)[:, 0, :])(encoder_o)
        initial_state = [initial_decoder_state, initial_attention_h]

        cell = DenseAnnotationAttention(cell=GRUCell(self.hidden_dim),
                                        units=self.hidden_dim,
                                        input_mode="concatenate",
                                        output_mode="cell_output")

        # TODO output_mode="concatenate", see TODO(3)/A
        decoder_o, decoder_h, decoder_c = RNN(cell=cell,
                                              return_sequences=True,
                                              return_state=True)(decoder_inputs_emb,
                                                                 initial_state=initial_state,
                                                                 constants=encoder_o)
        decoder_o = Dense(self.hidden_dim * 2)(concatenate([decoder_o, decoder_inputs_emb]))
        y_pred = TimeDistributed(Dense(self.num_decoder_tokens + 1,
                                       activation='softmax'))(decoder_o)

        model = Model([encoder_inputs, decoder_inputs], y_pred)
        return model


if __name__ == '__main__':
    model = seq2seq_attention(num_encoder_tokens=100, embedding_dim=64,
                              hidden_dim=128, num_decoder_tokens=50)
    summaryModel = model.get_model()
    summaryModel.compile(optimizer='Adam', loss='categorical_crossentropy')
