# https://github.com/hamelsmu/Seq2Seq_Tutorial
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding


class seq2seq:
    def __init__(self, num_encoder_tokens, embedding_dim,
                 hidden_dim, num_decoder_tokens):
        self.num_encoder_tokens = num_encoder_tokens
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_decoder_tokens = num_decoder_tokens

    def get_model(self):
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None,))
        encoder_inputs_emb = Embedding(input_dim=self.num_encoder_tokens,
                                       output_dim=self.embedding_dim,
                                       mask_zero=True)(encoder_inputs)
        encoder = LSTM(self.hidden_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs_emb)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]
        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None,))
        decoder_inputs_emb = Embedding(input_dim=self.num_decoder_tokens,
                                       output_dim=self.embedding_dim,
                                       mask_zero=True)(decoder_inputs)
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(self.hidden_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs_emb,
                                             initial_state=encoder_states)
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        return model
