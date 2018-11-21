import os
import heapq
import numpy as np

from keras import backend as K
from keras import initializers, regularizers, constraints
from keras.engine.base_layer import _collect_previous_mask
from keras.layers import Layer, InputSpec
from keras.layers import Input, Embedding, Bidirectional, RNN, GRU, GRUCell
from keras.layers import TimeDistributed, Dense, concatenate, Lambda
from keras.models import Model
from keras.optimizers import Adadelta
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.generic_utils import has_arg


# Inference
# https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
# Let's use the model to translate new sentences! To do this efficiently, two
# things must be done in preparation:
#  1) Build separate model for the encoding that is only done _once_ per input
#     sequence.
#  2) Build a model for the decoder (and output layers) that takes input states
#     and returns updated states for the recurrent part of the model, so that
#     it can be run one step at a time.
encoder_model = Model(x, [x_enc] + initial_state)

x_enc_new = Input(batch_shape=K.int_shape(x_enc))
initial_state_new = [Input((size,)) for size in cell.state_size]
h1_and_state_new = decoder_rnn(y_emb,
                               initial_state=initial_state_new,
                               constants=x_enc_new)
h1_new = h1_and_state_new[0]
updated_state = h1_and_state_new[1:]
h2_new = maxout_layer(concatenate([h1_new, y_emb]))
y_pred_new = output_layer(h2_new)
decoder_model = Model([y, x_enc_new] + initial_state_new,
                      [y_pred_new] + updated_state)


def translate_beam_search(input_text,
                          search_width=20,
                          branch_factor=None,
                          t_max=None):
    """Perform beam search to approximately find the translated sentence that
    maximises the conditional probability given the input sequence.

    Returns the completed sentences (reached end-token) in order of decreasing
    score (the first is most probable) followed by incomplete sentences in order
    of decreasing score - as well as the score for the respective sentence.

    References:
        [1] "Sequence to sequence learning with neural networks"
        (https://arxiv.org/pdf/1409.3215.pdf)
    """

    if branch_factor is None:
        branch_factor = search_width
    elif branch_factor > search_width:
        raise ValueError("branch_factor must be smaller than search_width")
    elif branch_factor < 2:
        raise ValueError("branch_factor must be >= 2")

    def k_largest_val_idx(a, k):
        """Returns top k largest values of a and their indices, ordered by
        decreasing value"""
        top_k = np.argpartition(a, -k)[-k:]
        return sorted(zip(a[top_k], top_k))[::-1]

    # initialisation of search
    t = 0
    y_0 = np.array(target_tokenizer.texts_to_sequences([start_token]))[0]
    end_idx = target_tokenizer.word_index[end_token]

    # run input encoding once
    x_ = np.array(input_tokenizer.texts_to_sequences([input_text]))
    encoder_output = encoder_model.predict(x_)
    x_enc_ = encoder_output[0]
    state_t = encoder_output[1:]
    # repeat to a batch of <search_width> samples
    x_enc_ = np.repeat(x_enc_, search_width, axis=0)

    if t_max is None:
        t_max = x_.shape[-1] * 2

    # A "search beam" is represented as the tuple:
    #   (score, outputs, state)
    # where:
    #   score: the average log likelihood of the output tokens
    #   outputs: the history of output tokens up to time t, [y_0, ..., y_t]
    #   state: the most recent state of the decoder_rnn for this beam

    # A list of the <search_width> number of beams with highest score is
    # maintained through out the search. Initially there is only one beam.
    incomplete_beams = [(0., [y_0], [s[0] for s in state_t])]
    # All beams that reached the end-token are kept separately.
    complete_beams = []

    while len(complete_beams) < search_width and t < t_max:
        t += 1
        # create a batch of inputs representing the incomplete_beams
        y_tm1 = np.vstack([beam[1][-1] for beam in incomplete_beams])
        state_tm1 = [
            np.vstack([beam[2][i] for beam in incomplete_beams])
            for i in range(len(state_t))
        ]

        # inference next tokes for every incomplete beam
        batch_size = len(incomplete_beams)
        decoder_output = decoder_model.predict(
            [y_tm1, x_enc_[:batch_size]] + state_tm1)
        y_pred_ = decoder_output[0]
        state_t = decoder_output[1:]
        # from each previous beam create new candidate beams and save the once
        # with highest score for next iteration.
        beams_updated = []
        for i, beam in enumerate(incomplete_beams):
            l = len(beam[1]) - 1  # don't count 'start' token
            for proba, idx in k_largest_val_idx(y_pred_[i, 0], branch_factor):
                new_score = (beam[0] * l + np.log(proba)) / (l + 1)
                not_full = len(beams_updated) < search_width
                ended = idx == end_idx
                if not_full or ended or new_score > beams_updated[0][0]:
                    # create new successor beam with next token=idx
                    beam_new = (new_score,
                                beam[1] + [np.array([idx])],
                                [s[i] for s in state_t])
                    if ended:
                        complete_beams.append(beam_new)
                    elif not_full:
                        heapq.heappush(beams_updated, beam_new)
                    else:
                        heapq.heapreplace(beams_updated, beam_new)
                else:
                    # if score is not among to candidates we abort search
                    # for this ancestor beam (next token processed in order of
                    # decreasing likelihood)
                    break
        # faster to process beams in order of decreasing score next iteration,
        # due to break above
        incomplete_beams = sorted(beams_updated, reverse=True)

    # want to return in order of decreasing score
    complete_beams = sorted(complete_beams, reverse=True)

    output_texts = []
    scores = []
    for beam in complete_beams + incomplete_beams:
        output_texts.append(target_tokenizer.sequences_to_texts(
            np.concatenate(beam[1])[None, :])[0])
        scores.append(beam[0])

    return output_texts, scores


# Translate one of sentences from validation data
input_text = input_texts_val[0]
print("Translating:\n", input_text)
output_greedy, score_greedy = translate_greedy(input_text)
print("Greedy output:\n", output_greedy)
outputs_beam, scores_beam = translate_beam_search(input_text)
print("Beam search output:\n", outputs_beam[0])

#
# encoder_model = Model(encoder_inputs, encoder_states)
#
# decoder_state_input_h = Input(shape=(latent_dim,))
# decoder_state_input_c = Input(shape=(latent_dim,))
# decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
# decoder_outputs, state_h, state_c = decoder_lstm(
#     decoder_inputs, initial_state=decoder_states_inputs)
# decoder_states = [state_h, state_c]
# decoder_outputs = decoder_dense(decoder_outputs)
# decoder_model = Model(
#     [decoder_inputs] + decoder_states_inputs,
#     [decoder_outputs] + decoder_states)
#
# def decode_sequence(input_seq):
#     # Encode the input as state vectors.
#     states_value = encoder_model.predict(input_seq)
#
#     # Generate empty target sequence of length 1.
#     target_seq = np.zeros((1, 1, num_decoder_tokens))
#     # Populate the first character of target sequence with the start character.
#     target_seq[0, 0, target_token_index['\t']] = 1.
#
#     # Sampling loop for a batch of sequences
#     # (to simplify, here we assume a batch of size 1).
#     stop_condition = False
#     decoded_sentence = ''
#     while not stop_condition:
#         output_tokens, h, c = decoder_model.predict(
#             [target_seq] + states_value)
#
#         # Sample a token
#         sampled_token_index = np.argmax(output_tokens[0, -1, :])
#         sampled_char = reverse_target_char_index[sampled_token_index]
#         decoded_sentence += sampled_char
#
#         # Exit condition: either hit max length
#         # or find stop character.
#         if (sampled_char == '\n' or
#            len(decoded_sentence) > max_decoder_seq_length):
#             stop_condition = True
#
#         # Update the target sequence (of length 1).
#         target_seq = np.zeros((1, 1, num_decoder_tokens))
#         target_seq[0, 0, sampled_token_index] = 1.
#
#         # Update states
#         states_value = [h, c]
#
#     return decoded_sentence