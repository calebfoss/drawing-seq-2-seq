import tensorflowjs as tfjs
import tensorflow as tf
from tensorflow import keras
from glob import glob
import os
layers = keras.layers
checkpoint_dir = './js_training_checkpoints'
RNN_UNITS = 64
BUFFER_SIZE = 5000
BATCH_SIZE = 2048
E_DIM = 8
def loss(labels, logits):
	mask = keras.backend.all(keras.backend.not_equal(labels, 0), axis=-1)
	mask = keras.backend.cast(mask, dtype='float32')
	result = mask * tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
	return result
chkpt_files = glob(os.path.join(checkpoint_dir, 'ckpt_*'))
initial_epoch = 0
max_epoch_idx = 0
for f in range(len(chkpt_files)):
	split = chkpt_files[f].split('_')
	num = int(split[-1])
	if num > initial_epoch:
		initial_epoch = num
		max_epoch_idx = f
model = keras.models.load_model(chkpt_files[max_epoch_idx], custom_objects={'loss': loss})
# en_in_x = model.get_layer('in_top_x').input
# en_in_y = model.get_layer('in_top_y').input
# en_in_p = model.get_layer('in_top_p').input
# em_x = model.layers[6]
# em_y = model.layers[7]
# em_p = model.layers[8]
# encoder_output, state_h, state_c = model.layers[13].output
# encoder_states = [state_h, state_c]
#
# de_in_x = model.get_layer('in_bottom_x').input
# de_in_y = model.get_layer('in_bottom_y').input
# de_in_p = model.get_layer('in_bottom_p').input
#
# decoder_lstm = model.layers[14]
# decoder_x = model.layers[15]
# decoder_y = model.layers[16]
# decoder_p = model.layers[17]
# de_flat_merged = model.get_layer('reshape_1').output
# encoder_model = keras.Model(inputs=[en_in_x, en_in_y, en_in_p], outputs=encoder_states)
# decoder_state_input_h = layers.Input(shape=(RNN_UNITS,))
# decoder_state_input_c = layers.Input(shape=(RNN_UNITS,))
# decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
# decoder_outputs, state_h, state_c = decoder_lstm(de_flat_merged, initial_state=decoder_states_inputs)
# decoder_states = [state_h, state_c]
# de_out_x = decoder_x(decoder_outputs)
# de_out_y = decoder_y(decoder_outputs)
# de_out_p = decoder_p(decoder_outputs)
# decoder_model = keras.Model([de_in_x, de_in_y, de_in_p] + decoder_states_inputs,
# 							[de_out_x, de_out_y, de_out_p] + decoder_states)
tfjs.converters.save_keras_model(model, './model_js')