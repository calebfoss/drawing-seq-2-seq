import tensorflow as tf
from tensorflow import keras
import numpy as np
from glob import glob
from utils import*
import random
layers = keras.layers

# top_x, top_y, top_p, bottom_x, bottom_y, bottom_p = crop_top([0, 512, 256, 1], [0, 512, 1, 512], [1, 2, 1, 1])
# # top_bottom_to_image('stack_test', top_x, top_y, top_p, bottom_x, bottom_y, bottom_p)
# top_bottom_to_image('stack_test', bottom_x, bottom_y, bottom_p, bottom_x, bottom_y, bottom_p)
# print(top_x, top_y, top_p, bottom_x, bottom_y, bottom_p)
# exit()

CAP_SEQ = 60
DATA_DIR = 'C:/Users/caleb/Documents/Tensorflow/combined_data'
OUT_DIR = './stacks'
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
RNN_UNITS = 64


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
model.summary()
print('Restored model from {}'.format(chkpt_files[max_epoch_idx]))
en_in_x = model.get_layer('in_top_x').input
en_in_y = model.get_layer('in_top_y').input
en_in_p = model.get_layer('in_top_p').input
em_x = model.layers[6]
em_y = model.layers[7]
em_p = model.layers[8]
encoder_output, state_h, state_c = model.layers[13].output
encoder_states = [state_h, state_c]

de_in_x = model.get_layer('in_bottom_x').input
de_in_y = model.get_layer('in_bottom_y').input
de_in_p = model.get_layer('in_bottom_p').input

decoder_lstm = model.layers[14]
decoder_x = model.layers[15]
decoder_y = model.layers[16]
decoder_p = model.layers[17]
de_flat_merged = model.get_layer('reshape_1').output

encoder_model = keras.Model(inputs=[en_in_x, en_in_y, en_in_p], outputs=encoder_states)
decoder_state_input_h = layers.Input(shape=(RNN_UNITS,))
decoder_state_input_c = layers.Input(shape=(RNN_UNITS,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(de_flat_merged, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
de_out_x = decoder_x(decoder_outputs)
de_out_y = decoder_y(decoder_outputs)
de_out_p = decoder_p(decoder_outputs)
decoder_model = keras.Model([de_in_x, de_in_y, de_in_p] + decoder_states_inputs,
							[de_out_x, de_out_y, de_out_p] + decoder_states)


files = glob(os.path.join(DATA_DIR, '*.npz'))
file = files[0]
name = os.path.split(file)[-1].split('.')[0]
print('Loading {}'.format(name))
data = np.load(file)
top_x = np.expand_dims(data['top_x'], -1)
top_y = np.expand_dims(data['top_y'], -1)
top_p = np.expand_dims(data['top_p'], -1)
bottom_x = np.expand_dims(data['bottom_x'], -1)
bottom_y = np.expand_dims(data['bottom_y'], -1)
bottom_p = np.expand_dims(data['bottom_p'], -1)
NUM_DRAWINGS = len(top_x)

r = random.randint(0, (NUM_DRAWINGS//CAP_SEQ))
input_x = np.expand_dims(top_x[CAP_SEQ * r: CAP_SEQ * (r+1)], 0)
input_y = np.expand_dims(top_y[CAP_SEQ * r: CAP_SEQ * (r+1)], 0)
input_p = np.expand_dims(top_p[CAP_SEQ * r: CAP_SEQ * (r+1)], 0)
for z in range(5):
	states_values = encoder_model.predict(({'in_top_x': input_x, 'in_top_y': input_y, 'in_top_p': input_p}))
	target_seq_x = np.zeros((1, 1, 1))
	target_seq_y = np.zeros((1, 1, 1))
	target_seq_p = np.zeros((1, 1, 1))
	target_seq_p[0, 0, 0] = 3
	pred_x = np.zeros(CAP_SEQ)
	pred_y = np.zeros(CAP_SEQ)
	pred_p = np.zeros(CAP_SEQ)
	movement = False
	i = 0
	while i < CAP_SEQ and target_seq_p[0, 0, 0] != 4:
		# print(target_seq_x[0, 0, 0],target_seq_y[0, 0, 0],target_seq_p[0, 0, 0])
		output_tokens_x, output_tokens_y, output_tokens_p, h, c = decoder_model.predict([target_seq_x, target_seq_y, target_seq_p] + states_values)
		# for idx in range(len(output_tokens_x[0])):
		# 	x = np.argmax(output_tokens_x[0, idx, :])
		# 	y = np.argmax(output_tokens_y[0, idx, :])
		# 	p = np.argmax(output_tokens_p[0, idx, :])
		# 	print('idx:{} x:{} y:{} p{}'.format(idx, x, y, p))
		pred_x[i] = np.argmax(output_tokens_x[0, -1, :])
		pred_y[i] = np.argmax(output_tokens_y[0, -1, :])
		pred_p[i] = np.argmax(output_tokens_p[0, -1, :])
		if pred_x[i] != 256 and pred_y[i] != 256:
			movement = True
		target_seq_x = np.zeros((1, 1, 1))
		target_seq_y = np.zeros((1, 1, 1))
		target_seq_p = np.zeros((1, 1, 1))
		target_seq_x[0, 0, 0] = pred_x[i]
		target_seq_y[0, 0, 0] = pred_y[i]
		target_seq_p[0, 0, 0] = pred_p[i]
		states_values = [h, c]
		i += 1
	# print(pred_x, pred_y, pred_p)
	if movement:
		image_name = os.path.join(OUT_DIR, 'stack{}'.format(z))
		top_bottom_to_image(image_name, np.squeeze(input_x), np.squeeze(input_y), np.squeeze(input_p),
							pred_x, pred_y, pred_p)
		top_x, top_y, top_p, input_x, input_y, input_p = crop_top(pred_x, pred_y, pred_p)
		top_bottom_to_image(os.path.join(OUT_DIR, 'crop{}'.format(z)), top_x, top_y, top_p, input_x, input_y, input_p)
		input_x = np.expand_dims(input_x, 0)
		input_x = np.expand_dims(input_x, -1)
		input_y = np.expand_dims(input_y, 0)
		input_y = np.expand_dims(input_y, -1)
		input_p = np.expand_dims(input_p, 0)
		input_p = np.expand_dims(input_p, -1)