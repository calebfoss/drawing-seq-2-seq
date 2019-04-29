import tensorflow as tf
from tensorflow import keras
import numpy as np
from glob import glob
from utils import*
import random
layers = keras.layers

DATA_DIR = 'C:/Users/caleb/Documents/Tensorflow/combined_data'
SAMPLE_DIR = './js_samples_from_saved'
files = glob(os.path.join(DATA_DIR, '*.npz'))
file_len = len(files)
RNN_UNITS = 256
BUFFER_SIZE = 5000
BATCH_SIZE = 1024
E_DIM = 8
CAP_SEQ = 60
# Directory where the checkpoints will be saved
checkpoint_dir = './js_training_checkpoints'
log_dir = './logs'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")


if not os.path.exists(SAMPLE_DIR):
	os.mkdir(SAMPLE_DIR)

def loss(labels, logits):
	mask = keras.backend.all(keras.backend.not_equal(labels, 0), axis=-1)
	mask = keras.backend.cast(mask, dtype='float32')
	result = mask * tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
	return result
	# return tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
def squeeze(x):
	result = keras.backend.squeeze(x, 2)
	return result
initial_epoch = 0
assert os.path.isdir(checkpoint_dir)
assert len(glob(os.path.join(checkpoint_dir, 'ckpt_*'))) > 0
chkpt_files = glob(os.path.join(checkpoint_dir, 'ckpt_*'))
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
em_x = model.get_layer('em_x')
em_y = model.get_layer('em_y')
em_p = model.get_layer('em_p')
encoder_output, state_h, state_c = model.get_layer('encoder').output
encoder_states = [state_h, state_c]

de_in_x = model.get_layer('in_bottom_x').input
de_in_y = model.get_layer('in_bottom_y').input
de_in_p = model.get_layer('in_bottom_p').input

decoder_lstm = model.get_layer('decoder')
decoder_x = model.get_layer('out_x')
decoder_y = model.get_layer('out_y')
decoder_p = model.get_layer('out_p')
de_flat_merged = model.get_layer('de_flat_merged').output

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

data = np.load(files[random.randint(0, len(files) - 1)])
top_x = np.expand_dims(data['top_x'], -1)
top_y = np.expand_dims(data['top_y'], -1)
top_p = np.expand_dims(data['top_p'], -1)
NUM_DRAWINGS = len(top_x) // CAP_SEQ

for d in range(1):
	r = random.randint(0, (((NUM_DRAWINGS // CAP_SEQ) // 5) * (d + 1)) - 1)
	input_x = np.expand_dims(top_x[CAP_SEQ * r: CAP_SEQ * (r + 1)], 0)
	input_y = np.expand_dims(top_y[CAP_SEQ * r: CAP_SEQ * (r + 1)], 0)
	input_p = np.expand_dims(top_p[CAP_SEQ * r: CAP_SEQ * (r + 1)], 0)
	print(input_x, input_y, input_p)
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
		output_tokens_x, output_tokens_y, output_tokens_p, h, c = decoder_model.predict(
			[target_seq_x, target_seq_y, target_seq_p] + states_values)
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
		image_name = os.path.join(SAMPLE_DIR, 'sample_epoch{}_draw{}'.format(initial_epoch, d))
		top_bottom_to_image(image_name, np.squeeze(input_x), np.squeeze(input_y), np.squeeze(input_p),
							pred_x, pred_y, pred_p)
