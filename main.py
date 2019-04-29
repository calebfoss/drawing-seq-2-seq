import tensorflow as tf
from tensorflow import keras
import numpy as np
from glob import glob
from utils import*
import random
layers = keras.layers

DATA_DIR = 'C:/Users/caleb/Documents/Tensorflow/combined_data'
SAMPLE_DIR = './samples'
files = glob(os.path.join(DATA_DIR, '*.npz'))
file_len = len(files)
RNN_UNITS = 64
BUFFER_SIZE = 5000
BATCH_SIZE = 2048
E_DIM = 8
# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
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
if not os.path.isdir(checkpoint_dir):
	os.mkdir(checkpoint_dir)
if len(glob(os.path.join(checkpoint_dir, 'ckpt_*'))) > 0:
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
else:
	en_in_x = layers.Input(shape=(None, 1,), dtype='uint16', name='in_top_x')
	en_in_y = layers.Input(shape=(None, 1,), dtype='uint16', name='in_top_y')
	en_in_p = layers.Input(shape=(None, 1,), dtype='uint8', name='in_top_p')
	em_x = layers.Embedding(input_dim=512, output_dim=E_DIM)
	em_y = layers.Embedding(input_dim=512, output_dim=E_DIM)
	em_p = layers.Embedding(input_dim=5, output_dim=E_DIM)
	en_emb_x = em_x(en_in_x)
	en_emb_y = em_y(en_in_y)
	en_emb_pen = em_p(en_in_p)
	en_merged_layer = layers.concatenate([en_emb_x, en_emb_y, en_emb_pen], axis=-1)
	en_flat_merged = layers.Reshape((-1, E_DIM * 3))(en_merged_layer)

	encoder = layers.CuDNNLSTM(RNN_UNITS, return_state=True, name='encoder')
	encoder_outputs, state_h, state_c = encoder(en_flat_merged)
	encoder_states = [state_h, state_c]

	de_in_x = layers.Input(shape=(None, 1,), dtype='uint16', name='in_bottom_x')
	de_in_y = layers.Input(shape=(None, 1,), dtype='uint16', name='in_bottom_y')
	de_in_p = layers.Input(shape=(None, 1,), dtype='uint8', name='in_bottom_p')
	de_emb_x = em_x(de_in_x)
	de_emb_y = em_y(de_in_y)
	de_emb_pen = em_p(de_in_p)
	de_merged_layer = layers.concatenate([de_emb_x, de_emb_y, de_emb_pen], axis=-1)
	de_flat_merged = layers.Reshape((-1, E_DIM * 3))(de_merged_layer)

	decoder_lstm = layers.CuDNNLSTM(RNN_UNITS, return_sequences=True, return_state=True, name='decoder')
	decoder_outputs, _, _ = decoder_lstm(de_flat_merged, initial_state=encoder_states)
	decoder_x = layers.Dense(512, activation='softmax', name='out_x')
	decoder_y = layers.Dense(512, activation='softmax', name='out_y')
	decoder_p = layers.Dense(5, activation='softmax', name='out_p')
	out_x = decoder_x(decoder_outputs)
	out_y = decoder_y(decoder_outputs)
	out_p = decoder_p(decoder_outputs)
	model = keras.Model(inputs=[en_in_x, en_in_y, en_in_p, de_in_x, de_in_y, de_in_p],
						outputs=[out_x, out_y, out_p])
	model.compile(loss=loss, optimizer='adam')
	model.summary(positions=[25, 75, 100, 125])
	print('Made a new model')

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

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=False)
if not os.path.isdir(log_dir):
	os.mkdir(log_dir)
log_callback=tf.keras.callbacks.TensorBoard(
	log_dir=log_dir,
	write_graph=True)

# def generator():
# 	for file in files:
#
CAP_SEQ = 60
in_top_x, in_top_y, in_top_p, in_bottom_x, in_bottom_y, in_bottom_p, out_x, out_y, out_p = \
	tf.placeholder(tf.uint16, shape=[None, 1]), tf.placeholder(tf.uint16, shape=[None, 1]), tf.placeholder(tf.uint8, shape=[None, 1]), \
	tf.placeholder(tf.uint16, shape=[None, 1]), tf.placeholder(tf.uint16, shape=[None, 1]), tf.placeholder(tf.uint8, shape=[None, 1]), \
	tf.placeholder(tf.uint16, shape=[None, 1]), tf.placeholder(tf.uint16, shape=[None, 1]), tf.placeholder(tf.uint8, shape=[None, 1])
dataset = tf.data.Dataset.from_tensor_slices(({'in_top_x': in_top_x, 'in_top_y': in_top_y, 'in_top_p': in_top_p,
													   "in_bottom_x": in_bottom_x, "in_bottom_y": in_bottom_y,
													   "in_bottom_p": in_bottom_p},
													  {"out_x": out_x, "out_y": out_y,
													   "out_p": out_p}))
dataset = dataset.batch(CAP_SEQ)
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
iter = dataset.make_initializable_iterator()
features, labels = iter.get_next()


def shift(a):
	result = np.expand_dims(np.append(np.delete(a, 0), 0), 1)
	result[CAP_SEQ - 1::CAP_SEQ] = 0
	return result.astype(a.dtype)

class InitIteratorCallback(tf.keras.callbacks.Callback):
	def on_epoch_begin(self, epoch, logs=None):
		print('Switching data to {}'.format(files[f]))
		sess = keras.backend.get_session()
		sess.run(iter.initializer, feed_dict={in_top_x: top_x, in_top_y: top_y, in_top_p: top_p,
												  in_bottom_x: bottom_x, in_bottom_y: bottom_y, in_bottom_p: bottom_p,
												  out_x: shift(bottom_x), out_y: shift(bottom_y),
												  out_p: shift(bottom_p)})
top_x = top_y = top_p = bottom_x = bottom_y = bottom_p = np.empty((0))
f = 0
full = 0
while True:
	f = 0
	print('Full epoch {}'.format(full))
	while f < file_len:
		file = files[f]
		name = os.path.split(file)[-1].split('.')[0]
		print('Loading {}'.format(name))
		data = np.load(file)
		top_x = np.expand_dims(data['top_x'], -1)
		top_y = np.expand_dims(data['top_y'], -1)
		top_p = np.expand_dims(data['top_p'], -1)
		bottom_x = np.expand_dims(data['bottom_x'], -1)
		bottom_y = np.expand_dims(data['bottom_y'], -1)
		bottom_p = np.expand_dims(data['bottom_p'], -1)


		# top_x, top_y, top_p, bottom_x, bottom_y, bottom_p = file_to_split(file, CAP_SEQ)
		NUM_DRAWINGS = len(top_x) // CAP_SEQ




		# dataset = tf.data.Dataset.from_tensor_slices(({'in_top_x': top_x, 'in_top_y': top_y, 'in_top_p': top_p,
		# 											   "in_bottom_x": bottom_x, "in_bottom_y": bottom_y,
		# 											   "in_bottom_p": bottom_p},
		# 											  {"out_x": shift(bottom_x), "out_y": shift(bottom_y),
		# 											   "out_p": shift(bottom_p)}))
		print('{} dataset created'.format(name))
		# del top_x, top_y, top_p
		# del bottom_p, bottom_y, bottom_x

		# dataset = dataset.batch(CAP_SEQ)
		# dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
		steps_per_epoch = NUM_DRAWINGS // BATCH_SIZE
		# while True:
		EPOCHS = initial_epoch + 1
		model.fit(iter, epochs=EPOCHS, initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback, InitIteratorCallback()])

		#	GENERATOR

		for d in range(5):
			r = random.randint(0, (((NUM_DRAWINGS//CAP_SEQ)// 5) * (d + 1)) - 1)
			input_x = np.expand_dims(top_x[CAP_SEQ * r: CAP_SEQ * (r+1)], 0)
			input_y = np.expand_dims(top_y[CAP_SEQ * r: CAP_SEQ * (r+1)], 0)
			input_p = np.expand_dims(top_p[CAP_SEQ * r: CAP_SEQ * (r+1)], 0)
			states_values = encoder_model.predict(({'in_top_x': input_x, 'in_top_y': input_y, 'in_top_p': input_p}))
			target_seq_x = np.zeros((1, 1, 1))
			target_seq_y = np.zeros((1, 1, 1))
			target_seq_p = np.zeros((1, 1, 1))
			target_seq_p[0, 0, 0] = 3;
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
				image_name = os.path.join(SAMPLE_DIR, 'sample_{}_epoch{}_draw{}'.format(name.replace(' ', '_'),
																						initial_epoch, d))
				top_bottom_to_image(image_name, np.squeeze(input_x), np.squeeze(input_y), np.squeeze(input_p),
									pred_x, pred_y, pred_p)

		initial_epoch += 1
		f += 1
	full += 1