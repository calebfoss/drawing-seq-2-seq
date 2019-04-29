import tensorflow as tf
from tensorflow import keras
import numpy as np
from utils import*
layers = keras.layers

# tf.enable_eager_execution()

NUM_DRAWINGS = 1
CAP_SEQ = 60
top_x, top_y, top_p, bottom_x, bottom_y, bottom_p = file_to_split('../DrawingRNN/data/apple.ndjson', NUM_DRAWINGS, CAP_SEQ)
NUM_DRAWINGS = len(top_x)

# dataset = tf.data.Dataset.from_tensor_slices(({"input_1": top_x, "input_2": top_y, "input_3": top_p},
# 											  {"out_x":bottom_x,"out_y":bottom_y,"out_p":bottom_p}))

RNN_UNITS = 64
BUFFER_SIZE = 1000
BATCH_SIZE = 1
# dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
print('Dataset shuffled into batches')
EPOCHS = 10
steps_per_epoch = NUM_DRAWINGS//BATCH_SIZE
E_DIM = 8

checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
# assert(os.path.exists(os.path.join(checkpoint_dir, 'checkpoint')))




def loss(labels, logits):
	mask = keras.backend.all(keras.backend.not_equal(labels, 0), axis=-1)
	mask = keras.backend.cast(mask, dtype='float32')
	result = mask * tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
	return result

def squeeze(x):
	 result = keras.backend.squeeze(x, 2)
	 return result

def loss(labels, logits):
	mask = keras.backend.all(keras.backend.not_equal(labels, 0), axis=-1)
	mask = keras.backend.cast(mask, dtype='float32')
	result = mask * tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
	return result

en_in_x = layers.Input(shape=(None, 1,), batch_size=BATCH_SIZE, dtype='uint16')
en_in_y = layers.Input(shape=(None, 1,), batch_size=BATCH_SIZE, dtype='uint16')
en_in_pen = layers.Input(shape=(None, 1,), batch_size=BATCH_SIZE, dtype='uint8')
en_emb_x = layers.Embedding(input_dim=512, output_dim=E_DIM, input_length=CAP_SEQ)(en_in_x)
en_emb_y = layers.Embedding(input_dim=512, output_dim=E_DIM, input_length=CAP_SEQ)(en_in_y)
en_emb_pen = layers.Embedding(input_dim=3, output_dim=E_DIM)(en_in_pen)
en_sq_pen = layers.Lambda(squeeze)(en_emb_pen)
en_merged_layer = layers.concatenate([en_emb_x, en_emb_y, en_emb_pen], axis=-1)
en_flat_merged = layers.Reshape((CAP_SEQ,E_DIM*3))(en_merged_layer)

encoder = layers.CuDNNLSTM(RNN_UNITS, return_state=True, batch_size=BATCH_SIZE)
encoder_outputs, state_h, state_c = encoder(en_flat_merged)
encoder_states = [state_h, state_c]

de_in_x = layers.Input(shape=(CAP_SEQ, 1,), batch_size=BATCH_SIZE, dtype='uint16')
de_in_y = layers.Input(shape=(CAP_SEQ, 1,), batch_size=BATCH_SIZE, dtype='uint16')
de_in_pen = layers.Input(shape=(CAP_SEQ, 1,), batch_size=BATCH_SIZE, dtype='uint8')
de_emb_x = layers.Embedding(input_dim=512, output_dim=E_DIM, input_length=CAP_SEQ)(de_in_x)
de_emb_y = layers.Embedding(input_dim=512, output_dim=E_DIM, input_length=CAP_SEQ)(de_in_y)
de_emb_pen = layers.Embedding(input_dim=5, output_dim=E_DIM)(de_in_pen)
de_sq_pen = layers.Lambda(squeeze)(de_emb_pen)
de_merged_layer = layers.concatenate([de_emb_x, de_emb_y, de_emb_pen], axis=-1)
de_flat_merged = layers.Reshape((CAP_SEQ,E_DIM*3))(de_merged_layer)

decoder_lstm = layers.CuDNNLSTM(RNN_UNITS, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(de_flat_merged, initial_state=encoder_states)
# decoder_dense = layers.Dense(512 + 512 + 3)
# decoder_outputs = decoder_dense(decoder_outputs)
decoder_x = layers.Dense(512, activation='softmax', name='out_x')
decoder_y = layers.Dense(512, activation='softmax', name='out_y')
decoder_p = layers.Dense(5, activation='softmax', name='out_p')
out_x = decoder_x(decoder_outputs)
out_y = decoder_y(decoder_outputs)
out_p = decoder_p(decoder_outputs)
model = keras.Model(inputs=[en_in_x, en_in_y, en_in_pen, de_in_x, de_in_y, de_in_pen], outputs=[out_x, out_y, out_p])


model.compile(loss=loss, optimizer=tf.train.AdamOptimizer(0.001))

latest_chkpt = tf.train.latest_checkpoint(checkpoint_dir)

model.load_weights(latest_chkpt)
print('Restored model from {}'.format(latest_chkpt))

encoder_model = keras.Model(inputs=[en_in_x, en_in_y, en_in_pen], outputs=encoder_states)

decoder_state_input_h = layers.Input(shape=(RNN_UNITS,))
decoder_state_input_c = layers.Input(shape=(RNN_UNITS,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(de_flat_merged, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
# decoder_outputs = decoder_dense(decoder_outputs)
de_out_x = decoder_x(decoder_outputs)
de_out_y = decoder_y(decoder_outputs)
de_out_p = decoder_p(decoder_outputs)
decoder_model = keras.Model([de_in_x, de_in_y, de_in_pen] + decoder_states_inputs,
							[de_out_x, de_out_y, de_out_p] + decoder_states)





input_x = np.expand_dims(top_x[:CAP_SEQ], 0)
input_y = np.expand_dims(top_y[:CAP_SEQ], 0)
input_p = np.expand_dims(top_p[:CAP_SEQ], 0)

print(bottom_x[:CAP_SEQ], bottom_y[:CAP_SEQ], bottom_p[:CAP_SEQ])

encoder_model.compile(loss=loss, optimizer='adam')



states_values = encoder_model.predict(({'input_1': input_x, 'input_2': input_y, 'input_3': input_p}))
target_seq_x = np.zeros((1, CAP_SEQ, 1,))
target_seq_y = np.zeros((1, CAP_SEQ, 1,))
target_seq_p = np.zeros((1, CAP_SEQ, 1,))
target_seq_p[0, 0, 0] = 3;
pred_x = np.zeros(CAP_SEQ)
pred_y = np.zeros(CAP_SEQ)
pred_p = np.zeros(CAP_SEQ)


for i in range(CAP_SEQ):
	print(target_seq_x[0, 0, 0],target_seq_y[0, 0, 0],target_seq_p[0, 0, 0])
	output_tokens_x, output_tokens_y, output_tokens_p, h, c = decoder_model.predict([target_seq_x, target_seq_y, target_seq_p] + states_values)
	pred_x[i] = np.argmax(output_tokens_x[0, -1, :])
	pred_y[i] = np.argmax(output_tokens_y[0, -1, :])
	pred_p[i] = np.argmax(output_tokens_p[0, -1, :])
	if i + 1 < CAP_SEQ:
		target_seq_x[0, 0, 0] = pred_x[i]
		target_seq_y[0, 0, 0] = pred_y[i]
		target_seq_p[0, 0, 0] = pred_p[i]
	states_values = [h, c]

# pred = encoder_model.predict((dataset.take(1)),steps=1)
# pred_x = np.argmax(pred[0], axis=2)
# pred_y = np.argmax(pred[1], axis=2)
# pred_p = np.argmax(pred[2], axis=2)
print(pred_x, pred_y, pred_p)
top_bottom_to_image('second_after_tutorial', top_x[:CAP_SEQ], top_y[:CAP_SEQ], top_p[:CAP_SEQ], pred_x, pred_y, pred_p)
