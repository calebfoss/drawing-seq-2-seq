import tensorflow as tf
from tensorflow import keras
import numpy as np
from utils import*
layers = keras.layers


NUM_DRAWINGS = np.inf
CAP_SEQ = 60
top_x, top_y, top_p, bottom_x, bottom_y, bottom_p = file_to_split('../DrawingRNN/data/apple.ndjson', NUM_DRAWINGS, CAP_SEQ)
NUM_DRAWINGS = len(top_x)
# print(top_x, top_y, top_p, bottom_x, bottom_y, bottom_p)
# top_bottom_to_image('pad_test', top_x, top_y, top_p, bottom_x, bottom_y, bottom_p)
# exit()

dataset = tf.data.Dataset.from_tensor_slices(({"input_1":top_x, "input_2":top_y, "input_3":top_p},
											  {"dense_1":bottom_x,"dense_2":bottom_y,"dense_3":bottom_p}))
print('Dataset created')
# del data_in_x, data_in_p, data_in_y, data_out_p, data_out_y, data_out_x
# del top_x, top_y, top_p
del bottom_p, bottom_y, bottom_x

dataset = dataset.batch(CAP_SEQ)
print('Dataset separted into sequences')

# if tf.test.is_gpu_available():
#     rnn = keras.layers.CuDNNLSTM
# else:
#     import functools
#     rnn = functools.partial(
#         keras.layers.LSTM, recurrent_activation='sigmoid')

RNN_UNITS = 64
BUFFER_SIZE = 1000
BATCH_SIZE = 2048
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
print('Dataset shuffled into batches')
EPOCHS = 10
steps_per_epoch = NUM_DRAWINGS//BATCH_SIZE
E_DIM = 8

def loss(labels, logits):
	mask = keras.backend.all(keras.backend.not_equal(labels, 0), axis=-1)
	mask = keras.backend.cast(mask, dtype='float32')
	result = mask * tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
	return result

in_x = layers.Input(shape=(CAP_SEQ, 1,), batch_size=BATCH_SIZE, dtype='uint16')
in_y = layers.Input(shape=(CAP_SEQ, 1,), batch_size=BATCH_SIZE, dtype='uint16')
in_pen = layers.Input(shape=(CAP_SEQ, 1,), batch_size=BATCH_SIZE, dtype='uint8')
emb_x = layers.Embedding(input_dim=512, output_dim=E_DIM, input_length=CAP_SEQ)(in_x)
emb_y = layers.Embedding(input_dim=512, output_dim=E_DIM, input_length=CAP_SEQ)(in_y)
emb_pen = layers.Embedding(input_dim=3, output_dim=E_DIM)(in_pen)
merged_layer = layers.concatenate([emb_x, emb_y, emb_pen], axis=-1)
flat_merged = layers.Reshape((CAP_SEQ,E_DIM*3))(merged_layer)
encoder = layers.CuDNNLSTM(RNN_UNITS, return_sequences=True, batch_size=BATCH_SIZE)(flat_merged)
decoder = layers.CuDNNLSTM(RNN_UNITS, return_sequences=True, batch_size=BATCH_SIZE)(encoder)
predictions = layers.Dense(256, activation='relu')(decoder)
out_x = layers.Dense(512)(predictions)
out_y = layers.Dense(512)(predictions)
out_p = layers.Dense(3)(predictions)
model = keras.Model(inputs=[in_x, in_y, in_pen], outputs=[out_x, out_y, out_p])


model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
model.summary()

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

if not os.path.isdir(checkpoint_dir):
	os.mkdir(checkpoint_dir)
elif os.path.exists(os.path.join(checkpoint_dir, 'checkpoint')):
    latest_chkpt = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest_chkpt)
    print('Restored model from {}'.format(latest_chkpt))
# input_x = np.expand_dims(top_x[:CAP_SEQ], 0)
# input_y = np.expand_dims(top_y[:CAP_SEQ], 0)
# input_p = np.expand_dims(top_p[:CAP_SEQ], 0)
# pred = model.predict(({'input_1':input_x, 'input_2':input_y, 'input_3':input_p}),steps=1)
# pred_x = np.argmax(pred[0], axis=2)
# pred_y = np.argmax(pred[1], axis=2)
# pred_p = np.argmax(pred[2], axis=2)
# print(pred_x, pred_y, pred_p)
# top_bottom_to_image('first_prediction', top_x[:CAP_SEQ], top_y[:CAP_SEQ], top_p[:CAP_SEQ], pred_x, pred_y, pred_p)
# exit()


checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)
model.fit(dataset.repeat(), epochs=EPOCHS, steps_per_epoch=steps_per_epoch,callbacks=[checkpoint_callback])

