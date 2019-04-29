
import tensorflow as tf
from utils import *
from glob import glob
import time
import psutil
DATA_DIR = './data'
OUTPUT_DIR = 'C:/Users/caleb/Documents/Tensorflow/expanded_data'
SAMPLE_DIR = './samples'
START_NUM = 0
OUTPUT_NUM = 200
CAP_SEQ = 60

def shift(a):
	result = np.expand_dims(np.append(np.delete(a, 0), 0), 1)
	result[CAP_SEQ - 1::CAP_SEQ] = 0
	return result.astype(a.dtype)

# files = glob(os.path.join(DATA_DIR, '*.ndjson'))
# for f in range(len(files)):
# 	top_x, top_y, top_p, bottom_x, bottom_y, bottom_p = file_to_split(files[f], CAP_SEQ)
# 	if f > 0:
# 		new_ds = tf.data.Dataset.from_tensor_slices(({'in_top_x': top_x, 'in_top_y': top_y, 'in_top_p': top_p,
# 													   "in_bottom_x": bottom_x, "in_bottom_y": bottom_y,
# 													   "in_bottom_p": bottom_p},
# 													  {"out_x": shift(bottom_x), "out_y": shift(bottom_y),
# 													   "out_p": shift(bottom_p)}))
# 		dataset = dataset.concatenate(new_ds)
# 	else:
# 		dataset = tf.data.Dataset.from_tensor_slices(({'in_top_x': top_x, 'in_top_y': top_y, 'in_top_p': top_p,
# 													   "in_bottom_x": bottom_x, "in_bottom_y": bottom_y,
# 													   "in_bottom_p": bottom_p},
# 													  {"out_x": shift(bottom_x), "out_y": shift(bottom_y),
# 													   "out_p": shift(bottom_p)}))
# exit()
# def generator():
# 	for file in files:
# 		top_x, top_y, top_p, bottom_x, bottom_y, bottom_p = file_to_split(file, CAP_SEQ)
# 		yield top_x, top_y, top_p, bottom_x, bottom_y, bottom_p
#
# dataset = tf.data.Dataset.from_generator(generator, output_types=(tf.uint16, tf.uint16, tf.uint8, tf.uint16, tf.uint16, tf.uint8))
# dataset.batch(CAP_SEQ)
# iter = dataset.make_initializable_iterator()
# el = iter.get_next()
# with tf.Session() as sess:
# 	sess.run(iter.initializer)
# 	print(sess.run(el))
# 	print(sess.run(el))
# exit()
#
#
# feature_description = {
# 		'in_top_x': tf.FixedLenFeature([], tf.int64, default_value=0),
# 		'in_top_y': tf.FixedLenFeature([], tf.int64, default_value=0),
# 		'in_top_p': tf.FixedLenFeature([], tf.int64, default_value=0),
# 		'in_bottom_x': tf.FixedLenFeature([], tf.int64, default_value=0),
# 		'in_bottom_y': tf.FixedLenFeature([], tf.int64, default_value=0),
# 		'in_bottom_p': tf.FixedLenFeature([], tf.int64, default_value=0),
# 		'out_x': tf.FixedLenFeature([], tf.int64, default_value=0),
# 		'out_y': tf.FixedLenFeature([], tf.int64, default_value=0),
# 		'out_p': tf.FixedLenFeature([], tf.int64, default_value=0)
# }
# def _parse_function(example_proto):
# 	print(example_proto)
# 	# Parse the input tf.Example proto using the dictionary above.
# 	return tf.parse_single_example(example_proto, feature_description)
#
# files = glob(os.path.join('./records', '*.tfrecord'))
# dataset = tf.data.TFRecordDataset(files)
# for record in dataset.take(1):
# 	print(repr(record))
# exit()
# dataset = dataset.map(_parse_function)


# dataset = dataset.batch(CAP_SEQ)

# out_files = glob(os.path.join(OUTPUT_DIR, '*.npz'))
# out_len = len(out_files)
# for f in range(out_len):
# 	name = os.path.join('./data_squeezed', 'dataset_' + str(f).zfill(3))
# 	data = np.load(out_files[f])
# 	top_x = np.squeeze(data['top_x'])
# 	top_y = np.squeeze(data['top_y'])
# 	top_p = np.squeeze(data['top_p'])
# 	bottom_x = np.squeeze(data['bottom_x'])
# 	bottom_y = np.squeeze(data['bottom_y'])
# 	bottom_p = np.squeeze(data['bottom_p'])
# 	np.savez(name, top_x=top_x, top_y=top_y, top_p=top_p, bottom_x=bottom_x, bottom_y=bottom_y,
# 							bottom_p=bottom_p)
# 
# exit()

# files = glob(os.path.join(DATA_DIR, '*.ndjson'))



# def _int64_feature(value):
#   """Returns an int64_list from a bool / enum / int / uint."""
#   value = np.squeeze(value).tolist()
#   return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
#
# def serialize_example(f0, f1, f2, f3, f4, f5, f6, f7, f8):
# 	feature = {
# 		'in_top_x': _int64_feature(f0),
# 		'in_top_y': _int64_feature(f1),
# 		'in_top_p': _int64_feature(f2),
# 		'in_bottom_x': _int64_feature(f3),
# 		'in_bottom_y': _int64_feature(f4),
# 		'in_bottom_p': _int64_feature(f5),
# 		'out_x': _int64_feature(f6),
# 		'out_y': _int64_feature(f7),
# 		'out_p': _int64_feature(f8)
# 	}
# 	example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
# 	return example_proto.SerializeToString()
#
#
# for f in range(len(files)):
# 	category = os.path.basename(files[f]).split('.')[0].replace(' ', '_')
# 	print(category)
# 	name = os.path.join(OUTPUT_DIR, category + '.tfrecord')
# 	top_x, top_y, top_p, bottom_x, bottom_y, bottom_p = file_to_split(files[f], cap_seq=60, inc=100)
# 	out_x = shift(bottom_x)
# 	out_y = shift(bottom_y)
# 	out_p = shift(bottom_p)
# 	ex = serialize_example(top_x, top_y, top_p, bottom_x, bottom_y, bottom_p, out_x, out_y, out_p)
# 	with tf.python_io.TFRecordWriter(name) as writer:
# 		writer.write(ex)



# def get_drawings(path):
# 	file = tf.read_file(path)
# 	top_x, top_y, top_p, bottom_x, bottom_y, bottom_p = file_to_split(file, cap_seq=60)
# 	return {
# 		'in_top_x': top_x,
# 		'in_top_y': top_y,
# 		'in_top_p': top_p,
# 		'in_bottom_x': bottom_x,
# 		'in_bottom_y': bottom_y,
# 		'in_bottom_p': bottom_p,
# 		'out_x': shift(top_x),
# 		'out_y': shift(top_y),
# 		'out_p': shift(top_p),
# 	}


#
#

#
# if not os.path.exists(OUTPUT_DIR):
# 	os.mkdir(OUTPUT_DIR)
#
#
#

#
#

# 	# Create a Features message using tf.train.Example.
#
# 	example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
# 	return example_proto.SerializeToString()
#
# def tf_serialize_example(f0, f1, f2, f3, f4, f5, f6, f7, f8):
# 	tf_string = tf.py_func(
# 		serialize_example,
# 		(f0, f1, f2, f3, f4, f5, f6, f7, f8),  # pass these args to the above function.
# 		tf.string)  # the return type is <a href="../../api_docs/python/tf#string"><code>tf.string</code></a>.
# 	return tf.reshape(tf_string, ())  # The result is a scalar
#
# for f in range(START_NUM, files_len):
# 	file = files[f]
# 	category = os.path.basename(file).split('.')[0].replace(' ', '_')
# 	print(category)
# 	name = os.path.join(OUTPUT_DIR, category + '.tfrecord')
#
# 	# f_top_x, f_top_y, f_top_p, f_bottom_x, f_bottom_y, f_bottom_p = file_to_split(file, cap_seq=60)
# 	top_x, top_y, top_p, bottom_x, bottom_y, bottom_p = file_to_split(file, cap_seq=CAP_SEQ, inc=1000)
# 	# dataset = tf.data.Dataset.from_tensor_slices(({'in_top_x': top_x, 'in_top_y': top_y, 'in_top_p': top_p,
# 	# 												   "in_bottom_x": bottom_x, "in_bottom_y": bottom_y,
# 	# 												   "in_bottom_p": bottom_p},
# 	# 												  {"out_x": shift(bottom_x), "out_y": shift(bottom_y),
# 	# 												   "out_p": shift(bottom_p)}))
# 	dataset = tf.data.Dataset.from_tensor_slices((top_x, top_y, top_p, bottom_x, bottom_y, bottom_p, shift(bottom_x), shift(bottom_y),
# 													   shift(bottom_p)))
# 	# dataset.batch(CAP_SEQ)
# 	dataset = dataset.map(tf_serialize_example)
# 	open(name, 'a').close()
# 	writer = tf.data.experimental.TFRecordWriter(name)
# 	writer.write(dataset)
if not os.path.exists(OUTPUT_DIR):
	os.mkdir(OUTPUT_DIR)
files = glob(os.path.join(DATA_DIR, '*.ndjson'))
files_len = len(files)
for f in range(52, files_len):
	file = files[f]
	category = os.path.basename(file).split('.')[0].replace(' ', '_')
	start_time = time.time()
	top_x, top_y, top_p, bottom_x, bottom_y, bottom_p = file_to_split(file, cap_seq=60)
	f_len = len(top_x)
	print('File {} of {}'.format(f, files_len))
	for i in range(OUTPUT_NUM):
		inc = ((f_len // 60) // OUTPUT_NUM) * 60
		b = inc * i
		e = b + inc
		if i % (OUTPUT_NUM // 20) == 0:
			print('\tSaving {}/{}'.format(i, OUTPUT_NUM))
		# name = os.path.join(OUTPUT_DIR, 'dataset_{}'.format(str(i).zfill(3)))

		name = os.path.join(OUTPUT_DIR, 'dataset_{}_{}'.format(str(f).zfill(3), str(i).zfill(3)))
		np.savez(name, top_x=top_x[b:e], top_y=top_y[b:e], top_p=top_p[b:e], bottom_x=bottom_x[b:e], bottom_y=bottom_y[b:e],
				 bottom_p=bottom_p[b:e])
		# if f > 0:
		# 	data = np.load(name+'.npz')
		# 	top_x = np.vstack((data['top_x'], f_top_x[b:e]))
		# 	top_y = np.vstack((data['top_y'], f_top_y[b:e]))
		# 	top_p = np.vstack((data['top_p'], f_top_p[b:e]))
		# 	bottom_x = np.vstack((data['bottom_x'], f_bottom_x[b:e]))
		# 	bottom_y = np.vstack((data['bottom_y'], f_bottom_y[b:e]))
		# 	bottom_p = np.vstack((data['bottom_p'], f_bottom_p[b:e]))
		# # 	top_x = np.append(data['top_x'], f_top_x[b:e])
		# # 	top_y = np.append(data['top_y'], f_top_y[b:e])
		# # 	top_p = np.append(data['top_p'], f_top_p[b:e])
		# # 	bottom_x = np.append(data['bottom_x'], f_bottom_x[b:e])
		# # 	bottom_y = np.append(data['bottom_y'], f_bottom_y[b:e])
		# # 	bottom_p = np.append(data['bottom_p'], f_bottom_p[b:e])
		# else:
		# top_x = f_top_x[b:e]
		# top_y = f_top_y[b:e]
		# top_p = f_top_p[b:e]
		# bottom_x = f_bottom_x[b:e]
		# bottom_y = f_bottom_y[b:e]
		# bottom_p = f_bottom_p[b:e]
			# top_x = np.squeeze(f_top_x)
			# top_y = np.squeeze(f_top_y)
			# top_p = np.squeeze(f_top_p)
			# bottom_x = np.squeeze(f_bottom_x)
			# bottom_y = np.squeeze(f_bottom_y)
			# bottom_p = np.squeeze(f_bottom_p)

		# print('\t', len(top_x), len(top_y), len(top_p), len(bottom_x), len(bottom_y), len(bottom_p))
		# print('\tbegin: {} end: {}'.format(b, e))
		# np.savez(name, top_x=top_x, top_y=top_y, top_p=top_p, bottom_x=bottom_x, bottom_y=bottom_y,
		# 					bottom_p=bottom_p)

# 	# eta = int((time.time() - start_time) * (files_len - f - 1))
# 	# hr = eta // (60 * 60)
# 	# mn = (eta // 60) % 60
# 	# sc = eta % 60
# 	# print('ETA: {}:{}:{}'.format(hr, mn, sc))
# # total_len = len(top_x)
# # print('Total: {}'.format(total_len))
# # inc = total_len // files_len
# # sec = inc // OUTPUT_NUM
# # for x in range(OUTPUT_NUM):
# # 	out_top_x = np.empty((0, 1), dtype=np.uint16)
# # 	out_top_y = np.empty((0, 1), dtype=np.uint16)
# # 	out_top_p = np.empty((0, 1), dtype=np.uint8)
# # 	out_bottom_x = np.empty((0, 1), dtype=np.uint16)
# # 	out_bottom_y = np.empty((0, 1), dtype=np.uint16)
# # 	out_bottom_p = np.empty((0, 1), dtype=np.uint8)
# # 	for y in range(files_len):
# # 		b = (inc * y) + (sec * x)
# # 		e = b + sec + 1
# # 		print('b: {} e: {} emts: {}'.format(b, e, e - b))
# # 		out_top_x = np.vstack((out_top_x, top_x[b:e]))
# # 		out_top_y = np.vstack((out_top_y, top_y[b:e]))
# # 		out_top_p = np.vstack((out_top_p, top_p[b:e]))
# # 		out_bottom_x = np.vstack((out_bottom_x, bottom_x[b:e]))
# # 		out_bottom_y = np.vstack((out_bottom_y, bottom_y[b:e]))
# # 		out_bottom_p = np.vstack((out_bottom_p, bottom_p[b:e]))
# # 		# print('Preparing dataset {} of {}'.format(x, OUTPUT_NUM))
# # 	name = os.path.join(OUTPUT_DIR, 'dataset_' + str(x).zfill(3))
# # 	np.savez_compressed(name, top_x=out_top_x, top_y=out_top_y, top_p=out_top_p, bottom_x=out_bottom_x,
# # 						bottom_y=out_bottom_y, bottom_p=out_bottom_p)
# # 	print('Saved as {}'.format(name))
#
# #
