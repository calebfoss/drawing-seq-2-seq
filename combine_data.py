import os
import time
import numpy as np
from glob import glob
EXP_DIR = 'C:/Users/caleb/Documents/Tensorflow/expanded_data'
OUT_DIR = 'C:/Users/caleb/Documents/Tensorflow/combined_data'
IN_NUM = 200
OUT_NUM = 345

if not os.path.exists(OUT_DIR):
	os.mkdir(OUT_DIR)

files = glob(os.path.join(EXP_DIR, '*.npz'))

start_time = time.time()
for f in range(5, OUT_NUM):
	top_x = np.empty((0, 1), dtype=np.uint16)
	top_y = np.empty((0, 1), dtype=np.uint16)
	top_p = np.empty((0, 1), dtype=np.uint8)
	bottom_x = np.empty((0, 1), dtype=np.uint16)
	bottom_y = np.empty((0, 1), dtype=np.uint16)
	bottom_p = np.empty((0, 1), dtype=np.uint8)
	for file in files[f::IN_NUM]:
		data = np.load(file)
		top_x = np.append(top_x, data['top_x'])
		top_y = np.append(top_y, data['top_y'])
		top_p = np.append(top_p, data['top_p'])
		bottom_x = np.append(bottom_x, data['bottom_x'])
		bottom_y = np.append(bottom_y, data['bottom_y'])
		bottom_p = np.append(bottom_p, data['bottom_p'])
	np.savez_compressed(os.path.join(OUT_DIR, 'data_{}'.format(str(f).zfill(3))), top_x=top_x, top_y=top_y, top_p=top_p, bottom_x=bottom_x, bottom_y=bottom_y, bottom_p=bottom_p)
	passed = time.time() - start_time
	eta = (passed / (f + 1)) * (OUT_NUM - (f + 1))
	hr = int(eta // (60 * 60))
	mn = int((eta // 60) % 60)
	sc = int(mn % 60)
	print('{}/{} Eta: {}h{}m{}s'.format(f, OUT_NUM, hr, mn, sc))