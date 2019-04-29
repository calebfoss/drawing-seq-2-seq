import os
import numpy as np
from PIL import Image, ImageDraw
import ndjson
from bresenham import bresenham
import svgwrite



def top_bottom_to_image(name, top_x, top_y, top_p, bottom_x, bottom_y, bottom_p):
	top_x = np.squeeze(top_x)
	top_y = np.squeeze(top_y)
	top_p = np.squeeze(top_p)
	bottom_x = np.squeeze(bottom_x)
	bottom_y = np.squeeze(bottom_y)
	bottom_p = np.squeeze(bottom_p)
	# svg = svgwrite.Drawing('{}.svg'.format(name), size=(256, 256))
	# svg.add(svg.rect(insert=(0,0), size=('100%', '100%'), fill=255))
	img_top = Image.new('L', (256, 256), 255)
	img_bottom = Image.new('L', (256, 256), 255)
	img_combined = Image.new('L', (256, 256), 255)
	draw_top = ImageDraw.Draw(img_top, 'L')
	draw_bottom = ImageDraw.Draw(img_bottom, 'L')
	draw_combined = ImageDraw.Draw(img_combined, 'L')
	y_top_val = 0
	top_len = len(top_x)
	prev_top = [0, 0]
	for i in range(1, top_len):
		if top_p[i-1] < 4:
			x = prev_top[0] + (top_x[i] - 256)
			y = prev_top[1] + (top_y[i] - 256)
			# print('\tTop coordinates: {},{}'.format(x, y))
			if top_p[i-1] == 1:
				draw_top.line(((prev_top[0], prev_top[1]), (x, y)), fill=0, width=1)
				draw_combined.line(((prev_top[0], prev_top[1]), (x, y)), fill=0, width=1)
				# svg.add(svg.line(((prev_top[0], prev_top[1]), (x, y)), stroke=0, stroke_width=1))
			prev_top = [x, y]
			y_top_val = max(y_top_val, y + 1)
		else:
			# print('Saved top as {}_top.png'.format(name))
			break
	img_top.save('{}_top.png'.format(name))
	bottom_len = len(bottom_x)
	prev = [bottom_x[0] - 256, bottom_y[0] - 256]
	min_x = 0
	min_y = 0
	for i in range(1, bottom_len):
		if bottom_p[i - 1] < 4:
			x = prev[0] + (bottom_x[i] - 256)
			y = prev[1] + (bottom_y[i] - 256)
			prev = [x, y]
			min_x = min(min_x, x)
			min_y = min(min_y, y)
		else:
			break
	prev_bottom = [bottom_x[0] - 256 - min_x, bottom_y[0] - 256 + y_top_val - min_y]
	for i in range(1, bottom_len):
		if bottom_p[i - 1] < 4:
			x = prev_bottom[0] + (bottom_x[i] - 256)
			y = prev_bottom[1] + (bottom_y[i] - 256)
			# print('\tBottom coordinates: {},{}'.format(x, y))
			# print(bottom_x[i], bottom_y[i], x, y)
			if bottom_p[i - 1] == 1:
				draw_bottom.line(((prev_bottom[0], prev_bottom[1]), (x, y)), fill=0, width=1)
				draw_combined.line(((prev_bottom[0], prev_bottom[1]), (x, y)), fill=0, width=1)
			prev_bottom = [x, y]
		else:
			break
	img_bottom.save('{}_bottom.png'.format(name))
	print('Saved bottom as {}_bottom.png'.format(name))
	img_combined.save('{}_combined.png'.format(name))
	print('Saved combined as {}_combined.png'.format(name))
	# svg.save()

def append_point(pen, stroke_x, stroke_y, prev):
	x_move = stroke_x - prev[0]
	y_move = stroke_y - prev[1]
	prev = [prev[0] + x_move, prev[1] + y_move]
	return prev, x_move + 256, y_move + 256, pen

def connect_points(y_target, pen, stroke_x, stroke_y, prev_1, prev_2):
	btw = list(bresenham(prev_1[0], prev_1[1], stroke_x, stroke_y))
	btw_len = len(btw)
	for p in range(btw_len - 1):
		if btw[p][1] == y_target:
			x_move_1 = btw[p][0] - prev_1[0]
			y_move_1 = btw[p][1] - prev_1[1]
			prev_1 = btw[p]
			x_move_2 = btw[p+1][0] - prev_2[0]
			y_move_2 = btw[p+1][1] - prev_2[1]
			x_move_3 = stroke_x - btw[p+1][0]
			y_move_3 = stroke_y - btw[p+1][1]
			prev_2 = [prev_2[0] + x_move_2 + x_move_3, prev_2[1] + y_move_2 + y_move_3]
			return prev_1, prev_2, [x_move_1 + 256, y_move_1 + 256, 2, x_move_2 + 256, y_move_2 + 256, 1, x_move_3 + 256, y_move_3 + 256, pen]
	print(y_target, pen, stroke_x, stroke_y, prev_1, prev_2)
	print('FAILED TO CONNECT POINTS')

def file_to_split(file, cap_seq, inc=1, start=0):
	print('Loading {}'.format(os.path.basename(file)))
	with open(file) as fs:
		data = ndjson.load(fs)
	inc = len(data) // inc
	start = start * inc
	data_end = min(start + inc, len(data))
	top_x = np.zeros((data_end * cap_seq, 1), dtype=np.uint16)
	top_y = np.zeros((data_end * cap_seq, 1), dtype=np.uint16)
	top_p = np.zeros((data_end * cap_seq, 1), dtype=np.uint8)
	bottom_x = np.zeros((data_end * cap_seq, 1), dtype=np.uint16)
	bottom_y = np.zeros((data_end * cap_seq, 1), dtype=np.uint16)
	bottom_p = np.zeros((data_end * cap_seq, 1), dtype=np.uint8)
	for i in range(start, data_end):
			if i > 0 and i % ((data_end-start) // 5) == 0:
				print('\tDrawing {}/{}'.format(i, data_end))
			drawing_in = data[i].get('drawing')
			draw_len = len(drawing_in)
			y_max = 0
			y_min = np.inf
			for s in range(draw_len):
				y_values = drawing_in[s][1]
				stroke_len = len(y_values)
				for p in range(stroke_len):
					y_max = max(y_max, y_values[p])
					y_min = min(y_min, y_values[p])
			y_top = max(y_max // 4, y_min + 1)
			prev_top = [0, 0]
			prev_bottom = [0, y_top + 1]
			last_top = True
			top_p[i*cap_seq] = 3
			bottom_p[i*cap_seq] = 3
			top_idx = i * cap_seq + 1
			bottom_idx = i * cap_seq + 1
			for s in range(draw_len):
				stroke = drawing_in[s]
				stroke_len = len(stroke[0])
				for p in range(stroke_len):
					if p < stroke_len - 1:
						pen = 1
					elif s < draw_len - 1:
						pen = 2
					else:
						pen = 4
					if (stroke[1][p] <= y_top) == last_top:
						if last_top:
							if top_idx < (i + 1) * cap_seq:
								prev_top, top_x[top_idx], top_y[top_idx], top_p[top_idx] = \
									append_point(pen, stroke[0][p], stroke[1][p], prev_top)
								top_idx += 1
						else:
							if bottom_idx < (i + 1) * cap_seq:
								prev_bottom, bottom_x[bottom_idx], bottom_y[bottom_idx], bottom_p[bottom_idx] = \
									append_point(pen, stroke[0][p], stroke[1][p], prev_bottom)
								# print(1, bottom_x[bottom_idx], bottom_y[bottom_idx])
								bottom_idx += 1
					else:
						if p == 0:
							if last_top:
								if bottom_idx < (i + 1) * cap_seq:
									prev_bottom, bottom_x[bottom_idx], bottom_y[bottom_idx], bottom_p[bottom_idx] = \
										append_point(pen, stroke[0][p], stroke[1][p], prev_bottom)
									# print(2, bottom_x[bottom_idx], bottom_y[bottom_idx])
									bottom_idx += 1
							else:
								if top_idx < (i + 1) * cap_seq:
									prev_top, top_x[top_idx], top_y[top_idx], top_p[top_idx] = \
										append_point(pen, stroke[0][p], stroke[1][p], prev_top)
									top_idx += 1
						else:
							if last_top:
								prev_top, prev_bottom, result = \
									connect_points(y_top, pen, stroke[0][p], stroke[1][p], prev_top, prev_bottom)
								if top_idx < (i + 1) * cap_seq:
									top_x[top_idx] = result[0]
									top_y[top_idx] = result[1]
									top_p[top_idx] = result[2]
									top_idx += 1
								if bottom_idx < (i + 1) * cap_seq:
									bottom_x[bottom_idx] = result[3]
									bottom_y[bottom_idx] = result[4]
									bottom_p[bottom_idx] = result[5]
									# print(3, bottom_x[bottom_idx], bottom_y[bottom_idx])
									bottom_idx += 1
								if bottom_idx < (i + 1) * cap_seq:
									bottom_x[bottom_idx] = result[6]
									bottom_y[bottom_idx] = result[7]
									bottom_p[bottom_idx] = result[8]
									bottom_idx += 1
							else:
								prev_bottom, prev_top, result = \
									connect_points(y_top + 1, pen, stroke[0][p], stroke[1][p], prev_bottom, prev_top)
								if bottom_idx < (i + 1) * cap_seq:
									bottom_x[bottom_idx] = result[0]
									bottom_y[bottom_idx] = result[1]
									bottom_p[bottom_idx] = result[2]
									# print(4, bottom_x[bottom_idx], bottom_y[bottom_idx])
									bottom_idx += 1
								if top_idx < (i + 1) * cap_seq:
									top_x[top_idx] = result[3]
									top_y[top_idx] = result[4]
									top_p[top_idx] = result[5]
									top_idx += 1
								if top_idx < (i + 1) * cap_seq:
									top_x[top_idx] = result[6]
									top_y[top_idx] = result[7]
									top_p[top_idx] = result[8]
									top_idx += 1
						last_top = not last_top
					if top_idx == (i + 1) * cap_seq and bottom_idx == (i + 1) * cap_seq:
						break
			top_p[top_idx - 1] = 4
			bottom_p[bottom_idx - 1] = 4
	return top_x, top_y, top_p, bottom_x, bottom_y, bottom_p

def crop_top(in_x, in_y, in_p):
	in_x = in_x.astype(np.uint16)
	in_y = in_y.astype(np.uint16)
	in_p = in_p.astype(np.uint8)
	out_top_x = np.zeros((1, 1), dtype=np.uint16)
	out_top_y = np.zeros((1, 1), dtype=np.uint16)
	out_top_p = np.zeros((1, 1), dtype=np.uint8)
	out_bottom_x = np.zeros((1, 1), dtype=np.uint16)
	out_bottom_y = np.zeros((1, 1), dtype=np.uint16)
	out_bottom_p = np.zeros((1, 1), dtype=np.uint8)
	prev = [0, 0]
	in_len = len(in_x)
	min_x = 0
	min_y = 0
	for i in range(in_len):
		x = prev[0] + in_x[i] - 256
		y = prev[1] + in_y[i] - 256
		min_x = min(min_x, x)
		min_y = min(min_y, y)
		prev = [x, y]
		if in_p[i] == 4:
			break
	in_x[0] -= min_x
	in_y[0] -= min_y
	max_y = 0
	top_y = 256
	prev = [0, 0]
	for i in range(in_len):
		y = prev[1] + in_y[i] - 256
		top_y = min(y, top_y)
		max_y = max(y, max_y)
		prev[1] = y
		if in_p[i] == 4:
			break
	y_target = top_y + max(((max_y - top_y) * 3) // 4, 1)
	prev_top = [0, 0]
	prev_bottom = [0, y_target]
	prev = [0, 0]
	last_top = True
	# print(min_x, min_y, max_y, y_target)
	for i in range(in_len):
		# print(prev_top, prev_bottom)
		x = prev[0] + in_x[i] - 256
		y = prev[1] + in_y[i] - 256
		prev = [x, y]
		# print(x, y, prev_top, prev_bottom)
		if ((y < y_target) == last_top):
			if y < y_target:
				out_top_x = np.append(out_top_x, in_x[i])
				out_top_y = np.append(out_top_y, in_y[i])
				out_top_p = np.append(out_top_p, in_p[i])
				prev_top = [x, y]
			else:
				out_bottom_x = np.append(out_bottom_x, in_x[i])
				out_bottom_y = np.append(out_bottom_y, in_y[i])
				out_bottom_p = np.append(out_bottom_p, in_p[i])
				prev_bottom = [x, y]
		elif i > 0 and in_p[i-1] == 2:
			if y < y_target:
				prev_top, b_x, b_y, b_p = \
					append_point(in_p[i], x, y, prev_top)
				out_top_x = np.append(out_top_x, b_x)
				out_top_y = np.append(out_top_y, b_y)
				out_top_p = np.append(out_top_p, b_p)
			else:
				prev_bottom, b_x, b_y, b_p = \
					append_point(in_p[i], x, y, prev_bottom)
				out_bottom_x = np.append(out_bottom_x, b_x)
				out_bottom_y = np.append(out_bottom_y, b_y)
				out_bottom_p = np.append(out_bottom_p, b_p)
		else:
			if last_top:
				prev_top, prev_bottom, result = \
					connect_points(y_target, in_p[i], x, y, prev_top, prev_bottom)
				out_top_x = np.append(out_top_x, result[0])
				out_top_y = np.append(out_top_y, result[1])
				out_top_p = np.append(out_top_p, result[2])
				out_bottom_x = np.append(out_bottom_x, result[3])
				out_bottom_y = np.append(out_bottom_y, result[4])
				out_bottom_p = np.append(out_bottom_p, result[5])
				out_bottom_x = np.append(out_bottom_x, result[6])
				out_bottom_y = np.append(out_bottom_y, result[7])
				out_bottom_p = np.append(out_bottom_p, result[8])
			else:
				prev_bottom, prev_top, result = \
					connect_points(y_target + 1, in_p[i], x, y, prev_bottom, prev_top)
				out_bottom_x = np.append(out_bottom_x, result[0])
				out_bottom_y = np.append(out_bottom_y, result[1])
				out_bottom_p = np.append(out_bottom_p, result[2])
				out_top_x = np.append(out_top_x, result[3])
				out_top_y = np.append(out_top_y, result[4])
				out_top_p = np.append(out_top_p, result[5])
				out_top_x = np.append(out_top_x, result[6])
				out_top_y = np.append(out_top_y, result[7])
				out_top_p = np.append(out_top_p, result[8])
		last_top = y < y_target
	return out_top_x, out_top_y, out_top_p, out_bottom_x, out_bottom_y, out_bottom_p

def stack_to_image(name, data):
	top_x = np.squeeze(top_x)
	top_y = np.squeeze(top_y)
	top_p = np.squeeze(top_p)
	bottom_x = np.squeeze(bottom_x)
	bottom_y = np.squeeze(bottom_y)
	bottom_p = np.squeeze(bottom_p)
	# svg = svgwrite.Drawing('{}.svg'.format(name), size=(256, 256))
	# svg.add(svg.rect(insert=(0,0), size=('100%', '100%'), fill=255))
	img_top = Image.new('L', (256, 256), 255)
	img_bottom = Image.new('L', (256, 256), 255)
	img_combined = Image.new('L', (256, 256), 255)
	draw_top = ImageDraw.Draw(img_top, 'L')
	draw_bottom = ImageDraw.Draw(img_bottom, 'L')
	draw_combined = ImageDraw.Draw(img_combined, 'L')
	y_top_val = 0
	top_len = len(top_x)
	prev_top = [0, 0]
	for i in range(1, top_len):
		if top_p[i-1] < 4:
			x = prev_top[0] + (top_x[i] - 256)
			y = prev_top[1] + (top_y[i] - 256)
			# print('\tTop coordinates: {},{}'.format(x, y))
			if top_p[i-1] == 1:
				draw_top.line(((prev_top[0], prev_top[1]), (x, y)), fill=0, width=1)
				draw_combined.line(((prev_top[0], prev_top[1]), (x, y)), fill=0, width=1)
				# svg.add(svg.line(((prev_top[0], prev_top[1]), (x, y)), stroke=0, stroke_width=1))
			prev_top = [x, y]
			y_top_val = max(y_top_val, y + 1)
		else:
			# print('Saved top as {}_top.png'.format(name))
			break
	img_top.save('{}_top.png'.format(name))
	bottom_len = len(bottom_x)
	prev = [bottom_x[0] - 256, bottom_y[0] - 256]
	min_x = 0
	min_y = 0
	for i in range(1, bottom_len):
		if bottom_p[i - 1] < 4:
			x = prev[0] + (bottom_x[i] - 256)
			y = prev[1] + (bottom_y[i] - 256)
			prev = [x, y]
			min_x = min(min_x, x)
			min_y = min(min_y, y)
		else:
			break
	prev_bottom = [bottom_x[0] - 256 - min_x, bottom_y[0] - 256 + y_top_val - min_y]
	for i in range(1, bottom_len):
		if bottom_p[i - 1] < 4:
			x = prev_bottom[0] + (bottom_x[i] - 256)
			y = prev_bottom[1] + (bottom_y[i] - 256)
			# print('\tBottom coordinates: {},{}'.format(x, y))
			# print(bottom_x[i], bottom_y[i], x, y)
			if bottom_p[i - 1] == 1:
				draw_bottom.line(((prev_bottom[0], prev_bottom[1]), (x, y)), fill=0, width=1)
				draw_combined.line(((prev_bottom[0], prev_bottom[1]), (x, y)), fill=0, width=1)
			prev_bottom = [x, y]
		else:
			break
	img_bottom.save('{}_bottom.png'.format(name))
	print('Saved bottom as {}_bottom.png'.format(name))
	img_combined.save('{}_combined.png'.format(name))
	print('Saved combined as {}_combined.png'.format(name))