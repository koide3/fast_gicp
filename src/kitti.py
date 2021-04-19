#!/usr/bin/python3
import os
import sys
import time
import numpy
import pygicp
from matplotlib import pyplot

def main():
	if len(sys.argv) < 2:
		print('usage: kitti.py /path/to/kitti/sequences/00/velodyne')
		return

	# List input files
	seq_path = sys.argv[1]
	filenames = sorted([seq_path + '/' + x for x in os.listdir(seq_path) if x.endswith('.bin')])

	# You can choose any of FastGICP, FastVGICP, FastVGICPCuda, or NDTCuda
	reg = pygicp.FastGICP()

	# pygicp classes have the same interface as the C++ version
	# reg.set_num_threads(8)
	# reg.set_max_correspondence_distance(2.0)

	stamps = [time.time()]		# for FPS calculation
	poses = [numpy.identity(4)]	# sensor trajectory

	for i, filename in enumerate(filenames):
		# Read and downsample input cloud
		points = numpy.fromfile(filename, dtype=numpy.float32).reshape(-1, 4)[:, :3]
		points = pygicp.downsample(points, 0.25)

		if i == 0:
			reg.set_input_target(points)
			delta = numpy.identity(4)
		else:
			reg.set_input_source(points)
			delta = reg.align()
			reg.swap_source_and_target()

		# Accumulate the delta to compute the full sensor trajectory
		poses.append(poses[-1].dot(delta))

		# FPS calculation for the last ten frames
		stamps = stamps[-9:] + [time.time()]
		print('fps:%.3f' % (len(stamps) / (stamps[-1] - stamps[0])))

		# Plot the estimated trajectory
		traj = numpy.array([x[:3, 3] for x in poses])

		if i % 30 == 0:
			pyplot.clf()
			pyplot.plot(traj[:, 0], traj[:, 1])
			pyplot.axis('equal')
			pyplot.pause(0.01)


if __name__ == '__main__':
	main()
