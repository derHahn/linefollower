import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class Sensor:
	def __init__(self):
		self.position = None
		self.reading = None

	def __repr__(self):
		return f"Position {self.position}, Reading {self.reading}\t"

class Robot:
	def __init__(self, floor_image_path='line.png'):
		self.size_robot = 5
		self.step_size = 1
		self.floor_image = self.load_image(floor_image_path)
		self._im_max_x = self.floor_image.shape[0]
		self._im_max_y = self.floor_image.shape[1]		
		self.position = self.get_starting_position()

		self.sensors = (Sensor(), Sensor(), Sensor())
		self.orientation = np.array([0, -1])

		self.initialize_sensors()
		self.get_sensor_readings()

		self.track = [self.position]
		self.tracks1 = [(self.position[0] - self.size_robot, self.position[1])]
		self.tracks2 = [(self.position[0] + self.size_robot, self.position[1])]
		self._time_steps = 10000

	def start(self):
		for _ in range(self._time_steps):
			self.make_decision()
			self.move()

	def get_starting_position(self):
		initial_pos = np.array([np.random.randint(self._im_max_x + 1 - self.size_robot), 
			                    np.random.randint(self._im_max_y + 1 - self.size_robot)])
		return initial_pos

	def load_image(self, floor_image_path):
		image = Image.open(floor_image_path)
		image = image.convert('RGB')
		return np.asarray(image, dtype=np.int32)

	def get_sensor_readings(self):
		for sensor in self.sensors:
			sensor.reading = int(sum(self.floor_image[(sensor.position[0], sensor.position[1])]))

	def check_valid_step(self):
		valid = True
		if (self.position + (self.orientation * self.step_size))[0] + self.size_robot >= self._im_max_x:
			valid = False
		elif (self.position + (self.orientation * self.step_size))[1] + self.size_robot >= self._im_max_y:
			valid = False
		elif (self.position + (self.orientation * self.step_size))[0] - self.size_robot <= 0: 
			valid = False
		elif (self.position + (self.orientation * self.step_size))[1] - self.size_robot <= 0:
			valid = False
		if not valid:	
			self.rotate()
			self.check_valid_step()

	def move(self):		
		self.check_valid_step()
		self.position += (self.orientation * self.step_size)
		for sensor in self.sensors:
			sensor.position += (self.orientation * self.step_size)
		self.get_sensor_readings()
		self.track.append((self.position[0], self.position[1]))
		self.tracks1.append(self.sensors[1].position)
		self.tracks2.append(self.sensors[2].position)


	def rotate(self):
		self.sensors[1].position = self.sensors[0].position.copy()
		self.sensors[0].position = self.sensors[2].position.copy()
		self.sensors[2].position = self.position + ((self.orientation * -1) * self.size_robot)

		self.get_sensor_readings()
		self.next_orientation()

	def next_orientation(self):
		orientations = [(0,-1), (1,0), (0,1), (-1,0)]
		index = orientations.index(tuple(self.orientation))

		if index == len(orientations) - 1:
			self.orientation = np.array(orientations[0])
		else:
			self.orientation = np.array(orientations[index + 1])

	def initialize_sensors(self):
		self.sensors[0].position = np.array([self.position[0], self.position[1] - self.size_robot])
		self.sensors[1].position = np.array([self.position[0] - self.size_robot, self.position[1]])
		self.sensors[2].position = np.array([self.position[0] + self.size_robot, self.position[1]])

	def make_decision(self):
		if self.sensors[0].reading > 500:
			if (self.sensors[1].reading - self.sensors[2].reading) > 50:
				self.rotate()
			elif (self.sensors[2].reading - self.sensors[1].reading) > 50:
				self.rotate()
				self.rotate()
				self.rotate()

	def visualize_track(self):
		plt.imshow(self.floor_image.swapaxes(0,1))
		plt.plot([step[0] for step in self.track], [step[1] for step in self.track], color='red', lw=2)
		plt.show()


if __name__ == "__main__":
	my_robot = Robot()
	my_robot.start()
	my_robot.visualize_track()