import Tkinter as tk
import sys


class Controller(object):
	def __init__(self, root, mode):
		self.root = root
		self.mode = mode
		self.dot_size = 6
		self.dot_rad = self.dot_size/2
		self.cwidth = 800
		self.cheight = 800
		self.boundx = self.cwidth + self.dot_rad # Dot will completely leave screen before wrapping around
		self.boundy = self.cheight + self.dot_rad
		self.vx = 5
		self.vy = 6
		self.cur_line = None


		self.canvas = tk.Canvas(root,bg='white', relief='sunken', bd=2)
		self.canvas.pack(fill=tk.BOTH, expand=True)

		self.dot = self.create_circle(-self.dot_rad, -self.dot_rad, \
			self.canvas)

	def draw(self):
		print self.canvas.coords(self.dot)
		self.canvas.move(self.dot, self.vx, self.vy)
		print self.canvas.coords(self.dot)
		print '-'*10
		self.root.after(30, self.draw)
		self.contain()

	def start(self):
		""" Start with a calibration line first
		"""
		if self.cur_line:  #start dots
			self.canvas.delete(self.cur_line)
			self.cur_line = None
			self.draw()
			return

		self.cur_line = self.canvas.create_line(0, self.boundy/2, \
								self.boundx, self.boundy/2, fill='black', width=10)
		self.root.after(30, self.start)

	def pos2box(self, pos):
		x, y = pos
		rad = self.dot_rad
		return (x-rad, y-rad, x+rad, y+rad)

	def box2pos(self, box):
		return (box[0] + self.dot_rad, box[1] + self.dot_rad)

	def contain(self):
		x, y = self.box2pos(self.canvas.coords(self.dot))
		#print x ,y
		x = x % self.boundx
		y = y % self.boundy
		#print x, y
		self.canvas.coords(self.dot, self.pos2box((x,y)))

	def create_circle(self, x, y, canvas, **kwargs):
		return canvas.create_oval(self.pos2box((x,y)), fill='black', **kwargs)

	def exit(self):
		self.root.destroy()

def main():
	if len(sys.argv) != 2:
		print "Usage: thesis_data mode"
		return

	root = tk.Tk()
	root.geometry("800x800")
	root.title("Thesis dataset generator")
	window = Controller(root, sys.argv[1])
	window.start()

	root.mainloop()

if __name__ == "__main__":
	main()