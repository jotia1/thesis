""" This program will create a window in which a dot will move. Parameters of
the dots movement can be set in code, or by using the control panel displayed.
This program will be used to create simple movement to be recorded by the
neuromorphic event-based dynmaic vision sesnor (DVS). 

Created by Joshua Arnold
5/09/2015
"""
try:
    import tkinter as tk
except ImportError:
    import Tkinter as tk
import sys
import random
import time
import math
import socket  #To send UDP to DAVIS

# Ratio between screen pixels to DVS pixels
PIXEL_FACTOR = 4

class Controller(object):
    """ This class will act as the controller of the display window.
    Interactions with the display canvas should be done using this class.
    """
    VERBOSE = False
    DELAY_MS = 30
    OUTSTR = "{0}, {1}, {2}\n"
    def __init__(self, root):
        """ Initialise all variables to do with dot movement.
        """
        self.root = root  #tkroot
        self.dot_size = 4
        self.dot_rad = self.dot_size//2
        self.vx = 5
        self.vy = 5
        self.speed = 6
        self.grad = self.vy / self.vx
        self.cur_line = None
        self.callback = None
        self.shouldFlash = False
        self.recording = False
        self.outfile = None
        self.last_file_write = None
        self.time_passed = 0

        # grey used was #A0A0A0
        self.canvas = tk.Canvas(root,bg='white', relief='sunken', bd=2)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.dot = self.create_circle(-self.dot_rad, -self.dot_rad, \
            self.canvas)
        self.createTopLevel()
        self.borders = [None for x in range(4)]
        self.markers = []
        self.canvas.bind("<Configure>", lambda e: self.setBounds(e.width, e.height))
        
    def change_speed(self):
        self.speed = int(self._speedE.get())

    def change_size(self):
        self.dot_size = int(self._sizeE.get())
        self.dot_rad = self.dot_size//2
        self.setBounds(self.cwidth, self.cheight)

    def recording_pressed(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(('', 0))
        addr = 'localhost', 8997
        BUFSIZE = 1024

        if self.recording:
            line = 'stoplogging' 
            self._recordingB.config(bg='grey')
            if not self.outfile:
                raise Exception("Outfile already closed")
            self.outfile.close()
            self.outfile = None
            self.last_file_write = None
            self.time_passed = 0
        else:
            fname = self._fnameE.get()
            line = 'startlogging ' + fname
            self._recordingB.config(bg='red')
            if self.outfile or self.last_file_write or self.time_passed != 0:
                raise Exception("Logging variables not reset")
            self.outfile = open(fname + '_log.csv', 'w')
            # send UDP reset command to DAVIS
            s.sendto(bytes("zerotimestamps", 'UTF-8'), addr)
            data, fromaddr = s.recvfrom(BUFSIZE)
            print('client recived %r from %r' % (data, fromaddr))
            # TODO not technically accurate but why not?
            self.last_file_write = time.perf_counter()
        self.recording = not self.recording

        s.sendto(bytes(line, 'UTF-8'), addr)
        data, fromaddr = s.recvfrom(BUFSIZE)
        print('client recived %r from %r' % (data, fromaddr))
        s.close()

    def createTopLevel(self):
        self.top = tk.Toplevel()
        self.top.title("Controls")
        self.top.geometry("120x300")
        tk.Label(self.top, text="Speed:").pack()
        self._speedE = tk.Entry(self.top)
        self._speedE.pack()
        tk.Button(self.top, text="ChangeSpeed", command=self.change_speed).pack()

        tk.Label(self.top, text="Size:").pack()
        self._sizeE = tk.Entry(self.top)
        self._sizeE.pack()
        tk.Button(self.top, text="ChangeSize", command=self.change_size).pack()

        tk.Label(self.top, text="Start/stop").pack()
        self._fnameE = tk.Entry(self.top)
        self._fnameE.pack()
        self._recordingB = tk.Button(self.top, text="Start", command=self.recording_pressed)
        self._recordingB.pack()

        tk.Button(self.top, text="Setup", command=self.togFlashBorders).pack()
        tk.Button(self.top, text="Exit All", command=self.root.destroy).pack()

    def setBounds(self, width, height):
        """ At initialisation and when window resizes the bounds need to be 
            reset.
        """
        self.cwidth = width
        self.cheight = height
        self.boundx = self.cwidth + self.dot_rad # Dot will completely leave screen before wrapping around
        self.boundy = self.cheight + self.dot_rad
        self.diagonal = self.cwidth + self.cheight

        # Order: Top, right, bottom, left
        tl = (self.dot_rad, self.dot_rad)
        tr = (self.cwidth - self.dot_rad, self.dot_rad)
        bl = (self.dot_rad, self.cheight - self.dot_rad)
        br = (self.cwidth - self.dot_rad, self.cheight - self.dot_rad)

        self.border_lines = [(tl, tr), (tr, br), (br, bl), (bl, tl)]

    def togFlashBorders(self):
        self.shouldFlash = not self.shouldFlash
        if self.shouldFlash:
            self.flashBorders()

    def flashBorders(self):
        draw_line = lambda l: self.canvas.create_line(l, \
                                                fill='black',  \
                                                width=self.dot_size)
        if self.borders[0]:
            for i, b in enumerate(self.borders):
                self.canvas.delete(b)
                self.borders[i] = None
        else:
            for i, l in enumerate(self.border_lines):
                self.borders[i] = draw_line(l)
        if self.shouldFlash:
            self.root.after(self.DELAY_MS, self.flashBorders)

    def draw(self):
        """ Responsible for moving the dot to it's next position and redrawing
            the screen. Will set a callback for 30ms to call itself.
        """
        self.canvas.move(self.dot, self.vx, self.vy)
        if self.recording:
            now = time.perf_counter()
            if not self.last_file_write:
                self.last_file_write = now
            cur_time = self.time_passed + (now - self.last_file_write)
            self.time_passed = cur_time
            self.last_file_write = now
            x, y = self.box2pos(self.canvas.coords(self.dot))
            timeus = int(cur_time * 1e6)
            self.outfile.write(self.OUTSTR.format(timeus, int(x/PIXEL_FACTOR),\
                     int(y/PIXEL_FACTOR)))

        if self.contain():  # went off screen
            # allow time for flashes
            self.callback = self.root.after(self.DELAY_MS * 4, self.draw)
        else:
            self.callback = self.root.after(self.DELAY_MS, self.draw)

        if (self.VERBOSE):
            print(self.canvas.coords(self.dot))
            print('-'*10)

    def start(self):
        self.draw()
        

    def flashMeta(self):
        """ Draw Meta-data to the screen and set a callback to clear the 
            screen after DELAY_MS ms.
        """
        x0, y0 = self.box2pos(self.canvas.coords(self.dot))
        x1 = x0 + self.diagonal * self.vx
        y1 = y0 + self.diagonal * self.vy
        self.cur_line = self.canvas.create_line(x0, y0, x1, y1, \
                                                fill='black',  \
                                                width=self.dot_size)
        self.root.after(self.DELAY_MS, self.clear)
        

    def clear(self):
        """ Clear cur_line (created in flashMeta) from the screen
        """
        self.canvas.delete(self.cur_line)
        self.cur_line = None

    def pos2box(self, pos):
        """ Convert a position (x, y) to a bounding box to draw the dot
        """
        x, y = pos
        rad = self.dot_rad
        return (x-rad, y-rad, x+rad, y+rad)

    def box2pos(self, box):
        """ Given a bounding box of the dot, return its center position (x, y)
        """
        return (box[0] + self.dot_rad, box[1] + self.dot_rad)

    def flashMarker(self):
        """ flash data in the middle of flash spikes
        """
        ypos = 0
        gap = self.cheight // 5
        lwidth = self.dot_size
        mcolour = 'black'
        self.markers.append(self.canvas.create_line(0, self.cheight//2, self.cwidth, \
                                                self.cheight//2, \
                                                fill=mcolour,  \
                                                width=lwidth))
        self.markers.append(self.canvas.create_line(self.cwidth//2, 0, self.cwidth//2, \
                                                self.cheight, \
                                                fill=mcolour,  \
                                                width=lwidth))

        self.root.after(self.DELAY_MS, self.delMarkers)

    def delMarkers(self):
        """ Clear any tk objects in the self.markers list
        """
        for i in self.markers:
            self.canvas.delete(i)
        self.markers = []

    def contain(self):
        """ Keep the dot from leaving the screen and update the gradient if 
            off screen
        """
        x, y = self.box2pos(self.canvas.coords(self.dot))
        if not (-self.dot_rad > x or x > self.boundx or -self.dot_rad > y or y > self.boundy): #Not out
            return False  #still on screen

        #self.root.after_cancel(self.callback)
        # this is a hack so flash meta works
        # flash is based on self.vx and vy (at end of run draw go backwards)
        bef = self.box2pos(self.canvas.coords(self.dot))
        bvx = self.vx
        bvy = self.vy
        self.vx = -self.vx
        self.vy = -self.vy
        self.flashMeta()

        
        # set the new gradient and velocity
        self.vx = random.random() * 4  # range is [0,4]
        self.vy = random.random() * 4
        theta = math.atan(self.vy / self.vx) # unlikely to be zero here
        self.vx = self.speed * math.cos(theta)
        self.vy = self.speed * math.sin(theta)

        # Generate new (x, y) and velocities
        edge = random.choice(["top", "bottom", "left", "right"])
        if edge == "top":
            y = 0
            x = random.randint(int(self.boundx*0.25), int(self.boundx*0.75))
            self.vx = self.vx if bool(random.getrandbits(1)) else -self.vx
        elif edge == "bottom":
            y = self.boundy
            x = random.randint(int(self.boundx*0.25), int(self.boundx*0.75))
            self.vy = -self.vy
            self.vx = self.vx if bool(random.getrandbits(1)) else -self.vx
        elif edge == "left":
            x = 0
            y = random.randint(int(self.boundy*0.25), int(self.boundy*0.75))
            self.vy = self.vy if bool(random.getrandbits(1)) else -self.vy
        else:
            x = self.boundx
            y = random.randint(int(self.boundy*0.25), int(self.boundy*0.75))
            self.vx = -self.vx
            self.vy = self.vy if bool(random.getrandbits(1)) else -self.vy

        self.canvas.coords(self.dot, self.pos2box((x,y)))
        # wait for last samples flash to finish then flash
        self.root.after(self.DELAY_MS * 2, self.flashMarker)
        self.root.after(self.DELAY_MS * 4, self.flashMeta) 
        return True

    def create_circle(self, x, y, canvas, **kwargs):
        """ Draw the circle on the canvas at the specified x and y position
        """
        return canvas.create_oval(self.pos2box((x,y)), fill='black', **kwargs)

    def exit(self):
        self.root.destroy()

def main():
    root = tk.Tk()
    camx, camy = (128, 128)
    winx, winy = (camx*PIXEL_FACTOR, camy*PIXEL_FACTOR)
    root.geometry(str(winx) + 'x' + str(winy))
    root.title("Thesis dataset generator")
    window = Controller(root)
    window.setBounds(winx, winy)
    window.start()

    root.mainloop()

if __name__ == "__main__":
    main()
