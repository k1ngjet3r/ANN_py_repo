from pyfirmata import Arduino
import time
import serial

# pin = 13
port = '/dev/cu.usbmodemFD131'
# board = Arduino(port)
# board.digital[pin].write(1)

# board.digital[pin].write(0)
# print(board.digital[pin].read())
s = serial.Serial(port, 9600)
while True:
    print(s.readline())
