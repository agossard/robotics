# SDA = pin.SDA_1
# SCL = pin.SCL_1
# SDA_1 = pin.SDA
# SCL_1 = pin.SCL

from adafruit_servokit import ServoKit
import board
import busio
import time
#from approxeng.input.selectbinder import ControllerResource


# On the Jetson Nano
# Bus 0 (pins 28,27) is board SCL_1, SDA_1 in the jetson board definition file
# Bus 1 (pins 5, 3) is board SCL, SDA in the jetson definition file
# Default is to Bus 1; We are using Bus 0, so we need to construct the busio first ...
print("Initializing Servos")
#i2c_bus0=(busio.I2C(board.SCL_1, board.SDA_1))
i2c_bus1=(busio.I2C(board.SCL, board.SDA))
print("Initializing ServoKit")
kit = ServoKit(channels=16, i2c=i2c_bus1)
print('Initialized')

# delay = 0.01

# for its in range(0, 3):
#     the_range = range(100, 180)

#     for i in the_range:
#         kit.servo[3].angle = i
#         kit.servo[0].angle = i
#         time.sleep(delay)

#     the_range = range(180, 100, -1)

#     for i in the_range:
#         kit.servo[3].angle = i
#         kit.servo[0].angle = i
#         time.sleep(delay)


while True:
    angle = input()

    commands = angle.split(' ')
    angle = commands[1]
    servo = commands[0]

    if angle == 'q':
        break

    print('Setting {} to angle {} degrees'.format(servo, angle))
    if servo == "tilt":
        kit.servo[3].angle = int(angle)
    else:
        kit.servo[0].angle = int(angle)

# kit[0] is the bottom servo
# kit[1] is the top serv
# print("Done initializing")

# for i in range(0,5):

#     print('Iteration {}'.format(i))

#     sweep = range(0,180)
#     for degree in sweep :
#         kit.servo[0].angle=degree
#         # kit.servo[1].angle=degree
#         time.sleep(0.01)

#     #time.sleep(2)
#     sweep = range(180,0, -1)
#     for degree in sweep :
#         kit.servo[0].angle=degree
#         time.sleep(0.01)
    
# last_presses = None
# while True:
#        with ControllerResource() as joystick:
#            print(type(joystick).__name__)
#            while joystick.connected:
#                 axis_list = [ 'lx', 'ry' ]
#                 for axis_name in axis_list:
#                     # desired_angle is in degrees
#                     joystick_value = joystick[axis_name]
#                     # The joystick value goes from -1.0 ... 1.0 (a range of 2)
#                     # Normalize within a range of 180 degrees
#                     desired_angle = (joystick_value+1)/2*180
                    
#                     if  axis_name == 'lx' :
#                         kit.servo[0].angle=desired_angle
#                         # print(axis_name, joystick[axis_name])
                        
#                     if axis_name == 'ry' :
#                          kit.continuous_servo[1].throttle=joystick[axis_name]
            
