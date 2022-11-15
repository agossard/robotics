#import jetson.inference
#import jetson.utils
import Jetson.GPIO as GPIO
import time
from dataclasses import dataclass
from enum import Enum
import numpy as np

class MotorController:

    MAX_SPEED = 100
    MIN_SPEED = 25

    def __init__(self,
                 pin_forward: int, 
                 pin_backward: int, 
                 pwm_pin: int):
        self.pin_forward = pin_forward
        self.pin_backward = pin_backward
        self.pwm_pin = pwm_pin
        
        self.speed = 0
        
        GPIO.setup(pwm_pin, GPIO.OUT)
        GPIO.setup(pin_forward, GPIO.OUT)
        GPIO.setup(pin_backward, GPIO.OUT)

        self.pwm = GPIO.PWM(pwm_pin, 100)

    def set_pwm(self, speed):
    
        if self.speed == 0 and speed != 0:
            self.pwm.start(abs(speed))
        elif self.speed != 0 and speed == 0:
            self.pwm.stop()
        else:
            self.pwm.ChangeDutyCycle(abs(speed))

        self.speed = speed

    def stop(self):
        self.set_motor(speed=0)

    def set_motor(self, speed: int=100):
        if speed > 0:
            GPIO.output(self.pin_forward, GPIO.HIGH)
            GPIO.output(self.pin_backward, GPIO.LOW)
        elif speed < 0:
            GPIO.output(self.pin_backward, GPIO.HIGH)
            GPIO.output(self.pin_forward, GPIO.LOW)
        else:
            GPIO.output(self.pin_backward, GPIO.LOW)
            GPIO.output(self.pin_forward, GPIO.LOW)

        self.set_pwm(speed)

    def increase(self, incr=1):
        new_speed = np.min((MotorController.MAX_SPEED, self.speed + incr))
        self.set_motor(new_speed)

    def decrease(self, incr=1):
        new_speed = np.max((-MotorController.MAX_SPEED, self.speed - incr))
        self.set_motor(new_speed)


class Side(Enum):
    LEFT = 1
    RIGHT = 2

class DriveController():

    def __init__(self):

        # This should probably happen outside this class
        GPIO.setmode(GPIO.BOARD)

        self.motors = { Side.LEFT: MotorController(pin_forward=37, pin_backward=35, pwm_pin=33),
                        Side.RIGHT: MotorController(pin_forward=29, pin_backward=31, pwm_pin=32)}
            

    def stop(self):
        for k, v in self.motors.items():
            v.stop()

    def forward(self, speed, duration=None):

        self.motors[Side.LEFT].set_motor(speed)
        self.motors[Side.RIGHT].set_motor(speed)

    def backward(self, speed, duration=None):

        self.motors[Side.LEFT].set_motor(-speed)
        self.motors[Side.RIGHT].set_motor(-speed)

    def spin_left(self, speed):
        self.motors[Side.LEFT].set_motor(-speed)
        self.motors[Side.RIGHT].set_motor(speed)

    def spin_right(self, speed):
        self.motors[Side.LEFT].set_motor(speed)
        self.motors[Side.RIGHT].set_motor(-speed)

    def speed_up(self, incr=1):
        for k, v in self.motors.items():
            v.increase(incr)

    def slow_down(self, incr=1):
        for k, v in self.motors.items():
            v.decrease(incr)

    def right(self, incr=1):
        self.motors[Side.LEFT].increase(incr)
        self.motors[Side.RIGHT].decrease(incr)

    def left(self, incr=1):
        self.motors[Side.LEFT].decrease(incr)
        self.motors[Side.RIGHT].increase(incr)

    def dump_status(self):
        print('DriveController')
        for k, v in self.motors.items():
            print('  {}: {}'.format(k, v.speed))