
from adafruit_servokit import ServoKit
import board
import busio
import time

class CameraController():

    def __init__(self, bus=0, pan_servo=0, tilt_servo=3):
        self._pan_servo = pan_servo
        self._tilt_servo = tilt_servo

        if bus == 0:
            i2c_bus = (busio.I2C(board.SCL_1, board.SDA_1))
        else:
            i2c_bus = (busio.I2c(board.SCL, board.SDA))

        self._kit = ServoKit(channels=16, i2c=i2c_bus)

        self._pan_points =  {"min": 80,
                             "mid": 120,
                             "max": 180 }

        self._tilt_points = {"min": 80,
                             "home": 90,
                             "mid": 120,
                             "max": 180 }

        self.pan_angle = self._pan_points["mid"]
        self.tilt_angle = self._tilt_points["mid"]

    def update(self):
        self._kit.servo[self._pan_servo].angle = self.pan_angle
        self._kit.servo[self._tilt_servo].angle = self.tilt_angle

    def PanHome(self):
        self.pan_angle = self._pan_points["mid"]
        self.update()

    def TiltHome(self):
        self.tilt_angle = self._tilt_points["home"]
        self.update()

    def _check_angle(self, points, degrees):
        return degrees >= points["min"] and degrees <= points["max"]

    def SetPanAngle(self, degrees):
        if self._check_angle(self._pan_points, degrees):
            self.pan_angle = degrees
            self.update()

    def SetTiltAngle(self, degrees):
        if self._check_angle(self._tilt_points, degrees):
            self.tilt_angle = degrees
            self.update()

    def PanRight(self, degrees=1):
        if self._check_angle(self._pan_points, self.pan_angle + degrees):
            self.pan_angle += degrees
            self.update()

    def PanLeft(self, degrees=1):
        if self._check_angle(self._pan_points, self.tilt_angle - degrees):
            self.pan_angle -= degrees
            self.update()

    def TiltUp(self, degrees=1):
        if self._check_angle(self._tilt_points, self.tilt_angle + degrees):
            self.tilt_angle += degrees
            self.update()

    def TiltDown(self, degrees=1):
        if self._check_angle(self._tilt_points, self.tilt_angle - degrees):
            self.tilt_angle -= degrees
            self.update()