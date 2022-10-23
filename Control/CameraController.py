
from adafruit_servokit import ServoKit
import board
import busio
import time

class CameraController():

    def __init__(self, bus=1, pan_servo=0, tilt_servo=3):
        self._pan_servo = pan_servo
        self._tilt_servo = tilt_servo

        if bus == 0:
            i2c_bus = (busio.I2C(board.SCL_1, board.SDA_1))
        else:
            i2c_bus = (busio.I2C(board.SCL, board.SDA))

        self._kit = ServoKit(channels=16, i2c=i2c_bus)

        self._pan_points =  {"min": 80,
                             "mid": 120,
                             "max": 180 }

        self._tilt_points = {"min": 0,
                             "home": 90,
                             "mid": 120,
                             "max": 180 }

        self.pan_angle = self._pan_points["mid"]
        self.tilt_angle = self._tilt_points["home"]

    def dump(self):
        print("CameraController:")
        print("  Pan:  {}".format(self.pan_angle))
        print("  Tilt: {}".format(self.tilt_angle))

    def update(self):
        self._kit.servo[self._pan_servo].angle = self.pan_angle
        self._kit.servo[self._tilt_servo].angle = self.tilt_angle

    def PanHome(self, do_update=True):
        self.pan_angle = self._pan_points["mid"]
        if do_update:
            self.update()

    def TiltHome(self, do_update=True):
        self.tilt_angle = self._tilt_points["home"]
        if do_update:
            self.update()

    def Home(self, do_update=True):
        self.PanHome(do_update=False)
        self.TiltHome(do_update=False)
        if do_update:
            self.update()

    def _check_angle(self, points, degrees):
        return degrees >= points["min"] and degrees <= points["max"]

    def SetPanAngle(self, degrees, do_update=True):
        if self._check_angle(self._pan_points, degrees):
            self.pan_angle = degrees
            if do_update:
                self.update()

    def SetTiltAngle(self, degrees, do_update=True):
        if self._check_angle(self._tilt_points, degrees):
            self.tilt_angle = degrees
            if do_update:
                self.update()

    def PanRight(self, degrees=1, do_update=True):
        if degrees == "max":
            self.pan_angle = self._pan_points["min"]
            if do_update:
                self.update()
        elif self._check_angle(self._pan_points, self.pan_angle - degrees):
            self.pan_angle -= degrees
            if do_update:
                self.update()

    def PanLeft(self, degrees=1, do_update=True):
        if degrees == "max":
            self.pan_angle = self._pan_points["max"]
            if do_update:
                self.update()
        elif self._check_angle(self._pan_points, self.pan_angle + degrees):
            self.pan_angle += degrees
            if do_update:
                self.update()

    def TiltUp(self, degrees=1, do_update=True):
        if self._check_angle(self._tilt_points, self.tilt_angle - degrees):
            self.tilt_angle -= degrees
            if do_update:
                self.update()

    def TiltDown(self, degrees=1, do_update=True):
        if self._check_angle(self._tilt_points, self.tilt_angle + degrees):
            self.tilt_angle += degrees
            if do_update:
                self.update()