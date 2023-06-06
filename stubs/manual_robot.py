import re
import shutil
import time
import random
import zipfile
import yaml
from datetime import datetime
import os
from os import listdir
from os.path import isfile, join
from pathlib import Path
from importlib import resources
import sys 

from robomaster.robot import Robot
import robomaster
import cv2
import select 
import platform
if platform.system() == "Windows":
    import msvcrt
else:
    from getch import getch, getche
    import tty
    import termios

"""
The purpose of this script is just to test moving and taking photos with a connected Robomaster robot.
"""

class NonBlockingConsole(object):

    def __enter__(self):
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, type, value, traceback):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)


    def get_data(self):
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.read(1)
        return False

def get_char():
    if platform.system() == "Windows":
        char = msvcrt.getch()
        print(f"{type(char)}: {char}")
        return char
    else:
        return getch()


def initialize_robo():
    # Initialise robot
    robot = Robot()
    robot.initialize(conn_type="ap", proto_type='tcp')
    robot.set_robot_mode('chassis_lead')
    robot_ver = robot.get_version()
    robot_mode = robot.get_robot_mode()
    print(f"ver: {robot_ver} | mode: {robot_mode}")
    this_ip = robomaster.config.LOCAL_IP_STR
    print(f"controller ip: {this_ip}")
    controller_ip = "192.168.2.32"
    print(f"setting controller ip to: {controller_ip}")
    robomaster.config.LOCAL_IP_STR = controller_ip
    #robot.camera.start_video_stream(display=False, resolution='720p')
    return robot 


def print_menu():
    print('''
    w, s, a, d  :   move robot
             t  :   take photo
             q  :   quit
    ''')


if __name__ == '__main__':
    
    is_robo = False
    print('Initialise Robot? (Y/n)')
    robo_input = get_char().lower()
    robot = None
    if robo_input == b'y':
        is_robo = True
        robot = initialize_robo()
    key = True
    try:
        #with NonBlockingConsole() as nbc:  # Uncomment to use NonBlockingConsole to test in Linux-based OS.
        if True:  # Uncomment to use this placeholder to test in Windows os.            
            while True:
                if key:
                    print_menu()
                    
                key = get_char().lower()
                #key = nbc.get_data()
                
                x_val = 0
                y_val = 0
                
                if key == b'x':
                    print("exiting.")
                    break
                elif key == b't':
                    if is_robo: 
                        cam_version = robot.camera.get_version()
                        print(f"camera version: {cam_version}")
                        vid_stream_addr = robot.camera.video_stream_addr
                        print(f"vid stream addr:{vid_stream_addr}")
                        
                        print('\nRobot taking photo...')
                        #is_photo_taken = robot.camera.take_photo()
                        #print(f"is photo taken? {is_photo_taken}")
                        
                        # Robot.take_photo() doesn't seem to work, but the below code block works.
                        robot.camera.start_video_stream(display=False, resolution='720p')
                        img = robot.camera.read_cv2_image(strategy='newest')
                        robot.camera.stop_video_stream()
                        print(f"image: {img}")
                        print(f"Robot: photo taken.")
                        #cv2.imshow("Robot", img)
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        img_path = f"D:/TIL-AI 2023/til-22-finals/til-23-finals/data/imgs/robot_photo_{timestamp}.jpg"
                        print(f"attempting save to: {img_path}")
                        cv2.imwrite(img_path,img)
                    else:
                        print(f"no robot to take photo with.")

                elif key == b'w':
                    x_val = 0.2
                    z_val = 0
                elif key == b's':
                    x_val = -0.2
                    z_val = 0
                elif key == b'a':
                    y_val = -0.2
                    z_val = 0
                elif key == b'd':
                    y_val = 0.2
                    z_val = 0
                elif key == b'q':
                    z_val = -30
                elif key == b'e':
                    z_val = 30
                elif key == b'\x03':
                    exit()
                else:
                    break

                if is_robo and (x_val != 0 or y_val != 0 or z_val != 0):
                    robot.chassis.drive_speed(x=x_val, y=y_val, z=z_val,  timeout=0.5)
                    time.sleep(0.5)
                    robot.chassis.drive_speed(x=0.00, y=0.00, z=z_val, timeout=0.5)  # stop.
                    x_val = 0
                    y_val = 0
    finally:
        if is_robo:
            robot.chassis.drive_speed(x=0.00, y=0.00, z=0.00, timeout=0.5)  # stop.
            print("closing robot...\n")
            robot.close()
