#!/usr/bin/env python3

import rospy
import rospkg
import sys
from skills_util_msgs.srv import RunTree
from pybullet_simulation.srv import SpawnModel
from pybullet_simulation.srv import DeleteModel
from pybullet_simulation.srv import SaveState, DeleteState
from geometry_msgs.msg import Pose
import pyexcel_ods3 as od
import numpy as np

rospack = rospkg.RosPack()
path = rospack.get_path('pybullet_simulation')
path = path + '/src'
sys.path.append(path)

import pybullet_sim_class

py_class = pybullet_sim_class.PybulletSim('uppa')