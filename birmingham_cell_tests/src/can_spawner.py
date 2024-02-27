#!/usr/bin/env python3

import rospy
import rospkg
import sys
import tf
from skills_util_msgs.srv import RunTree
from pybullet_utils.srv import SpawnModel
from pybullet_utils.srv import DeleteModel
from pybullet_utils.srv import SaveState
import pyexcel_ods3 as od
from shape_msgs.msg import MeshTriangle, Mesh
import pyassimp
from geometry_msgs.msg import WrenchStamped, Pose, Point, TransformStamped
from moveit_msgs.msg import CollisionObject, PlanningScene, PlanningSceneComponents, AttachedCollisionObject
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from moveit_python import PlanningSceneInterface
from moveit_python import MoveGroupInterface, PlanningSceneInterface
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import JointState
from threading import Thread


class TFSubscriver:
    def __init__(self, c_obj, apply_scene_clnt, scenes, tfl):
        self.c_obj = c_obj
        self.apply_scene_clnt = apply_scene_clnt
        self.tfl = tfl
        self.scenes = scenes
        rospy.Subscriber('/tf',
                         TFMessage,
                         self.tfSubscriver)

    def tfSubscriver(self,data):
        for tf in data.transforms:
            if ( tf.child_frame_id == can_name ):
                print('IN')
                rospy.sleep(0.5)
                if c_obj in self.scenes[0].world.collision_objects:
                    self.scenes[0].world.collision_objects.remove(self.c_obj)
                    self.c_obj.operation = self.c_obj.REMOVE
                    self.scenes[0].world.collision_objects.append(self.c_obj)
                    self.apply_scene_clnt.call(self.scenes[0])
                    self.scenes[0].world.collision_objects.remove(self.c_obj)
                self.c_obj.operation = self.c_obj.ADD
                self.scenes[0].world.collision_objects.append(self.c_obj)
                self.apply_scene_clnt.call(self.scenes[0])

def attached_check(c_obj, apply_scene_clnt, scenes, tfl):
    scenes_ = scenes
    a_obj = AttachedCollisionObject()
    a_obj.object = c_obj
    attached = False

    while not rospy.is_shutdown():
#        print('A')
        if (tfl.frameExists(can_name,)):
            if (rospy.has_param('/' + can_name + '/attached')):
                if (rospy.has_param('/' + can_name + '/attached_link')):
                    if (rospy.has_param('/' + can_name + '/touch_links')):
                        if(rospy.get_param('/' + can_name + '/attached')):
                            a_obj.link_name = rospy.get_param('/' + can_name + '/attached_link')
                            a_obj.touch_links = rospy.get_param('/' + can_name + '/touch_links')
                            if not attached:
                                rospy.loginfo('Attach')
#                                scenes[0].world.collision_objects.remove(c_obj)
#                                c_obj.operation = c_obj.REMOVE
#                                scenes[0].world.collision_objects.append(c_obj)
                                scenes[0].robot_state.attached_collision_objects.append(a_obj)
                                apply_scene_clnt.call(scenes[0])
                                attached = True
                        else:
                            if attached:
                                rospy.loginfo("Deattach")
                                scenes[0].robot_state.attached_collision_objects.remove(a_obj)
                                a_obj.object.operation = a_obj.object.REMOVE
                                scenes[0].robot_state.attached_collision_objects.append(a_obj)
                                apply_scene_clnt.call(scenes[0])
                                scenes[0].robot_state.attached_collision_objects.remove(a_obj)
                                a_obj.object.operation = a_obj.object.ADD
                                scenes[0].world.collision_objects.append(a_obj.object)
                                apply_scene_clnt.call(scenes[0])
                                attached = False


if __name__ == '__main__':

    rospy.init_node('can_spawner')
    rospy.sleep(10)

    rospack = rospkg.RosPack()
    mesh_path = rospack.get_path('battery_cell_description') + '/meshes/can.STL'

    can_name = 'object'

    mesh = Mesh()
    with pyassimp.load(mesh_path) as mesh_file:
        print('Mesh loaded')
        for face in mesh_file.meshes[0].faces:
            triangle = MeshTriangle()
            if len(face) == 3:
                triangle.vertex_indices = [face[0],
                                           face[1],
                                           face[2]]
            mesh.triangles.append(triangle)
        for vertex in mesh_file.meshes[0].vertices:
            point = Point()
            point.x = vertex[0]
            point.y = vertex[1]
            point.z = vertex[2]
            mesh.vertices.append(point)

    pose = Pose()
    pose.position.x    = 0
    pose.position.y    = 0
    pose.position.z    = -0.115
    pose.orientation.x = 0
    pose.orientation.y = 0
    pose.orientation.z = 0
    pose.orientation.w = 1

    mesh_pose = Pose()
    mesh_pose.position.x = 0
    mesh_pose.position.y = 0
    mesh_pose.position.z = 0
    mesh_pose.orientation.x = 0
    mesh_pose.orientation.y = 0
    mesh_pose.orientation.z = 0
    mesh_pose.orientation.w = 1

    c_obj = CollisionObject()
    c_obj.header.frame_id = can_name
    c_obj.id = can_name + '_'
    c_obj.pose = pose
    c_obj.meshes.append(mesh)
    c_obj.mesh_poses.append(mesh_pose)
    c_obj.operation = c_obj.ADD

    get_scene_clnt = rospy.ServiceProxy('get_planning_scene', GetPlanningScene)
    apply_scene_clnt = rospy.ServiceProxy('apply_planning_scene', ApplyPlanningScene)

    req = PlanningSceneComponents()
    req.components = sum([PlanningSceneComponents.WORLD_OBJECT_NAMES,
                          PlanningSceneComponents.WORLD_OBJECT_GEOMETRY,
                          PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS])
    scenes = []
    scene = PlanningScene()
    scenes.append(scene)
    scenes[0] = get_scene_clnt.call(req).scene
    scenes[0].robot_state.joint_state.name     = []
    scenes[0].robot_state.joint_state.position = []
    scenes[0].robot_state.joint_state.velocity = []
    scenes[0].robot_state.joint_state.effort   = []
    scenes[0].is_diff = True
    scenes[0].robot_state.is_diff = True

    tfl = tf.TransformListener()
    tfs = TFSubscriver(c_obj, apply_scene_clnt, scenes, tfl)

    attached_check_thread = Thread(target=attached_check, args=(c_obj, apply_scene_clnt, scenes, tfl))
    attached_check_thread.start()

    rospy.spin()

    attached_check_thread.join()
