#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from std_srvs.srv import SetBool, SetBoolResponse
from threading import Thread

def move_gripper(srv, controlled_joint_name, target_position, joint_target_publish_rate, real_js, js_topic_name):
    joint_min_val = 0.0
    joint_max_val = 1.0

    target_velocity = (joint_max_val - joint_min_val) / 2

    rospy.loginfo('Wait for ' + js_topic_name)
    initial_joint_state = rospy.wait_for_message(js_topic_name, JointState)
    rospy.loginfo(js_topic_name + ' message recived')

    target_position[0] = initial_joint_state.position[initial_joint_state.name.index(controlled_joint_name)]

    finish = False

    if (srv.data):
        dx = -abs(target_velocity) / joint_target_publish_rate
    else:
        dx = abs(target_velocity) / joint_target_publish_rate

    rate = rospy.Rate(joint_target_publish_rate)
    rospy.loginfo('dx: ' + str(dx))
    while not finish:
        target_position[0] += dx
        if (target_position[0] < 0):
            target_position[0] = 0
            finish = True
        real_position = real_js[0].position[real_js[0].name.index(controlled_joint_name)]
        if ((real_position < joint_min_val and dx < 0) or (real_position > joint_max_val and dx > 0)):
            finish = True
            rospy.loginfo('Joint limit: ' + str(real_position))
        if ((target_position[0] < joint_min_val and dx < 0) or (target_position[0] > joint_max_val and dx > 0)):
            finish = True
            rospy.loginfo('Target joint limit: ' + str(target_position[0]))
        rate.sleep()

    return SetBoolResponse(True,"Gripper motion ends")


def jointStateSubscriber(data, args):
    real_js = args[0]
    real_js[0] = data


def jointTargetPublisher(jt_pub, joint_target_publish_rate, controlled_joint_name, target_position):
    rate = rospy.Rate(joint_target_publish_rate)
    jt_msg = JointState()

    while not rospy.is_shutdown():
        jt_msg.header = Header()
        jt_msg.header.stamp = rospy.Time.now()
        jt_msg.name = [controlled_joint_name]
        jt_msg.position = target_position
        jt_msg.velocity = [0]
        jt_msg.effort = [0]
        jt_pub.publish(jt_msg)
        rate.sleep()


def main():
    node_name = 'panda_hand_server'
    robot_name =  'panda'
    gripper_name = 'panda_hand'
    controlled_joint_name = 'panda_left_joint_circle'

    rospy.loginfo('node_name = ' + node_name)
    rospy.loginfo('robot_name = ' + robot_name)
    rospy.loginfo('gripper_name = ' + gripper_name)
    rospy.loginfo('controlled_joint_name = ' + controlled_joint_name)

    rospy.init_node(node_name)

    if (rospy.has_param('/pybullet_simulation/joint_target_publish_rate')):
        joint_target_publish_rate = rospy.get_param('/pybullet_simulation/joint_target_publish_rate')
    else:
        rospy.logerr('/pybullet_simulation/joint_target_publish_rate pasam not set')
        raise SystemExit

    rospy.loginfo('Waiting for /joint_states topic')

    real_js = []

    try:
        real_js = [rospy.wait_for_message('/joint_states', JointState, timeout=10)]
        js_topic_name = '/joint_states'
    except (rospy.ROSException):
        rospy.loginfo("/joint_states doesn't recived")
        rospy.loginfo('Waiting for /' + robot_name + '/joint_states topic')
        try:
            real_js = [rospy.wait_for_message('/' + robot_name + '/joint_states', JointState, timeout=2)]
            js_topic_name = '/' + robot_name + '/joint_states'
        except (rospy.ROSException):
            rospy.logerr("/" + robot_name + "/joint_states doesn't recived")
            exit(0)
        except (rospy.ROSInterruptException):
            rospy.logerr("ROSInterruptException")
            exit(0)
    except (rospy.ROSInterruptException):
        rospy.logerr("ROSInterruptException")
        exit(0)



    position = [ real_js[0].position[real_js[0].name.index(controlled_joint_name)] ]

    rospy.Subscriber(js_topic_name, JointState, jointStateSubscriber, (real_js, ))

    jt_pub = rospy.Publisher('/' + robot_name + '/joint_target', JointState, queue_size=1)

    jt_pub_thread = Thread(target=jointTargetPublisher, args=(jt_pub, joint_target_publish_rate, controlled_joint_name, position))
    jt_pub_thread.start()

    rospy.Service("/move_" + gripper_name, SetBool, lambda msg: move_gripper(msg, controlled_joint_name, position, joint_target_publish_rate, real_js, js_topic_name))

    print("Service /move_" + gripper_name + " started")

    rospy.spin()
    jt_pub_thread.join()


if __name__ == '__main__':
    main()
