#!/usr/bin/env python3

import rospy

from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray


if __name__ == '__main__':

    rospy.init_node('rosparam_to_rostopic', anonymous=True)

    pub_dict = {}
    rate = rospy.Rate(10)

    exec_params_ns = 'exec_params'
    
    while not rospy.is_shutdown():
        if rospy.has_param('/' + exec_params_ns):
            exec_params = rospy.get_param('/' + exec_params_ns)
            if type(exec_params) is dict:
                if 'actions' in exec_params.keys():
                    if type(exec_params['actions']) is dict:
                        for action_name in exec_params['actions'].keys():
                            action = exec_params['actions'][action_name]
                            if type(action) is dict:
                                for action_param_name in action.keys():
                                    if (action_param_name == 'skills'):
                                        for skill_name in action['skills'].keys():
                                            skill = action['skills'][skill_name]
                                            
                                            if type(skill) is float:
                                                key = '/' + exec_params_ns + '/'+action_name+'/'+skill_name
                                                if key not in pub_dict:
                                                    pub_dict[key] = rospy.Publisher(key, Float32, queue_size=1)
                                                pub_dict[key].publish(skill)
                                            if type(skill) is int:
                                                key = '/' + exec_params_ns + '/'+action_name+'/'+skill_name
                                                if key not in pub_dict:
                                                    pub_dict[key] = rospy.Publisher(key, Float32, queue_size=1)
                                                pub_dict[key].publish(skill)

                                            if type(skill) is dict:
                                                for param_name in skill.keys():

                                                    if type(skill[param_name]) is int:
                                                        key = '/' + exec_params_ns + '/'+action_name+'/'+skill_name+'/'+param_name
                                                        if key not in pub_dict:
                                                            pub_dict[key] = rospy.Publisher(key, Float32, queue_size=1)
                                                        pub_dict[key].publish(skill[param_name])

                                                    if type(skill[param_name]) is float:
                                                        key = '/' + exec_params_ns + '/'+action_name+'/'+skill_name+'/'+param_name
                                                        if key not in pub_dict:
                                                            pub_dict[key] = rospy.Publisher(key, Float32, queue_size=1)
                                                        pub_dict[key].publish(skill[param_name])

                                                    if type(skill[param_name]) is list:
                                                        if type(skill[param_name][0]) is int or type(skill[param_name][0]) is float:
                                                            key = '/' + exec_params_ns + '/'+action_name+'/'+skill_name+'/'+param_name
                                                            if key not in pub_dict:
                                                                pub_dict[key] = rospy.Publisher(key, Float32MultiArray, queue_size=1)

                                                            msg = Float32MultiArray()
                                                            msg.data = skill[param_name]
                                                            pub_dict[key].publish(msg)
                                    else:
                                        if type(action[action_param_name]) is float:
                                            key = '/' + exec_params_ns + '/'+action_name+'/'+action_param_name
                                            if key not in pub_dict:
                                                pub_dict[key] = rospy.Publisher(key, Float32, queue_size=1)
                                            pub_dict[key].publish(action[action_param_name])
                                        if type(action[action_param_name]) is int:
                                            key = '/' + exec_params_ns + '/'+action_name+'/'+action_param_name
                                            if key not in pub_dict:
                                                pub_dict[key] = rospy.Publisher(key, Float32, queue_size=1)
                                            pub_dict[key].publish(action[action_param_name])

                                        if type(action[action_param_name]) is dict:
                                            for param_name in action[action_param_name].keys():

                                                if type(action[action_param_name][param_name]) is int:
                                                    key = '/' + exec_params_ns + '/'+action_name+'/'+action_param_name+'/'+param_name
                                                    if key not in pub_dict:
                                                        pub_dict[key] = rospy.Publisher(key, Float32, queue_size=1)
                                                    pub_dict[key].publish(action[action_param_name][param_name])

                                                if type(action[action_param_name][param_name]) is float:
                                                    key = '/' + exec_params_ns + '/'+action_name+'/'+action_param_name+'/'+param_name
                                                    if key not in pub_dict:
                                                        pub_dict[key] = rospy.Publisher(key, Float32, queue_size=1)
                                                    pub_dict[key].publish(action[action_param_name][param_name])

                                                if type(action[action_param_name][param_name]) is list:
                                                    if type(action[action_param_name][param_name][0]) is int or type(action[action_param_name][param_name][0]) is float:
                                                        key = '/' + exec_params_ns + '/'+action_name+'/'+action_param_name+'/'+param_name
                                                        if key not in pub_dict:
                                                            pub_dict[key] = rospy.Publisher(key, Float32MultiArray, queue_size=1)

                                                        msg = Float32MultiArray()
                                                        msg.data = action[action_param_name][param_name]
                                                        pub_dict[key].publish(msg)
        rate.sleep()

    print('End')
