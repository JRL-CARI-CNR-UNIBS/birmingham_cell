#!/usr/bin/env python3
import copy
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

    
logdir = '/home/gauss/projects/personal_ws/src/birmingham_cell/birmingham_cell_tests/log/final_log'

repositories = ['f_g','f_gg','f_i','f_ii','g_t_p','t_p','f_p_i_h']

for repository in repositories:
    ea = event_accumulator.EventAccumulator(logdir + '/' + repository)
    ea.Reload()

    if repository == 'f_g' or repository == 'f_gg':
        model_name = 'Grasping forces model'
    if repository == 'f_i' or repository == 'f_i':
        model_name = 'Insertion forces model'
    if repository == 'g_t_p':
        model_name = 'Generic theoretical position model'
    if repository == 't_p':
        model_name = 'Theoretical position model'
    if repository == 'f_p_i_h':
        model_name = 'Peg-in-hole forces model'

    tags = ea.Tags()["scalars"]

    for tag in tags:
        events = ea.Scalars(tag)
        steps = [event.step for event in events]
        values = [event.value for event in events]
        
        # print(tag)
        tag_name = copy.copy(tag)
        tag_name = tag_name.replace('rollout/','')
        tag_name = tag_name.replace('train/','')
        if 'time' in tag_name:
            continue

        plt.figure()
        plt.plot(steps, values, label=tag)
        plt.gca().xaxis.get_major_formatter().set_scientific(True)
        plt.gca().xaxis.get_major_formatter().set_powerlimits((-1, 1))
        plt.xlabel('Step',fontsize=16)
        if tag_name == 'ep_len_mean':
            plt.ylabel('Value [step]',fontsize=16)
        elif tag_name == 'ep_rew_mean':
            plt.ylabel('Value [u]',fontsize=16)
        elif tag_name == 'success_rate':
            plt.ylabel('Value [%/100]',fontsize=16)
        else:
            plt.ylabel('Value',fontsize=16)
        plt.grid(True, which='both')
        plt.savefig(logdir + '/' + repository +  '/' + repository + '_' + tag_name)
        plt.close()
        






