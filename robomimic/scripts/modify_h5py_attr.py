#!/usr/bin/env python
# coding: utf-8

# In[3]:


import h5py
import sys
import os
import numpy as np
import copy

import cv2


# In[ ]:


import robosuite as suite

# create environment instance
env = suite.make(
    env_name="PickPlaceCan", # try with other tasks like "Stack" and "Door"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

# reset the environment
env.reset()

new_xml = env.model.get_xml()


# In[2]:


src_data_path = '/media/yixuan_2T/diffusion_policy/data/robomimic/datasets/can/mh/demo_v141.hdf5'
tgt_data_path = '/media/yixuan_2T/diffusion_policy/data/robomimic/datasets/can/mh/new_demo_v141.hdf5'


# In[4]:


os.system('cp {} {}'.format(src_data_path, tgt_data_path))


# In[5]:


f = h5py.File(tgt_data_path, 'r+')


# In[10]:


for demo_ep in list(f['data'].keys()):
    f['data'][demo_ep].attrs.modify('model_file', new_xml)


# In[ ]:




