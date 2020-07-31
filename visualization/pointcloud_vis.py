#
# Copyright (c) 2020 JinTian.
#
# This file is part of alfred
# (see http://jinfagang.github.io).
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
"""
showing 3d point cloud using open3d
"""
import numpy as np
#import cv2
import matplotlib.pyplot as plt
try:
    from open3d import *
    import open3d as o3d
except ImportError:
    print('importing 3d_vis in alfred-py need open3d installed.')
    exit(0)


def draw_pcs_open3d(geometries,i):
    """
    drawing the points using open3d
    it can draw points and linesets
    ```
    point_cloud = PointCloud()
    point_cloud.points = Vector3dVector(pcs)


    points = [[0,0,0],[1,0,0],[0,1,0],[1,1,0],
                [0,0,1],[1,0,1],[0,1,1],[1,1,1]]
    lines = [[0,1],[0,2],[1,3],[2,3],
                [4,5],[4,6],[5,7],[6,7],
                [0,4],[1,5],[2,6],[3,7]]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = LineSet()
    line_set.points = Vector3dVector(points)
    line_set.lines = Vector2iVector(lines)
    line_set.colors = Vector3dVector(colors)
    draw_pcs_open3d([point_cloud, line_set])
    ```
    """
    def capture_depth(vis):
        depth = vis.capture_depth_float_buffer()
        plt.imshow(np.asarray(depth))
        plt.show()
        return False

    def capture_image(vis):
        image = vis.capture_screen_float_buffer()
        plt.imshow(np.asarray(image))
        plt.show()
        return False
#我添的
    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(0.0, 10.0)
        return False


    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for g in geometries:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 1
    
#我添的
    vc=vis.get_view_control()
    #vc.set_lookat(np.array([[0],[0],[10]]))
    vc.set_lookat(np.array([[0],[0],[0]]))
#前视角参数lookat0，0，10；rotate0，-520；550，0；scale-32
#纯前视角参数lookat0，0，2；rotate0，-523；550，0；scale-34
#俯视参数lookat0，0，0；scale-29
    vc.scale(-29.0)
    #vc.scale(-32.0)
    #vc.rotate(0,-520.0)
    #vc.rotate(550.0,0)
    #vc.set_front(np.array([[0],[0],[0]]))
    #vc.set_zoom(-1000.0)
    #vc.rotate(x=200.0,y=500.0)
    #print(vc.get_field_of_view())
    
    
    #opt.show_coordinate_frame = True
    
#我添的
    #vis.register_animation_callback(rotate_view)
    file_root='../result/frame'
    file_name=str(i)
    file_tile='.png'
    file_name=file_root+file_name+file_tile
    #vis.capture_screen_image(r'C:\Users\xuqian\Desktop\csi.png', do_render=True)
    vis.capture_screen_image(file_name,do_render=True)
    
    
    #vis.run()
    #vis.destroy_window()
