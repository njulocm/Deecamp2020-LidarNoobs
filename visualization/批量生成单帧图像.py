# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 10:36:40 2020

@author: xuqian
"""


import open3d as o3d
import numpy as np
import pandas as pd
from alfred.fusion.kitti_fusion import load_pc_from_file
from alfred.vis.pointcloud.pointcloud_vis import draw_pcs_open3d
from alfred.fusion.common import compute_3d_box_lidar_coords

    
    
df=pd.read_csv('../data/dbsample3_size3_lr1_batch8_video_result.csv')
frame_list=[]
for i in range(11001,13001):
    dv=df.loc[df['image_idx']==i].values.tolist()
    frame_list.append(dv)
for i in range(11001,13001):
    fileroot='../data/test_video_filter/0'
    filename=str(i)
    filetile='.bin'
    filename=fileroot+filename+filetile
    pcs=load_pc_from_file(filename)
    pcs = pd.DataFrame(pcs[:, :3])
    percentile=pcs[2].quantile(0.00016)
    pcs=pcs.drop(index=pcs.loc[pcs[2]<=percentile].index).values
    geometries = []
    pcobj = o3d.geometry.PointCloud()
    pcobj.points = o3d.utility.Vector3dVector(pcs)
    geometries.append(pcobj)
    
    for p in frame_list[i-11001]:
        xyz = np.array([p[2:5]])
        lwh = np.array([p[5:8]])
        r_y = [p[-1]]
        pts3d = compute_3d_box_lidar_coords(xyz, lwh, angles=r_y, origin=(0.5, 0.5, 0.5), axis=2)
        lines = [[0,1],[1,2],[2,3],[3,0],
                 [4,5],[5,6],[6,7],[7,4],
                 [0,4],[1,5],[2,6],[3,7]]
        if p[1]=='Car':#pink
            colors = [[1, 0, 1] for i in range(len(lines))]
        elif p[1]=='Cyclist':#yellow
            colors = [[255, 255, 0] for i in range(len(lines))]
        elif p[1]=='Truck':#green
            colors = [[0, 255, 0] for i in range(len(lines))]
        elif p[1]=='Pedestrian':#red
            colors = [[255, 0, 0] for i in range(len(lines))]
        else:#Tricar-white
            colors = [[255, 255, 255] for i in range(len(lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(pts3d[0])
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(line_set)
    draw_pcs_open3d(geometries, i)
