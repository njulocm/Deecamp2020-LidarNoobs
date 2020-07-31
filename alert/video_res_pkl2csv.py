import pickle
import pandas as pds
import numpy as np
import sys

filename = sys.argv[1]

# filename = 'baseline_video_result.pkl'
print('Start converting ...')

with open(filename, 'rb') as f:
    video_result = pickle.load(f)
video_result_df = pds.DataFrame(columns='image_idx,name,x,y,z,l,w,h,rotation_y'.split(','))
for i in range(len(video_result)):
    # 获取当前帧的信息
    frame_info = video_result[i]
    frame_df = pds.DataFrame(columns='image_idx,name,x,y,z,l,w,h,rotation_y'.split(','))
    frame_df['name'] = frame_info['name']
    frame_df.loc[:, 'x,y,z'.split(',')] = frame_info['location']
    frame_df.loc[:, 'l,w,h'.split(',')] = frame_info['dimensions']
    frame_df.loc[:, 'rotation_y'] = frame_info['rotation_y']
    frame_df['image_idx'] = frame_info['metadata']['image_idx']
    # 拼接
    video_result_df = pds.concat([video_result_df, frame_df])
# 计算距离
dist = (video_result_df['x'].to_numpy()**2 + video_result_df['y'].to_numpy()**2).tolist()
dist = np.sqrt(np.array(dist))
video_result_df['dist'] = dist
video_result_df.to_csv(filename[:-3] + 'csv', index=False)
print('Success!')

