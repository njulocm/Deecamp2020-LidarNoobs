# 1 文件说明
* Det3D是最终采用的模型相关配置信息
* CenterPoint.zip是没有采纳的模型
* visualization是可视化相关代码
* alert是雷路行车预警系统相关代码

# 2 Det3D模型测试说明
## 2.1 文件说明
* config/dbsample3_size3_lr1_batch8.py 是模型配置信息
* video_infos.pkl是video的info
* tools/test.py用于对video的每一帧进行检测，这边会同时输出FPS
* model中是最终模型final.pth和日志信息
* others是进行其他尝试写的脚本

## 2.2 测试方法
* 用tools/test.py替换原来的test.py
* 将video_info/video_info.pkl放入/home/ma-user/work/workspace/DeepCamp_Lidar/video_infos.pkl路径下
* 将配置文件config/dbsample3_size3_lr1_batch8.py放在Det3D/examples/文件夹下
* 将模型final.pth放在Det3D/res/文件夹下
* 测试命令

```
cd Det3D
python3.6 tools/test.py examples/dbsample3_size3_lr1_batch8.py res/final.pth --out res/video_result.pkl
```

## 2.3 测试结果
* 我们测试的速度是11.6FPS


# 3 雷路行车预警系统
## 3.1 将pkl文件转为csv文件
```
python video_res_pkl2csv.py video_result.pkl
```
## 3.2 运行雷路行车预警系统
```
python alert.py video_result.csv 1
```



