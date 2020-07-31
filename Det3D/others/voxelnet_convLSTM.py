from ..registry import DETECTORS
from .single_stage import SingleStageDetector

# NEW
from .convlstm import ConvLSTM
import torch
import time

@DETECTORS.register_module
class VoxelNet(SingleStageDetector):
    def __init__(
            self,
            reader,
            backbone,
            neck,
            bbox_head,
            train_cfg=None,
            test_cfg=None,
            pretrained=None,
            convLSTM_cfg=None # NEW
    ):
        super(VoxelNet, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )

        self.head_input_channels = bbox_head['in_channels'] # 获取head的input_channels
        self.neck_ds_conv = torch.nn.Conv2d(128, self.head_input_channels, 1) # 对neck出来的feature进一步下采样减少通道

        self.convLSTM_cfg = convLSTM_cfg
        if self.convLSTM_cfg is None:
            self.convLSTM_layer = None
        else:  # 配置convLSTM的参数
            self.convLSTM_layer = ConvLSTM(input_dim=self.convLSTM_cfg['input_channels'],
                                           hidden_dim=self.convLSTM_cfg['hidden_channels'],
                                           kernel_size=self.convLSTM_cfg['kernel_size'],
                                           num_layers=len(self.convLSTM_cfg['hidden_channels']),
                                           batch_first=self.convLSTM_cfg['batch_first'])
            self.features_before = {'batch': 0, 'features_tensor': None}
            # self.features_before用于存储前几帧的feature，类型是dict，
            # 其中feature_before['batch']是当前进行了几个batch，用来判断是否可以过convlstm，
            # feature_before['features_tensor']用来存放之前的feature tensor
            
            if self.convLSTM_cfg['skip']:  # 如果需要间隔采样
                self.skip_num = 2
            else:
                self.skip_num = 1

    def extract_feat(self, data):

        # 提取最后一帧的feature（这边是原代码）
        input_features = self.reader(data["features"], data["num_voxels"])
        x = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        if self.with_neck:
            x = self.neck(x)  # x.shape=(4,128,176,352)
            x = self.neck_ds_conv(x) # x.shape=(4,head_channels,176,352)

        # 过convLSTM来融合前几帧信息
        # features_before = {'batch':None, 'features_tensor':None}
        if self.convLSTM_layer is None:  # 不过convlstm
            xx = x
        elif self.features_before['batch'] == 0:  # 之前还没有batch
            self.features_before['features_tensor'] = x
            self.features_before['features_tensor'].detach_() # 不需要求导
            self.features_before['batch'] += 1
            xx = x
        elif self.features_before['batch'] == 1:  # 之前有过一个batch
            self.features_before['features_tensor'] = torch.cat([self.features_before['features_tensor'], x])
            self.features_before['features_tensor'].detach_() # 不需要求导
            self.features_before['batch'] += 1
            xx = x
        else:  # 有过至少两个batch，可以过convLSTM
            # features_before['features_tensor'].shape 应该是 (8,128,176,352)
            # convLSTM的输入(B,S,C,W,H)=(4,9,128,176,352)
            yy = torch.cat([self.features_before['features_tensor'], x])  # 先拼起来
            # xx = torch.cat([yy[i:i + 9, :, :, :].unsqueeze(dim=0) for i in range(4)])
            xx = torch.cat(
                    [yy[[j for j in range(i, i + 9, self.skip_num)], :, :, :].unsqueeze(dim=0) for i in range(4)])
            # xx的shape为（4，seq，channel，176，352）
            [xx], _ = self.convLSTM_layer(xx)  # xx的shape为（4，9，128，176， 352）
            # xx = xx.mean(1)  # 对序列取均值，shape为（4，128，176，352）
            xx = xx[:,-1,:,:] # 也可以取序列最后一个，shape为（4，128，176，352）
            # print('走了convLSTM，成功！！！！')
            xx = xx + x  #res结构

            # 更新features_before
            self.features_before['features_tensor'] = torch.cat([self.features_before['features_tensor'][4:8,:,:,:], xx])
            self.features_before['features_tensor'].detach_() # 不需要求导
            self.features_before['batch'] += 1
            # if self.features_before['batch'] == 9000//4
            #     self.features_before['batch'] = 0

            # 一个epoch结束，batch要归0，但是这边注意一下test的时候可能有问题，要不就先别归0了！！！！！需要讨论

        return xx

    def forward(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x = self.extract_feat(data)
        preds = self.bbox_head(x)  # 得到一个list，每个task对应一个元素（为dict），dict的key包括box_preds、cla_preds和dir_preds(可选)

        if return_loss:
            return self.bbox_head.loss(example, preds)  # train阶段返回loss，应该后面会接反向传播
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)  # test阶段，用来产生需要的框
