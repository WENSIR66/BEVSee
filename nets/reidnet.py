from collections import OrderedDict

import torch
from reid.utils.serialization import load_checkpoint
from reid.utils import to_torch
from reid import models
from torchvision import transforms
import warnings

warnings.filterwarnings("ignore")

'''
该代码实现了一个行人重识别（ReID）模型 reidnet，其核心组件是基于ResNet50构建的网络。
该模型通过 forward 方法进行前向传播，接收图像输入，输出特征。并提供了一个 loadmodel 方法，允许从文件加载训练好的权重，方便恢复和使用训练好的模型。
'''
# only accept 256, 128
class reidnet(torch.nn.Module):
    def __init__(self):
        super(reidnet, self).__init__()

        self.model = models.create('resnet50', num_features=1024, dropout=0, num_classes=751)
        checkpoint = load_checkpoint('models/reid_checkpoint.pth.tar')
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.train()

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)
        output = self.model(x, 'pool5')
        return output

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        process_dict = OrderedDict()
        for key in state['state_dict_reid'].keys():
            if key.startswith('module.'):
                process_dict[key[7:]] = state['state_dict_reid'][key]
            elif 'Mixed_6' in key:
                continue
            else:
                process_dict[key] = state['state_dict_reid'][key]
        self.load_state_dict(process_dict)
        print('Load reidnet all parameter from: ', filepath)
