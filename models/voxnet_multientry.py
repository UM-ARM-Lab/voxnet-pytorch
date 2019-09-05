import torch
from torch import nn
from collections import OrderedDict


class VoxNetBody(nn.Module):

    def __init__(self):
        super(VoxNetBody, self).__init__()
        self.net = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(in_channels=1,
                                out_channels=32, kernel_size=5, stride=2)),
            ('lkrelu1', nn.LeakyReLU()),
            ('drop1', nn.Dropout(p=0.2)),
            ('conv2', nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)),
            ('lkrelu2', nn.LeakyReLU()),
            ('pool2', nn.MaxPool3d(2)),
            ('drop2', nn.Dropout(p=0.3))
        ]))

    def forward(self, x):
        return self.net(x)


class VoxNetHead(nn.Module):

    def __init__(self, num_features, num_classes):
        super(VoxNetHead, self).__init__()

        self.net = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(num_features, 128)),
            ('relu1', nn.ReLU()),
            ('drop3', nn.Dropout(p=0.4)),
            ('fc2', nn.Linear(128, num_classes))
        ]))

    def forward(self, x):
        return self.net(x)


class VoxNetMultiEntry(nn.Module):

    def __init__(self, num_classes, input_shape=(32, 32, 32),
                 use_same_net=False,
                 num_grids=3):
                 #weights_path=None,
                 #load_body_weights=True,
                 #load_head_weights=True):
        """
        VoxNet: A 3D Convolutional Neural Network for Real-Time Object Recognition.

        Modified in order to accept different input shapes.

        Also modified for multiple voxel-grids input

        Parameters
        ----------
        num_classes: int, optional
            Default: 10
        input_shape: (x, y, z) tuple, optional
            Default: (32, 32, 32)
        use_same_network: Use same body for each input voxel grid?
        Notes
        -----
        Weights available at: url to be added

        If you want to finetune with custom classes, set load_head_weights to False.
        Default head weights are pretrained with ModelNet10.
        """
        super(VoxNetMultiEntry, self).__init__()

        self.use_same_network = use_same_net
        self.num_grids = num_grids
        self.input_shape = input_shape
        # Use just one body network or have multiple copies for each grid?
        if self.use_same_network:
            self.body = [VoxNetBody()]

        else:
            self.body = []
            for i in range(num_grids):
                self.body.append(VoxNetBody)

        # Trick to accept different input shapes
        x = self.body[0](torch.autograd.Variable(
            torch.rand((1, 1) + input_shape)))
        first_fc_in_features = num_grids
        for n in x.size()[1:]:
            first_fc_in_features *= n

        print(first_fc_in_features)
        self.head = VoxNetHead(first_fc_in_features, num_classes)

        #if weights_path is not None:
        #    weights = torch.load(weights_path)
        #    if load_body_weights:
        #        self.body.load_state_dict(weights["body"])
        #    elif load_head_weights:
        #        self.head.load_state_dict(weights["head"])

    def forward(self, voxel_grids):
        '''

        :param voxel_grids: Batch_size x Num_grids x grid_size x grid_size x grid_size

        :return:
        '''
        features = []
        for i in range(self.num_grids):
            body_idx = 0 if self.use_same_network else i
            x = self.body[body_idx](voxel_grids[:, i].view(-1, 1, *self.input_shape))
            features.append(x.view(x.size()[0], -1))

        # Stack features into one big vector
        features = torch.cat(features, 1)
        return self.head(features)
