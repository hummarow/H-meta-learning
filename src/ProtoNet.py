# coding=utf-8
from prototypical_loss import prototypical_loss as loss_fn
from H_ProtoNet import First_Map
import torch
from DRTLearner import resnet101_dy
from torch import nn


class ProtoNet(nn.Module):
    def __init__(self, args):
        super(ProtoNet, self).__init__()

        self.middle_lr = args.middle_lr
        self.k_spt = args.k_spt

        if args.dataset == 'domainnet':
            self.map = resnet101_dy()
        else:
            self.map = First_Map(x_dim=args.channel)

        self.optim = torch.optim.Adam(params=self.map.parameters(),
                                      lr=self.middle_lr)

    def forward(self, x_spt, y_spt, x_qry, y_qry, train_or_test):
        task_num = len(x_spt)
        loss_num = 0
        acc_sum = 0

        self.map.train()
        for i in range(task_num):  # meta_batch_num = 1
            self.optim.zero_grad()
            x, y = torch.vstack((x_spt[i], x_qry[i])), torch.cat((y_spt[i], y_qry[i]))
            model_output = self.map(x)
            loss, acc = loss_fn(model_output, target=y,
                                n_support=self.k_spt)

            if train_or_test == 'train':
                loss.backward()
                self.optim.step()

            loss_num += loss
            acc_sum += acc

        loss_num /= task_num
        acc_sum /= task_num
        return loss_num, acc_sum, 0
