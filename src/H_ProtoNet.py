import torch.nn as nn
import torch.optim
from prototypical_loss import prototypical_loss as loss_fn
from DRTLearner import resnet101_dy
from H_MAML import contrastive_step


def conv_block(in_channels, out_channels, pooling_size=2):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(pooling_size),
    )


class First_Map(nn.Module):
    def __init__(self, x_dim=1, hid_dim=64, z_dim=64, first_pooling_size=2):
        super(First_Map, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim, first_pooling_size),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )

    def forward(self, x):
        x = self.encoder(x)  # x:(600,64,1,1)
        return x.view(x.size(0), -1)

    def reset_map(self):
        for layer in self.encoder:
            for sub_layer in layer:
                if sub_layer.__class__.__name__ == "ReLU":
                    continue
                layer.reset_parameters()


class Second_Map(nn.Module):
    def __init__(self, input_dim=64, hid_dim=64, z_dim=64):
        super(Second_Map, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, z_dim),
        )

    def forward(self, x):
        return self.encoder(x)

    def reset_map(self):
        for layer in self.encoder:
            if layer.__class__.__name__ == "ReLU":
                continue
            layer.reset_parameters()


class H_ProtoNet(nn.Module):
    def __init__(self, args):
        super(H_ProtoNet, self).__init__()
        self.top_lr = args.top_lr
        self.middle_lr = args.middle_lr

        # 暂时用task_spt控制middle_step_num
        # self.bottom_step_num = args.bottom_step_num
        # self.bottom_step_num_test = args.bottom_step_num_test
        # self.middle_step_num = args.middle_step_num
        # self.middle_step_num_test = args.middle_step_num_test

        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.k_qry_test = args.k_qry_test
        self.n_way = args.n_way

        self.task_spt = args.task_spt
        self.task_qry = args.task_qry
        self.task_qry_test = args.task_qry_test

        self.task_cluster_batch_num = args.task_cluster_batch_num
        self.task_cluster_batch_num_test = args.task_cluster_batch_num_test
        assert (
            self.task_cluster_batch_num == 1 and self.task_cluster_batch_num_test == 1
        )  # 有必要batch吗,先写无batch版

        if args.dataset == "domainnet":
            # self.first_map = resnet101_dy()
            self.first_map = First_Map(x_dim=args.channel, first_pooling_size=4)
            self.second_map = Second_Map(input_dim=2048, hid_dim=2048, z_dim=2048)
        else:
            self.first_map = First_Map(x_dim=args.channel)
            self.second_map = Second_Map()
        self.net = nn.Sequential(self.first_map, self.second_map)
        self.first_optim = torch.optim.Adam(self.first_map.parameters(), lr=self.top_lr)
        self.second_optim = torch.optim.Adam(
            self.second_map.parameters(), lr=self.middle_lr
        )

        self.reset = args.reset_second_map
        self.second_order = not args.reset_second_map

        self.dataset = args.dataset
        self.contrastive = args.contrastive

    def learning_on_the_high_level_task(
        self,
        mode,
        task_num,
        x_spt,
        x_qry,
        y_spt,
        y_qry,
    ):
        q_loss_sum = 0
        q_acc_sum = 0
        for j in range(task_num):  # 这里是循环还是并列求平均没想清楚？
            if mode == "T_S":
                if self.contrastive:
                    loss_s_task = contrastive_step(
                        self.net,
                        self.second_order,
                        self.net.parameters(),
                        x_spt,
                        y_spt,
                        x_qry,
                        y_qry,
                        self.middle_lr,
                    )
                    loss = loss_s_task[-1]
                    acc = 0
                    # contrastive 1st order adaptation
                else:
                    x = torch.vstack((x_spt[j], x_qry[j]))
                    y = torch.cat((y_spt[j], y_qry[j]))

                    self.first_optim.zero_grad()
                    x_cluster_space = self.second_map(self.first_map(x))  # 二阶映射
                    loss, acc = loss_fn(x_cluster_space, y, n_support=self.k_spt)

                loss.backward()
                self.first_optim.step()
            elif mode == "T_Q":
                q_loss_sum += loss
                q_acc_sum += acc

        return q_loss_sum / task_num, q_acc_sum / task_num

    def forward(
        self, eight_cluster_batch, train_or_test
    ):  # 快更新first_map，慢更新second_map效果比反过来好
        for i in range(self.task_cluster_batch_num):  # task_cluster_batch_num=1
            eight_cluster_item = [eight_cluster_batch[j][i] for j in range(8)]

            # T_S
            x_spt_task_spt_set = eight_cluster_item[0]
            x_spt_task_qry_set = eight_cluster_item[1]
            # T_Q
            x_qry_task_spt_set = eight_cluster_item[2]
            x_qry_task_qry_set = eight_cluster_item[3]

            y_spt_task_spt_set = eight_cluster_item[4]
            y_spt_task_qry_set = eight_cluster_item[5]
            y_qry_task_spt_set = eight_cluster_item[6]
            y_qry_task_qry_set = eight_cluster_item[7]

            # 初始化网络参数、优化器、计算图
            self.reset = False
            if self.reset:
                self.first_map.reset_map()
                self.first_optim = torch.optim.Adam(
                    self.first_map.parameters(), lr=self.middle_lr
                )

            # learning on T_S
            self.learning_on_the_high_level_task(
                "T_S",
                self.task_spt,
                x_spt_task_spt_set,
                x_spt_task_qry_set,
                y_spt_task_spt_set,
                y_spt_task_qry_set,
            )

            self.second_optim.zero_grad()
            q_loss_avg, q_acc_avg = self.learning_on_the_high_level_task(
                "T_Q",
                self.task_qry,
                x_qry_task_spt_set,
                x_qry_task_qry_set,
                y_qry_task_spt_set,
                y_qry_task_qry_set,
            )
            breakpoint()
            if train_or_test == "train":
                q_loss_avg.backward()
                self.second_optim.step()

            return q_loss_avg, q_acc_avg, 0
