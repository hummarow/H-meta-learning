import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch import optim
import numpy as np
from learner import Learner
import math
import torch.multiprocessing as mp
from threading import Thread
from pytorch_metric_learning import losses


class My_thread(Thread):
    def __init__(self, func, arg_dict):
        Thread.__init__(self)
        self.arg_dict = arg_dict
        self.func = func

    def run(self):
        self.result = self.func(self.arg_dict)

    def get_result(self):
        return self.result


def loss_func(y_pre, y_true, dataset):
    if dataset == "sinusoid":
        y_true = y_true.view(y_true.size(0), -1)
        return F.mse_loss(y_pre, y_true)
    else:
        return F.cross_entropy(y_pre, y_true)


def compute_correct(logits_q, y_true):
    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
    correct = torch.eq(pred_q, y_true).sum().item()
    return correct


# Contrastive step shouldn't be in the few-shot setting.
def contrastive_step(net, second_order, middle_weights, x_spt, x_qry, bottom_lr, bottom_step_num, reptile=False):
    # Choose either of them
    # A desirable way is not to use labels of images.
    # Therefore the self-supervised way is desired.
    nt_xent_loss = losses.NTXentLoss()
    self_supervised_loss = losses.SelfSupervisedLoss(nt_xent_loss)
    # # For simple experiment, choose ntxent loss.
    # loss_func = nt_xent_loss
    loss_func = self_supervised_loss

    org_spt, ref_spt = x_spt[0].squeeze(), x_spt[1].squeeze()
    org_qry, ref_qry = x_qry[0].squeeze(), x_qry[1].squeeze()
    org = torch.cat([org_spt, org_qry], dim=0)
    ref = torch.cat([ref_spt, ref_qry], dim=0)

    loss_q_list = [0 for _ in range(bottom_step_num + 1)]
    task_num = org.shape[0]

    fast_weights = middle_weights

    for bot_iter in range(bottom_step_num):
        org_emb = net.forward(org, fast_weights, bn_training=True, feature=True)
        ref_emb = net.forward(ref, fast_weights, bn_training=True, feature=True)
        loss = loss_func(org_emb, ref_emb)
        loss_q_list[bot_iter] += loss

        grad = torch.autograd.grad(loss, fast_weights, create_graph=bool(second_order), allow_unused=True)
        # grad can be shorter than fast_weights, so we need to pad it
        grad = [g if g is not None else torch.zeros_like(p) for g, p in zip(grad, fast_weights)]
        fast_weights = list(
            map(lambda p: p[1] - bottom_lr * p[0], zip(grad, fast_weights))
        )

    org_emb = net.forward(org, fast_weights, bn_training=True, feature=True)
    ref_emb = net.forward(ref, fast_weights, bn_training=True, feature=True)
    loss_q_list[-1] += loss_func(org_emb, ref_emb)

    # 根据loss_q计算梯度，更新middle model

    loss_q_list = [loss_q / task_num for loss_q in loss_q_list]
    # print(loss_q_list)
    if not reptile:
        fast_weights = None
    return loss_q_list, fast_weights


class H_MAML(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args):
        """

        :param args:
        """
        super(H_MAML, self).__init__()

        self.top_lr = args.top_lr
        self.middle_lr = args.middle_lr
        self.bottom_lr = args.bottom_lr

        self.second_order = args.second_order
        self.third_order = args.third_order

        self.middle_step_num = args.middle_step_num
        self.bottom_step_num = args.bottom_step_num
        self.middle_step_num_test = args.middle_step_num_test
        self.bottom_step_num_test = args.bottom_step_num_test
        self.contrastive_lr = args.contrastive_lr
        self.contrastive_step_num = args.contrastive_step_num
        self.contrastive_step_num_test = args.contrastive_step_num_test

        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.k_qry_test = args.k_qry_test
        self.n_way = args.n_way

        self.task_spt = args.task_spt
        self.task_qry = args.task_qry
        self.task_qry_test = args.task_qry_test

        self.task_cluster_batch_num = args.task_cluster_batch_num
        self.task_cluster_batch_num_test = args.task_cluster_batch_num_test

        self.net = Learner(args.config)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.top_lr)
        self.train_parallel = args.train_parallel
        if args.train_parallel != 0:
            self.task_pool_processer = mp.Pool(args.train_parallel)

        self.dataset = args.dataset

        self.contrastive = args.contrastive
        self.reptile = args.reptile

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1.0 / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm / counter

    def thread_wait_other_process(self, arg_dict):
        # 初始化top model在query task上的平均损失(详细来说，是所有task cluster的中所有query task上的query set的损失的均值)
        loss_q_batch_list = [
            [0 for _ in range(arg_dict["bottom_step_num"] + 1)]
            for _ in range(arg_dict["middle_step_num"] + 1)
        ]
        correct_q_batch_list = [
            [0 for _ in range(arg_dict["bottom_step_num"] + 1)]
            for _ in range(arg_dict["middle_step_num"] + 1)
        ]
        mid_grad_list = [[] for _ in range(arg_dict["middle_step_num"])]

        # 等待结果返回
        for i, ret in enumerate(arg_dict["ret_list"]):
            now_grad, loss_cluster, correct_cluster, mid_grad = ret.get()
            if i == 0:
                total_grad = list(now_grad)

                for k in range(arg_dict["middle_step_num"]):
                    mid_grad_list[k] = list(mid_grad[k])
            else:
                for j, row in enumerate(now_grad):
                    total_grad[j] += row

                for k in range(arg_dict["middle_step_num"]):
                    for j, row in enumerate(mid_grad[k]):
                        mid_grad_list[k][j] += mid_grad[k][j]

            for k in range(arg_dict["middle_step_num"] + 1):
                for t in range(arg_dict["bottom_step_num"] + 1):
                    loss_q_batch_list[k][t] += loss_cluster[k][t]
                    correct_q_batch_list[k][t] += correct_cluster[k][t]

        return total_grad, loss_q_batch_list, correct_q_batch_list, mid_grad_list

    def thread_run_main_process(self, arg_dict):
        loss_q_batch_list = [
            [0 for _ in range(arg_dict["bottom_step_num"] + 1)]
            for _ in range(arg_dict["middle_step_num"] + 1)
        ]
        correct_q_batch_list = [
            [0 for _ in range(arg_dict["bottom_step_num"] + 1)]
            for _ in range(arg_dict["middle_step_num"] + 1)
        ]
        mid_grad_list = [[] for _ in range(arg_dict["middle_step_num"])]
        batch = arg_dict["eight_cluster_batch"]
        self.cluster_num = batch[0].shape[0]
        for i in range(self.cluster_num):
            batch_per_cluster = [batch[j][i] for j in range(len(batch))]
            now_grad, loss_cluster, correct_cluster, mid_grad = self.cluster_learning(
                arg_dict["net"],
                batch_per_cluster,
                arg_dict["middle_lr"],
                arg_dict["bottom_lr"],
                arg_dict["middle_step_num"],
                arg_dict["bottom_step_num"],
                arg_dict["train_or_test"],
                arg_dict["second_order"],
                arg_dict["third_order"],
                arg_dict["dataset"],
                arg_dict["contrastive"],
            )
            if i == 0:
                total_grad = list(now_grad)

                for k in range(arg_dict["middle_step_num"]):
                    mid_grad_list[k] = list(mid_grad[k])
            else:
                for j, row in enumerate(now_grad):
                    total_grad[j] += row

                for k in range(arg_dict["middle_step_num"]):
                    for j, row in enumerate(mid_grad[k]):
                        mid_grad_list[k][j] += mid_grad[k][j]

            for k in range(arg_dict["middle_step_num"] + 1):
                for t in range(arg_dict["bottom_step_num"] + 1):
                    loss_q_batch_list[k][t] += loss_cluster[k][t]
                    correct_q_batch_list[k][t] += correct_cluster[k][t]

        return total_grad, loss_q_batch_list, correct_q_batch_list, mid_grad_list

    def forward(self, batch, mode):
        """
        :param x_task_cluster_batch:   [task_cluster_batch_num, task_spt+task_qry, k_spt+k_qry]
        :param y_task_cluster_batch:   [task_cluster_batch_num, task_spt+task_qry, k_spt+k_qry]
        :param train_or_test: train or test
        :return:
        """

        """
        self.net 是 meta model，也叫top model，负责处理多个批次任务的信息
        middle model，负责处理单个批次任务的信息
        bottom model，负责处理单个任务的信息
        """

        # 区分train和task的训练测试数据
        if mode == "train":
            task_cluster_batch_num = self.task_cluster_batch_num
            middle_step_num = self.middle_step_num
            bottom_step_num = self.bottom_step_num
            task_qry_num = self.task_qry
            k_qry_num = self.k_qry
        else:
            task_cluster_batch_num = self.task_cluster_batch_num_test
            middle_step_num = self.middle_step_num_test
            bottom_step_num = self.bottom_step_num_test
            task_qry_num = self.task_qry_test
            k_qry_num = self.k_qry_test

        # 下来多个 task cluster
        num_tasks_per_process = math.ceil(
            task_cluster_batch_num / (self.train_parallel + 1)
        )
        if self.train_parallel != 0:
            ret_list = []
            # 这里的多个task batch是SGD意义上的mini batch，因为显存不够大，而不是划分了任务
            # 提交任务给其它进程处理
            for i in range(num_tasks_per_process, task_cluster_batch_num):
                # 单个batch的八项数据数据（包含support task 和 query task，spt set 和 qry set，x 和 y）
                batch_per_process = [batch[j][i] for j in range(8)]

                ret_list.append(
                    self.task_pool_processer.apply_async(
                        self.cluster_learning,
                        (
                            self.net,
                            batch_per_process,
                            self.middle_lr,
                            self.bottom_lr,
                            middle_step_num,
                            bottom_step_num,
                            mode,
                            self.second_order,
                            self.third_order,
                            self.dataset,
                            self.contrastive,
                        ),
                    )
                )

            # 主进程别闲着
            main_process = My_thread(
                self.thread_run_main_process,
                {
                    "net": self.net,
                    "eight_cluster_batch": [
                        batch[j][:num_tasks_per_process] for j in range(8)
                    ],
                    "middle_lr": self.middle_lr,
                    "bottom_lr": self.bottom_lr,
                    "middle_step_num": middle_step_num,
                    "bottom_step_num": bottom_step_num,
                    "train_or_test": mode,
                    "second_order": self.second_order,
                    "third_order": self.third_order,
                    "dataset": self.dataset,
                    "per_process_do_num": num_tasks_per_process,
                    "contrastive": self.contrastive,
                },
            )
            # 收集其它进程结果
            other_process = My_thread(
                self.thread_wait_other_process,
                {
                    "ret_list": ret_list,
                    "middle_step_num": middle_step_num,
                    "bottom_step_num": bottom_step_num,
                },
            )

            main_process.start()
            other_process.start()
            main_process.join()
            other_process.join()

            (
                main_grad,
                main_loss_list,
                main_correct_list,
                main_mid_grad_list,
            ) = main_process.get_result()
            (
                other_grad,
                other_loss_list,
                other_correct_list,
                other_mid_grad_list,
            ) = other_process.get_result()

            # 合并主进程与其它进程的返回结果
            for i, row in enumerate(other_grad):
                main_grad[i] += row
            total_grad = main_grad
            loss_q_batch_list = [
                [
                    (main_loss + other_loss) / task_cluster_batch_num
                    for main_loss, other_loss in zip(main_loss_task, other_loss_task)
                ]
                for main_loss_task, other_loss_task in zip(
                    main_loss_list, other_loss_list
                )
            ]

            acc_q_batch_list = [
                [
                    (main_correct + other_correct)
                    / (
                        # task_cluster_batch_num * task_qry_num * self.n_way * k_qry_num
                        self.cluster_num
                        * self.n_way
                        * self.k_qry
                    )
                    for main_correct, other_correct in zip(
                        main_correct_task, other_correct_task
                    )
                ]
                for main_correct_task, other_correct_task in zip(
                    main_correct_list, other_correct_list
                )
            ]

            mid_grad_batch_list = [
                [
                    (main_mid_row + other_mid_row) / task_cluster_batch_num
                    for main_mid_row, other_mid_row in zip(
                        main_mid_grad, other_mid_grad
                    )
                ]
                for main_mid_grad, other_mid_grad in zip(
                    main_mid_grad_list, other_mid_grad_list
                )
            ]

        else:
            (
                main_grad,
                main_loss_list,
                main_correct_list,  # middle_step_num * bottom_step_num
                main_mid_grad_list,
            ) = self.thread_run_main_process(
                {
                    "net": self.net,
                    "eight_cluster_batch": batch,
                    "middle_lr": self.middle_lr,
                    "bottom_lr": self.bottom_lr,
                    "middle_step_num": middle_step_num,
                    "bottom_step_num": bottom_step_num,
                    "train_or_test": mode,
                    "second_order": self.second_order,
                    "third_order": self.third_order,
                    "dataset": self.dataset,
                    "per_process_do_num": num_tasks_per_process,
                    "contrastive": self.contrastive,
                }
            )

            total_grad = main_grad
            loss_q_batch_list = [
                [main_loss / task_cluster_batch_num for main_loss in main_loss_task]
                for main_loss_task in main_loss_list
            ]
            # task_qry_num = self.task_qry_test
            # k_qry_num = self.k_qry_test
            acc_q_batch_list = [
                [
                    main_correct
                    / (
                        # task_cluster_batch_num * task_qry_num * self.n_way * k_qry_num
                        self.cluster_num
                        * self.n_way
                        * self.k_qry
                    )  # task_cluster_batch_num * task_qry_num * N * QRY
                    for main_correct in main_correct_task
                ]
                for main_correct_task in main_correct_list
            ]

            mid_grad_batch_list = [
                [
                    main_mid_row / task_cluster_batch_num
                    for main_mid_row in main_mid_grad
                ]
                for main_mid_grad in main_mid_grad_list
            ]

        if mode == "train":
            self.meta_optim.zero_grad()
            for i in range(len(total_grad)):
                self.net.vars[i].grad = total_grad[i] / task_cluster_batch_num
            self.meta_optim.step()
        # breakpoint()
        return (
            [
                [loss_q_batch.cpu().numpy() for loss_q_batch in loss_q_batch_task]
                for loss_q_batch_task in loss_q_batch_list
            ],
            acc_q_batch_list,
            [
                [mid_grad_row.cpu().numpy() for mid_grad_row in mid_grad]
                for mid_grad in mid_grad_batch_list
            ],
        )

    def cluster_learning(
        self,
        net,
        batch,
        middle_lr,
        bottom_lr,
        middle_step_num,
        bottom_step_num,
        train_or_test,
        second_order,
        third_order,
        dataset,
        contrastive: bool = False,
    ):
        def middle_step(middle_weights, x_spt, y_spt, x_qry, y_qry, reptile=False):
            loss_q_list = [0 for _ in range(bottom_step_num + 1)]
            correct_q_list = [0 for _ in range(bottom_step_num + 1)]
            task_num = len(x_spt)

            # for i in range(task_num):
            # 复制bottom model
            fast_weights = middle_weights

            # 计算不更新时的loss和acc
            # logits_q = net.forward(x_qry[i], fast_weights, bn_training=True)
            logits_q = net.forward(x_qry, fast_weights, bn_training=True)
            loss_q_list[0] += loss_func(logits_q, y_qry, dataset)
            correct_q_list[0] += compute_correct(logits_q, y_qry)
            # inner(bottom) loop
            # bottom_model走bottom_step步，到达适合该support task batch下所有task的公共最优点，等价于MAML的公共最优点
            for bot_iter in range(bottom_step_num):
                logits = net.forward(x_spt, fast_weights, bn_training=True)
                loss = loss_func(logits, y_spt, dataset)

                # create_graph为计算出的grad添加计算图，可以传递到后面fast_weight上
                grad = torch.autograd.grad(
                    loss, fast_weights, create_graph=bool(second_order)
                )

                # list(map) 新开了内存存权重和计算路径，所以改动fast_weight不改变middle_weight，但是计算操作会记录到logits_q上
                fast_weights = list(
                    map(lambda p: p[1] - bottom_lr * p[0], zip(grad, fast_weights))
                )

                logits_q = net.forward(x_qry, fast_weights, bn_training=True)
                loss_q_list[bot_iter + 1] += loss_func(logits_q, y_qry, dataset)
                correct_q_list[bot_iter + 1] += compute_correct(logits_q, y_qry)

            # 根据loss_q计算梯度，更新middle model

            loss_q_list = [loss_q / task_num for loss_q in loss_q_list]
            # print(f'{train_or_test}: {y_qry.shape}')
            if not reptile:
                fast_weights = None
            return loss_q_list, correct_q_list, fast_weights

        (
            ST_x_spt,
            ST_x_qry,
            QT_x_spt,
            QT_x_qry,
            y_spt,
            y_qry,
        ) = batch

        # 复制middle model
        middle_weights = net.vars

        # 记录中间几步的loss和acc
        loss_list = []
        correct_list = []
        mid_grad_list = []

        # 开始前计算一下q_task上的loss和acc
        loss_q_task, correct_q_task, _ = middle_step(
            middle_weights,
            QT_x_spt,
            y_spt,
            QT_x_qry,
            y_qry,
        )
        correct_list.append(correct_q_task)
        loss_list.append(loss_q_task)

        # middle loop
        for mid_iter in range(middle_step_num):
            if contrastive:
                loss_s_task, fast_weights = contrastive_step(
                    net,
                    second_order,
                    middle_weights,
                    ST_x_spt,
                    ST_x_qry,
                    bottom_lr=self.contrastive_lr,
                    bottom_step_num=self.contrastive_step_num if train_or_test=="train" else self.contrastive_step_num_test,
                    reptile=self.reptile,
                )
            else:
                loss_s_task, correct_s_task, fast_weights = middle_step(
                    middle_weights,
                    ST_x_spt,
                    y_spt,
                    ST_x_qry,
                    y_qry,
                    reptile=self.reptile,
                )

            middle_grad = torch.autograd.grad(
                loss_s_task[-1], middle_weights, create_graph=bool(third_order), allow_unused=True
            )
            middle_grad = [g if g is not None else torch.zeros_like(p) for g, p in zip(middle_grad, middle_weights)]

            middle_weights = list(
                map(lambda p: p[1] - middle_lr * p[0], zip(middle_grad, middle_weights))
            )

            # 每次middle_step后计算middle_weights在task_qry上的性能
            loss_q_task, correct_q_task, _ = middle_step(
                middle_weights,
                QT_x_spt,
                y_spt,
                QT_x_qry,
                y_qry,
            )
            loss_list.append(loss_q_task)
            correct_list.append(correct_q_task)
            mid_grad_list.append(middle_grad)
        # 处理query batch
        # 初始化middle model在该task batch中的support task上的总损失
        # （详细来说，是该task batch下所有support task的query set的损失）
        loss_q_task, correct_q_task = loss_list[-1][-1], correct_list[-1][-1]
        loss_ret = [[loss.detach() for loss in loss_task] for loss_task in loss_list]
        mid_grad_ret = [
            [mid_grad_row.detach() for mid_grad_row in mid_grad]
            for mid_grad in mid_grad_list
        ]
        if train_or_test == "train":
            if self.reptile:
                top_grad = [fast_weights[i] - self.net.vars[i] for i in range(len(fast_weights))]
            else:
                top_grad = torch.autograd.grad(loss_q_task, net.vars)
        else:
            top_grad = [0]

        return top_grad, loss_ret, correct_list, mid_grad_ret
    

def main():
    pass


if __name__ == "__main__":
    main()
