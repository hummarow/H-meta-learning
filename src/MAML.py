import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import numpy as np
from learner import Learner
import torch.multiprocessing as mp
from threading import Thread
import math


class My_thread(Thread):
    def __init__(self, func, arg_list):
        Thread.__init__(self)
        self.arg_list = arg_list
        self.func = func

    def run(self):
        self.result = self.func(self.arg_list)

    def get_result(self):
        return self.result


def thread_wait_other_process(arg_list):
    # 等待结果返回
    ret_list = arg_list[0]
    for i, ret in enumerate(ret_list):
        now_grad, loss, correct = ret.get()
        if i == 0:
            total_grad = list(now_grad)
            loss_task = list(loss)
            correct_task = list(correct)
        else:
            for j, row in enumerate(now_grad):
                total_grad[j] += row
            for j, row in enumerate(loss):
                loss_task[j] += loss[j]
                correct_task[j] += correct[j]

    return total_grad, loss_task, correct_task


def thread_run_main_process(arg_list):
    for i in range(arg_list[-1]):
        now_grad, loss, correct = task_learning(
            arg_list[0],
            arg_list[1][i],
            arg_list[2][i],
            arg_list[3][i],
            arg_list[4][i],
            arg_list[5],
            arg_list[6],
            arg_list[7],
            arg_list[8],
            arg_list[9],
        )
        if i == 0:
            total_grad = list(now_grad)
            loss_task = list(loss)
            correct_task = list(correct)
        else:
            for j, row in enumerate(now_grad):
                total_grad[j] += row
            for j, row in enumerate(loss):
                loss_task[j] += loss[j]
                correct_task[j] += correct[j]

    return total_grad, loss_task, correct_task


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


def task_learning(
    net,
    x_spt,
    x_qry,
    y_spt,
    y_qry,
    bottom_lr,
    bottom_step_num,
    train_or_test,
    second_order,
    dataset,
):
    losses_q = [0 for _ in range(bottom_step_num + 1)]
    corrects_q = [0 for _ in range(bottom_step_num + 1)]
    logits = net(x_spt, vars=None, bn_training=True)
    loss = loss_func(logits, y_spt, dataset)
    grad = torch.autograd.grad(loss, net.parameters())
    fast_weights = list(
        map(lambda p: p[1] - bottom_lr * p[0], zip(grad, net.parameters()))
    )

    # this is the loss and accuracy before first update
    with torch.no_grad():
        logits_q = net(x_qry, net.parameters(), bn_training=True)
        loss_q = loss_func(logits_q, y_qry, dataset)
        losses_q[0] += loss_q
        correct = compute_correct(logits_q, y_qry)
        corrects_q[0] += correct

    # this is the loss and accuracy after first update
    with torch.no_grad():
        logits_q = net(x_qry, fast_weights, bn_training=True)
        loss_q = loss_func(logits_q, y_qry, dataset)
        losses_q[1] += loss_q
        correct = compute_correct(logits_q, y_qry)
        corrects_q[1] += correct

    # the following update
    for k in range(1, bottom_step_num):
        # 1. run the i-th task and compute loss for k=1~K-1
        logits = net(x_spt, fast_weights, bn_training=True)
        loss = loss_func(logits, y_spt, dataset)
        # 2. compute grad on theta_pi
        grad = torch.autograd.grad(loss, fast_weights, create_graph=bool(second_order))
        # 3. theta_pi = theta_pi - train_lr * grad
        fast_weights = list(
            map(lambda p: p[1] - bottom_lr * p[0], zip(grad, fast_weights))
        )

        logits_q = net(x_qry, fast_weights, bn_training=True)
        loss_q = loss_func(logits_q, y_qry, dataset)
        losses_q[k + 1] += loss_q
        correct = compute_correct(logits_q, y_qry)
        corrects_q[k + 1] += correct

    if train_or_test == "train":
        grads = torch.autograd.grad(losses_q[-1], net.parameters())
        return (
            [grad for grad in grads],
            [step_loss.detach().cpu().numpy() for step_loss in losses_q],
            corrects_q,
        )
    else:
        return (
            [0],
            [step_loss.detach().cpu().numpy() for step_loss in losses_q],
            corrects_q,
        )


class MAML(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, args):
        """

        :param args:
        """
        super(MAML, self).__init__()

        self.top_lr = args.top_lr
        self.bottom_lr = args.bottom_lr

        self.bottom_step_num = args.bottom_step_num
        self.bottom_step_num_test = args.bottom_step_num_test

        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.n_way = args.n_way

        self.net = Learner(args.config)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.top_lr)

        self.task_pool_processer = mp.Pool(args.train_parallel)
        self.train_parallel = args.train_parallel
        self.second_order = args.second_order
        self.dataset = args.dataset

        self.task_cluster_batch_num = args.task_cluster_batch_num
        self.task_cluster_batch_num_test = args.task_cluster_batch_num_test

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

    def forward(self, batch, train_or_test):
        """

        :param x_spt:   [b, setsz, c_, h, w] # num_clusters, num_images, c_, h, w
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        if train_or_test == "train":
            task_cluster_batch_num = self.task_cluster_batch_num
            bottom_step_num = self.bottom_step_num
        else:
        # elif train_or_test == "test":
            task_cluster_batch_num = self.task_cluster_batch_num_test
            bottom_step_num = self.bottom_step_num_test
        num_clusters = batch[0].shape[0]

        ret_list = []

        num_clusters = batch[0].shape[0]
        for i in range(num_clusters):
            ret_list.append(
                task_learning(
                    self.net,
                    batch[0][i],
                    batch[1][i],
                    batch[2][i],
                    batch[3][i],
                    self.bottom_lr,
                    bottom_step_num,
                    train_or_test,
                    self.second_order,
                    self.dataset,
                )
            )

        for i, ret in enumerate(ret_list):
            now_grad, loss, correct = ret
            if i == 0:
                total_grad = list(now_grad)
                total_loss = list(loss)
                total_correct = list(correct)
            else:
                for j, row in enumerate(now_grad):
                    total_grad[j] += row
                for j, row in enumerate(loss):
                    total_loss[j] += loss[j]
                    total_correct[j] += correct[j]

        if train_or_test == "train":
            self.meta_optim.zero_grad()
            for i in range(len(total_grad)):
                self.net.parameters()[i].grad = total_grad[i] / num_clusters

            self.meta_optim.step()

        loss_q = [step_loss / num_clusters for step_loss in total_loss]
        acc_q = [
            step_correct / (num_clusters * self.k_qry * self.n_way)
            for step_correct in total_correct
        ]

        return loss_q, acc_q, 0


def main():
    pass


if __name__ == "__main__":
    main()
