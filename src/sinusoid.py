import random
import numpy as np


class sinusoid():
    def __init__(self, args):

        np.random.seed(args.seed)
        random.seed(args.seed)

        self.a_range = args.a_range
        self.p_range = args.p_range
        self.x_range = args.x_range
        self.a_step_ratio = args.a_step_ratio
        self.p_step_ratio = args.p_step_ratio
        self.a_range_test = args.a_range_test
        self.p_range_test = args.p_range_test

        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.k_qry_test = args.k_qry_test

        self.task_spt = args.task_spt
        self.task_qry = args.task_qry
        self.task_qry_test = args.task_qry_test

        self.task_num = args.task_num
        self.task_num_test = args.task_num_test
        self.task_cluster_batch_num = args.task_cluster_batch_num
        self.task_cluster_batch_num_test = args.task_cluster_batch_num_test

        self.test_spt_task_from = args.test_spt_task_from
        self.if_val = args.if_val

        if bool(self.if_val):
            self.a_range_train = args.a_range_train
            self.p_range_train = args.p_range_train
            self.a_range_val = args.a_range_val
            self.p_range_val = args.p_range_val

    def construct_task(self, task_num, a_range, p_range, k_spt, k_qry):
        x_spt_task_spt_set_cluster = np.zeros((0, k_spt))
        x_spt_task_qry_set_cluster = np.zeros((0, k_qry))
        y_spt_task_spt_set_cluster = np.zeros((0, k_spt))
        y_spt_task_qry_set_cluster = np.zeros((0, k_qry))

        for j in range(task_num):
            amplitude = np.random.uniform(a_range[0], a_range[1])
            phase = np.random.uniform(p_range[0], p_range[1])

            x_spt = np.random.uniform(self.x_range[0], self.x_range[1], k_spt)
            x_qry = np.random.uniform(self.x_range[0], self.x_range[1], k_qry)
            y_spt = [amplitude * np.sin(x + phase) for x in x_spt]
            y_qry = [amplitude * np.sin(x + phase) for x in x_qry]

            x_spt_task_spt_set_cluster = np.vstack((x_spt_task_spt_set_cluster, x_spt))
            x_spt_task_qry_set_cluster = np.vstack((x_spt_task_qry_set_cluster, x_qry))
            y_spt_task_spt_set_cluster = np.vstack((y_spt_task_spt_set_cluster, y_spt))
            y_spt_task_qry_set_cluster = np.vstack((y_spt_task_qry_set_cluster, y_qry))

        return x_spt_task_spt_set_cluster, x_spt_task_qry_set_cluster, y_spt_task_spt_set_cluster, y_spt_task_qry_set_cluster

    def next_uniform(self, mode):
        # 训练集任务随机均匀取自同一任务分布，测试集取自另一分布（或者与训练集相同）
        # 训练集形成了单级层级关系
        x_range = self.x_range
        if mode == 'train':
            a_range = self.a_range
            p_range = self.p_range
            k_qry = self.k_qry
            task_num = self.task_num

        elif mode == 'test':
            a_range = self.a_range_test
            p_range = self.p_range_test
            k_qry = self.k_qry_test
            task_num = self.task_num_test
        else:
            raise NotImplementedError

        x_spt_batch = np.zeros((0, self.k_spt))
        y_spt_batch = np.zeros((0, self.k_spt))
        x_qry_batch = np.zeros((0, k_qry))
        y_qry_batch = np.zeros((0, k_qry))

        for i in range(task_num):
            amplitude = np.random.uniform(a_range[0], a_range[1])
            phase = np.random.uniform(p_range[0], p_range[1])

            x_spt = np.random.uniform(x_range[0], x_range[1], self.k_spt)
            y_spt = [amplitude * np.sin(x + phase) for x in x_spt]
            x_qry = np.random.uniform(x_range[0], x_range[1], k_qry)
            y_qry = [amplitude * np.sin(x + phase) for x in x_qry]

            x_spt_batch = np.vstack((x_spt_batch, x_spt))
            y_spt_batch = np.vstack((y_spt_batch, y_spt))
            x_qry_batch = np.vstack((x_qry_batch, x_qry))
            y_qry_batch = np.vstack((y_qry_batch, y_qry))

        return x_spt_batch, y_spt_batch, x_qry_batch, y_qry_batch

    def next_single_batch_domain(self, mode):
        # 每个batch的训练集任务随机均匀取自训练集总体分布的一个连续的子分布，测试集取自另一分布
        # 训练集形成了两级层级关系：每个task batch中的task属于一个task子分布，每个task内的data point属于该子分布的一个data子分布
        x_range = self.x_range
        if mode == 'train':
            # 计算子分布的长度
            a_step = (self.a_range[1] - self.a_range[0]) * self.a_step_ratio
            p_step = (self.p_range[1] - self.p_range[0]) * self.p_step_ratio

            # 随机选取子分部的起始点
            a_range_begin = np.random.uniform(self.a_range[0], self.a_range[1] - a_step)
            p_range_begin = np.random.uniform(self.p_range[0], self.p_range[1] - p_step)

            # 获得子分部范围
            a_range = a_range_begin, a_range_begin + a_step
            p_range = p_range_begin, p_range_begin + p_step

            task_num = self.task_num
            k_qry = self.k_qry

        elif mode == 'test':
            a_range = self.a_range_test
            p_range = self.p_range_test
            task_num = self.task_num_test
            k_qry = self.k_qry_test
        else:
            raise NotImplementedError

        x_spt_batch = np.zeros((0, self.k_spt))
        y_spt_batch = np.zeros((0, self.k_spt))
        x_qry_batch = np.zeros((0, k_qry))
        y_qry_batch = np.zeros((0, k_qry))

        for i in range(task_num):
            amplitude = np.random.uniform(a_range[0], a_range[1])
            phase = np.random.uniform(p_range[0], p_range[1])

            x_spt = np.random.uniform(x_range[0], x_range[1], self.k_spt)
            y_spt = [amplitude * np.sin(x + phase) for x in x_spt]
            x_qry = np.random.uniform(x_range[0], x_range[1], k_qry)
            y_qry = [amplitude * np.sin(x + phase) for x in x_qry]

            x_spt_batch = np.vstack((x_spt_batch, x_spt))
            y_spt_batch = np.vstack((y_spt_batch, y_spt))
            x_qry_batch = np.vstack((x_qry_batch, x_qry))
            y_qry_batch = np.vstack((y_qry_batch, y_qry))

        return x_spt_batch, y_spt_batch, x_qry_batch, y_qry_batch

    def next_hierarchical_concentrate(self, mode):
        k_spt = self.k_spt
        task_spt=self.task_spt
        if mode == 'train':
            k_qry=self.k_qry
            task_qry=self.task_qry
            task_cluster_batch_num=self.task_cluster_batch_num
        else:
            k_qry = self.k_qry_test
            task_qry = self.task_qry_test
            task_cluster_batch_num = self.task_cluster_batch_num_test

        x_spt_task_spt_set_batch = np.zeros((0, task_spt, k_spt))
        x_spt_task_qry_set_batch = np.zeros((0, task_spt, k_qry))
        x_qry_task_spt_set_batch = np.zeros((0, task_qry, k_spt))
        x_qry_task_qry_set_batch = np.zeros((0, task_qry, k_qry))
        y_spt_task_spt_set_batch = np.zeros((0, task_spt, k_spt))
        y_spt_task_qry_set_batch = np.zeros((0, task_spt, k_qry))
        y_qry_task_spt_set_batch = np.zeros((0, task_qry, k_spt))
        y_qry_task_qry_set_batch = np.zeros((0, task_qry, k_qry))

        for i in range(self.task_cluster_batch_num):
            if mode == 'train':
                a_step = (self.a_range[1] - self.a_range[0]) * self.a_step_ratio
                p_step = (self.p_range[1] - self.p_range[0]) * self.p_step_ratio
                a_range_begin = np.random.uniform(self.a_range[0], self.a_range[1] - a_step)
                p_range_begin = np.random.uniform(self.p_range[0], self.p_range[1] - p_step)

                a_range = a_range_begin, a_range_begin + a_step
                p_range = p_range_begin, p_range_begin + p_step
            else:
                a_range = self.a_range_test
                p_range = self.p_range_test

            # 这里有两种方案：1. spt task和qry task在分布上不相交. 2. spt task和qry task i.i.d的采样，此处实现第二种
            # support task and query task
            x_spt_task_spt_set_cluster, x_spt_task_qry_set_cluster, y_spt_task_spt_set_cluster, y_spt_task_qry_set_cluster = self.construct_task(
                task_spt, a_range, p_range, k_spt, k_qry)

            x_qry_task_spt_set_cluster, x_qry_task_qry_set_cluster, y_qry_task_spt_set_cluster, y_qry_task_qry_set_cluster = self.construct_task(
                task_qry, a_range, p_range, k_spt, k_qry)

            x_spt_task_spt_set_cluster = np.expand_dims(x_spt_task_spt_set_cluster, axis=0)
            x_spt_task_qry_set_cluster = np.expand_dims(x_spt_task_qry_set_cluster, axis=0)
            x_qry_task_spt_set_cluster = np.expand_dims(x_qry_task_spt_set_cluster, axis=0)
            x_qry_task_qry_set_cluster = np.expand_dims(x_qry_task_qry_set_cluster, axis=0)
            y_spt_task_spt_set_cluster = np.expand_dims(y_spt_task_spt_set_cluster, axis=0)
            y_spt_task_qry_set_cluster = np.expand_dims(y_spt_task_qry_set_cluster, axis=0)
            y_qry_task_spt_set_cluster = np.expand_dims(y_qry_task_spt_set_cluster, axis=0)
            y_qry_task_qry_set_cluster = np.expand_dims(y_qry_task_qry_set_cluster, axis=0)

            x_spt_task_spt_set_batch = np.vstack((x_spt_task_spt_set_batch, x_spt_task_spt_set_cluster))
            x_spt_task_qry_set_batch = np.vstack((x_spt_task_qry_set_batch, x_spt_task_qry_set_cluster))
            x_qry_task_spt_set_batch = np.vstack((x_qry_task_spt_set_batch, x_qry_task_spt_set_cluster))
            x_qry_task_qry_set_batch = np.vstack((x_qry_task_qry_set_batch, x_qry_task_qry_set_cluster))
            y_spt_task_spt_set_batch = np.vstack((y_spt_task_spt_set_batch, y_spt_task_spt_set_cluster))
            y_spt_task_qry_set_batch = np.vstack((y_spt_task_qry_set_batch, y_spt_task_qry_set_cluster))
            y_qry_task_spt_set_batch = np.vstack((y_qry_task_spt_set_batch, y_qry_task_spt_set_cluster))
            y_qry_task_qry_set_batch = np.vstack((y_qry_task_qry_set_batch, y_qry_task_qry_set_cluster))

        return [x_spt_task_spt_set_batch, x_spt_task_qry_set_batch, x_qry_task_spt_set_batch, x_qry_task_qry_set_batch,
                y_spt_task_spt_set_batch, y_spt_task_qry_set_batch, y_qry_task_spt_set_batch, y_qry_task_qry_set_batch]

    def next_hierarchical_uniform(self, mode):
        task_spt = self.task_spt
        k_spt = self.k_spt
        if mode == 'train':
            task_cluster_batch_num = self.task_cluster_batch_num
            task_qry = self.task_spt
            k_qry = self.k_qry
            if bool(self.if_val):
                a_spt_range = self.a_range_train
                p_spt_range = self.p_range_train
                a_qry_range = self.a_range_val
                p_qry_range = self.p_range_val
            else:
                a_spt_range = a_qry_range = self.a_range
                p_spt_range = p_qry_range = self.p_range
        else:
            task_cluster_batch_num = self.task_cluster_batch_num_test
            task_qry = self.task_qry_test
            k_qry = self.k_qry_test
            if bool(self.if_val):
                a_spt_range = self.a_range_train
                p_spt_range = self.p_range_train
                a_qry_range = self.a_range_test
                p_qry_range = self.p_range_test
            else:
                a_spt_range = a_qry_range = self.a_range_test
                p_spt_range = p_qry_range = self.p_range_test

        x_spt_task_spt_set_batch = np.zeros((0, task_spt, k_spt))
        x_spt_task_qry_set_batch = np.zeros((0, task_spt, k_qry))
        x_qry_task_spt_set_batch = np.zeros((0, task_qry, k_spt))
        x_qry_task_qry_set_batch = np.zeros((0, task_qry, k_qry))
        y_spt_task_spt_set_batch = np.zeros((0, task_spt, k_spt))
        y_spt_task_qry_set_batch = np.zeros((0, task_spt, k_qry))
        y_qry_task_spt_set_batch = np.zeros((0, task_qry, k_spt))
        y_qry_task_qry_set_batch = np.zeros((0, task_qry, k_qry))
        for i in range(task_cluster_batch_num):
            # support task and query task
            x_spt_task_spt_set_cluster, x_spt_task_qry_set_cluster, y_spt_task_spt_set_cluster, y_spt_task_qry_set_cluster = self.construct_task(
                self.task_spt, a_spt_range, p_spt_range, k_spt, k_qry)
            x_qry_task_spt_set_cluster, x_qry_task_qry_set_cluster, y_qry_task_spt_set_cluster, y_qry_task_qry_set_cluster = self.construct_task(
                self.task_qry, a_qry_range, p_qry_range, k_spt, k_qry)

            x_spt_task_spt_set_cluster = np.expand_dims(x_spt_task_spt_set_cluster, axis=0)
            x_spt_task_qry_set_cluster = np.expand_dims(x_spt_task_qry_set_cluster, axis=0)
            x_qry_task_spt_set_cluster = np.expand_dims(x_qry_task_spt_set_cluster, axis=0)
            x_qry_task_qry_set_cluster = np.expand_dims(x_qry_task_qry_set_cluster, axis=0)
            y_spt_task_spt_set_cluster = np.expand_dims(y_spt_task_spt_set_cluster, axis=0)
            y_spt_task_qry_set_cluster = np.expand_dims(y_spt_task_qry_set_cluster, axis=0)
            y_qry_task_spt_set_cluster = np.expand_dims(y_qry_task_spt_set_cluster, axis=0)
            y_qry_task_qry_set_cluster = np.expand_dims(y_qry_task_qry_set_cluster, axis=0)

            x_spt_task_spt_set_batch = np.vstack((x_spt_task_spt_set_batch, x_spt_task_spt_set_cluster))
            x_spt_task_qry_set_batch = np.vstack((x_spt_task_qry_set_batch, x_spt_task_qry_set_cluster))
            x_qry_task_spt_set_batch = np.vstack((x_qry_task_spt_set_batch, x_qry_task_spt_set_cluster))
            x_qry_task_qry_set_batch = np.vstack((x_qry_task_qry_set_batch, x_qry_task_qry_set_cluster))
            y_spt_task_spt_set_batch = np.vstack((y_spt_task_spt_set_batch, y_spt_task_spt_set_cluster))
            y_spt_task_qry_set_batch = np.vstack((y_spt_task_qry_set_batch, y_spt_task_qry_set_cluster))
            y_qry_task_spt_set_batch = np.vstack((y_qry_task_spt_set_batch, y_qry_task_spt_set_cluster))
            y_qry_task_qry_set_batch = np.vstack((y_qry_task_qry_set_batch, y_qry_task_qry_set_cluster))

        return [x_spt_task_spt_set_batch, x_spt_task_qry_set_batch, x_qry_task_spt_set_batch,
                x_qry_task_qry_set_batch,
                y_spt_task_spt_set_batch, y_spt_task_qry_set_batch, y_qry_task_spt_set_batch,
                y_qry_task_qry_set_batch]
