import argparse
import numpy as np
import os


class Options():

    def __init__(self):
        self.argparser = argparse.ArgumentParser()
        self.argparser.add_argument('--seed', type=int, help='random seed', default=222)
        self.argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
        self.argparser.add_argument('--start_epoch', type=str, help='from check point', default='no')

        # data point shot num
        self.argparser.add_argument('--n_way', type=int, help='n way', default=5)
        self.argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
        self.argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=5)
        self.argparser.add_argument('--k_qry_test', type=int, help='k shot for query set', default=5)

        # task shot num
        self.argparser.add_argument('--task_spt', type=int, help='k shot for support task', default=5)
        self.argparser.add_argument('--task_qry', type=int, help='k shot for query task', default=5)
        self.argparser.add_argument('--task_qry_test', type=int, help='k shot for query task', default=5)

        # task cluster batch num
        self.argparser.add_argument('--task_num', type=int, help='meta task size, namely task num',
                                    default=24)
        self.argparser.add_argument('--task_num_test', type=int, help='meta task size for test, namely task num test',
                                    default=24)
        self.argparser.add_argument('--task_cluster_batch_num', type=int,
                                    help='meta task batch size, namely task batch num',
                                    default=1)
        self.argparser.add_argument('--task_cluster_batch_num_test', type=int,
                                    help='meta task batch size, namely task batch num',
                                    default=1)

        # learning rate
        self.argparser.add_argument('--top_lr', type=float, help='meta-level outer learning rate', default=1e-3)
        self.argparser.add_argument('--middle_lr', type=float, help='meta-level outer learning rate', default=1e-3)
        self.argparser.add_argument('--bottom_lr', type=float, help='task-level inner update learning rate',
                                    default=1e-3)

        # retain second order and third order derivative
        self.argparser.add_argument('--second_order', type=int, help='compute Hessian matrix', default=0)
        self.argparser.add_argument('--third_order', type=int, help='compute partial derivative of the Hessian matrix',
                                    default=0)

        # reset H-ProtoNet's second_map
        self.argparser.add_argument('--reset_second_map', type=int,
                                    help='compute partial derivative of the Hessian matrix',
                                    default=1)

        # step num
        self.argparser.add_argument('--middle_step_num', type=int, help='task-batch-level inner update steps',
                                    default=2)
        self.argparser.add_argument('--bottom_step_num', type=int, help='task-level inner update steps', default=2)
        self.argparser.add_argument('--middle_step_num_test', type=int, help='task-batch-level inner update steps',
                                    default=2)
        self.argparser.add_argument('--bottom_step_num_test', type=int, help='task-level inner update steps', default=2)

        # range
        self.argparser.add_argument('--a_range', nargs='+', help='range of amplitude', default=(0.1, 10))
        self.argparser.add_argument('--p_range', nargs='+', help='range of phase', default=(0, 2 * np.pi))
        self.argparser.add_argument('--x_range', nargs='+', help='range of data point', default=(-5, 5))
        self.argparser.add_argument('--a_step_ratio', type=float, help='size of sub distribution of amplitude',
                                    default=0.1)
        self.argparser.add_argument('--p_step_ratio', type=float, help='size of sub distribution of phase', default=0.1)
        self.argparser.add_argument('--a_range_test', nargs='+', help='range of amplitude of tesing set',
                                    default=(0.1, 10))
        self.argparser.add_argument('--p_range_test', nargs='+', help='range of phase of tesing set',
                                    default=(0, 2 * np.pi))
        self.argparser.add_argument('--test_range_mode', type=str, help='type of test task range', default='auto')

        # learner, sample mode, scheme name
        self.argparser.add_argument('--method', type=str, help='method, e.g. MAML, Hierarchical_MAML',
                                    default='Hierarchical_MAML')
        self.argparser.add_argument('--sample_mode', type=str,
                                    help='sample mode, e.g. hierarchical_uniform, hierarchical_concentrate',
                                    default='hierarchical_uniform')
        self.argparser.add_argument('--test_spt_task_from', type=str, help='where are the test spt task from',
                                    default='meta_test')
        self.argparser.add_argument('--if_val', type=int, help='qry task for middle step (imitate testing set)',
                                    default=0)
        self.argparser.add_argument('--scheme_name', type=str, help='name of test scheme',
                                    default='test_scheme')

        # 下面是新加的对图片分类数据集进行切分的
        self.argparser.add_argument('--train_split', type=int, help='split index in the raw dataset for train and test',
                                    default=800)
        self.argparser.add_argument('--val_split', type=int, help='split index for validation',
                                    default=600)
        self.argparser.add_argument('--candidate_num', type=int, help='candidate num for hierarchical concentrate',
                                    default=10)

        # other
        self.argparser.add_argument('--gpu_id', type=str, help='id of gpu', default='0')
        self.argparser.add_argument('--dataset', type=str, help='dataset', default='omniglot')
        self.argparser.add_argument('--train_parallel', type=int, help='parallel number of training', default=0)

    def parse(self):
        # parse
        args = self.argparser.parse_args()
        args.a_range = (float(args.a_range[0]), float(args.a_range[1]))
        args.p_range = (float(args.p_range[0]), float(args.p_range[1]))
        args.x_range = (float(args.x_range[0]), float(args.x_range[1]))
        args.a_range_test = (float(args.a_range_test[0]), float(args.a_range_test[1]))
        args.p_range_test = (float(args.p_range_test[0]), float(args.p_range_test[1]))

        if args.dataset == 'omniglot':
            args.imgsz = 28
            args.channel = 1
        elif args.dataset == 'imagenet-1k' or 'miniimagenet':
            args.imgsz = 84
            args.channel = 3
        elif args.dataset == 'domainnet':
            args.imgsz = 224
            args.channel = 3

        if args.test_range_mode == 'auto':
            # 改变每个小任务分布的范围，看看啥情况
            args.a_range_test = args.a_range[1] * 1.1, args.a_range[1] * 1.1 + (
                    args.a_range[1] - args.a_range[0]) * args.a_step_ratio
            args.p_range_test = args.p_range[1] * 1.1, args.p_range[1] * 1.1 + (
                    args.p_range[1] - args.p_range[0]) * args.p_step_ratio
        if bool(args.if_val):
            assert args.test_spt_task_from == 'meta_train'

            args.a_range_train = args.a_range[0], args.a_range[0] + (args.a_range[1] - args.a_range[0]) * (
                    1 - args.a_step_ratio)
            args.p_range_train = args.p_range[0], args.p_range[0] + (args.p_range[1] - args.p_range[0]) * (
                    1 - args.p_step_ratio)

            args.a_range_val = args.a_range[0] + (args.a_range[1] - args.a_range[0]) * (1 - args.a_step_ratio), \
                               args.a_range[1]
            args.p_range_val = args.p_range[0] + (args.p_range[1] - args.p_range[0]) * (1 - args.p_step_ratio), \
                               args.p_range[1]

        if args.start_epoch != 'no':
            args.load_model_dir = os.path.join('../output/model/', args.dataset, args.scheme_name,
                                               args.scheme_name + '_' + str(args.start_epoch) + '.pth')
            args.scheme_name += '_load_from_' + args.start_epoch + '_epoch'
            args.seed += 1  # 不加这个1就相当于重新用第0个epoch的数据, 最好每用load checkpoint一次就+1

        # tensorboard, model, config dir
        args.tb_dir = os.path.join('../output/tensorboard/', args.dataset, args.scheme_name)
        args.model_dir = os.path.join('../output/model/', args.dataset, args.scheme_name)
        args.config_dir = os.path.join('../output/model/', args.dataset, args.scheme_name,
                                       args.scheme_name + '_config.txt')
        args.root = os.path.join('../data', args.dataset)

        if args.test_spt_task_from == 'meta_test':
            assert bool(args.if_val) is False

        # model architecture
        if args.dataset == 'omniglot':
            args.config = [
                ('conv2d', [64, 1, 3, 3, 2, 0]),
                ('relu', [True]),
                ('bn', [64]),
                ('conv2d', [64, 64, 3, 3, 2, 0]),
                ('relu', [True]),
                ('bn', [64]),
                ('conv2d', [64, 64, 3, 3, 2, 0]),
                ('relu', [True]),
                ('bn', [64]),
                ('conv2d', [64, 64, 2, 2, 1, 0]),
                ('relu', [True]),
                ('bn', [64]),
                ('flatten', []),
                ('linear', [args.n_way, 64])
            ]
        elif args.dataset == 'miniimagenet' or args.dataset == "imagenet-1k":
            args.config = [
                ('conv2d', [32, 3, 3, 3, 1, 0]),
                ('relu', [True]),
                ('bn', [32]),
                ('max_pool2d', [2, 2, 0]),
                ('conv2d', [32, 32, 3, 3, 1, 0]),
                ('relu', [True]),
                ('bn', [32]),
                ('max_pool2d', [2, 2, 0]),
                ('conv2d', [32, 32, 3, 3, 1, 0]),
                ('relu', [True]),
                ('bn', [32]),
                ('max_pool2d', [2, 2, 0]),
                ('conv2d', [32, 32, 3, 3, 1, 0]),
                ('relu', [True]),
                ('bn', [32]),
                ('max_pool2d', [2, 1, 0]),
                ('flatten', []),
                ('linear', [args.n_way, 32 * 5 * 5])
            ]
        elif args.dataset == 'domainnet':
            args.config = 1
        elif args.dataset == 'sinusoid':
            args.config = [
                ('flatten', [0, 0]),
                ('linear', [40, 1]),
                ('relu', [True]),
                ('linear', [40, 40]),
                ('relu', [True]),
                ('linear', [1, 40]),
            ]

        return args