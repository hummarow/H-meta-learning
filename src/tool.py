import os
import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

class Tensorboard_manager:
    def __init__(self, logdir):
        import os

#         if not os.path.exists(logdir):
#             os.mkdir(logdir)
#         # Remove all files
#         for file in os.listdir(logdir):
#             os.remove(os.path.join(logdir, file))
        # Create a new directory instead.
        Path(logdir).mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.rmtree(logdir)

        self.summary_writer = SummaryWriter(logdir, flush_secs=10)

    def update_graph(self, epoch, report_dict=None):
        if report_dict != None:
            for key in report_dict:
                if type(report_dict[key]) == type({}):
                    for sub_key in report_dict[key]:
                        self.summary_writer.add_scalar(key + '_' + sub_key,
                                                       report_dict[key][sub_key], global_step=epoch)
                else:
                    self.summary_writer.add_scalar(key, report_dict[key], global_step=epoch)


def change_tfevent_tag_name(file):
    import tensorflow
    import os
    from tensorflow.python.summary.summary_iterator import summary_iterator

    summary_writer = SummaryWriter(file[:file.rfind('/')], flush_secs=10)
    for event in summary_iterator(file):
        for value in event.summary.value:
            if value.HasField('simple_value'):
                summary_writer.add_scalar('loss', value.simple_value, global_step=event.step)


def reshape_loss_curve(file):
    # 观测多条损失曲线时，曲线波动导致难以区分性能，这里将曲线整形成单调递减的形式，便于比较收敛速度、收敛点
    import tensorflow
    import os
    from tensorflow.python.summary.summary_iterator import summary_iterator

    summary_writer = SummaryWriter(file[:file.rfind('/')], flush_secs=10)
    assert len(summary_iterator) == 1  # 以防把loss之外的东西搞坏了
    minimal_value = 10000000
    for event in summary_iterator(file):
        for value in event.summary.value:
            if value.HasField('simple_value'):
                if value.simple_value < minimal_value:
                    minimal_value = value.simple_value

                summary_writer.add_scalar('decrease_loss', minimal_value, global_step=event.step)


def load_network(method, load_model_dir, device):
    model = torch.load(load_model_dir, map_location=device)
    method.net = model.net
    method.meta_optim = model.meta_optim
    return method


def save_network(model, save_dir, args):
    # save_epoch = save_dir.split('_', -1)[-1][:-4]
    # model_dir = os.path.dirname(save_dir)
    # model_pth_list = os.listdir(model_dir)
    # for model_pth in model_pth_list:
    #     if 'pth' not in model_pth:
    #         continue
    #     now_epoch = model_pth.split('_', -1)[-1][:-4]
    #     if save_epoch <= now_epoch:  # 防止从中间加载的的模型继续训练覆盖后面已经保存的模型
    #         return

    if args.train_parallel != 0:
        tmp_process_pool = model.task_pool_processer
        del model.task_pool_processer
    torch.save(model, save_dir)
    if args.train_parallel != 0:
        model.task_pool_processer = tmp_process_pool


def redraw_step_acc(file):
    from tensorflow.python.summary.summary_iterator import summary_iterator
    import matplotlib.pyplot as plt
    from fnmatch import fnmatch
    # assert len(summary_iterator) == 1  # 以防把loss之外的东西搞坏了

    bottom_step_0 = []
    bottom_step_1 = []
    bottom_step_2 = []

    train_bottom_step_0 = []
    train_bottom_step_1 = []
    train_bottom_step_2 = []

    data_dict_acc = {'train_acc_0_0': [],
                     'train_acc_0_1': [],
                     'train_acc_0_2': [],
                     'train_acc_1_0': [],
                     'train_acc_1_1': [],
                     'train_acc_1_2': [],
                     'train_acc_2_0': [],
                     'train_acc_2_1': [],
                     'train_acc_2_2': [],
                     'acc_0_0': [],
                     'acc_0_1': [],
                     'acc_0_2': [],
                     'acc_1_0': [],
                     'acc_1_1': [],
                     'acc_1_2': [],
                     'acc_2_0': [],
                     'acc_2_1': [],
                     'acc_2_2': []
                     }
    data_dict = {'loss':[],
                 'train_loss':[],
                 'train_loss_0_0': [],
                 'train_loss_0_1': [],
                 'train_loss_0_2': [],
                 'train_loss_1_0': [],
                 'train_loss_1_1': [],
                 'train_loss_1_2': [],
                 'train_loss_2_0': [],
                 'train_loss_2_1': [],
                 'train_loss_2_2': [],
                 'loss_0_0': [],
                 'loss_0_1': [],
                 'loss_0_2': [],
                 'loss_1_0': [],
                 'loss_1_1': [],
                 'loss_1_2': [],
                 'loss_2_0': [],
                 'loss_2_1': [],
                 'loss_2_2': []
                 }

    for event in summary_iterator(file):
        for value in event.summary.value:
            if 'loss' not in value.tag:
                continue
            data_dict[value.tag].append(value.simple_value)

    step_num = len(data_dict['loss_0_0'])
    x = list(range(step_num))
    plt.plot(x, data_dict['loss_0_2'], alpha=0.5, label='02')
    plt.plot(x, data_dict['loss_1_2'], alpha=0.5, label='12')
    plt.plot(x, data_dict['loss_2_2'], alpha=0.5, label='22')
    plt.legend()
    plt.title('test loss')
    plt.show()


# 自定义的绘图函数
# 下面是绘制损失曲线的分层曲线，但是还没完全改代码
def redraw_step_loss_hjy(file, start_index, end_index, bottom_step_num=2,middle_step_num=2, isShow=False):
    """
    :param file:
    :param start_index:
    :param end_index:
    :param bottom_step_num:
    :param isShow:
    :return:
    """

    from tensorflow.python.summary.summary_iterator import summary_iterator
    import matplotlib.pyplot as plt
    from fnmatch import fnmatch
    # assert len(summary_iterator) == 1  # 以防把loss之外的东西搞坏了

    data_dict = {}
    for mid_step in list(range(middle_step_num+1)):
        for bot_step in list(range(bottom_step_num+1)):
            data_dict['train_loss_' + str(mid_step) + '_' + str(bot_step)]=[]
            data_dict['loss_' + str(mid_step) + '_' + str(bot_step)] = []

    for event in summary_iterator(file):
        for value in event.summary.value:
            # print(value)
            if 'loss' not in value.tag:
                continue
            data_dict[value.tag].append(value.simple_value)

        # exit()

    # print("打印数据字典")
    # print(data_dict)
    # exit()
    # return

    step_num = len(data_dict['loss_0_0'])

    if end_index > step_num:
        end_index = step_num

    x = list(range(step_num))[start_index:end_index]
    # plt.plot(x, data_dict['loss_0_2'][start_index:end_index], alpha=0.5, label='02')
    # plt.plot(x, data_dict['loss_1_2'][start_index:end_index], alpha=0.5, label='12')
    # plt.plot(x, data_dict['loss_2_2'][start_index:end_index], alpha=0.5, label='22')
    # plt.legend()
    # plt.title('test loss')

    # 保存的文件夹的名称

    assert root_dir[-1]!='/'
    save_dir_name = file.split('/')[-2]
    save_dir_path = os.path.join('../output/compare', root_dir.split('/', -1)[-1], save_dir_name)

    # print(save_dir_name)
    # print(save_dir_path)
    #
    # exit(0)

    if not os.path.isdir(save_dir_path):
        os.mkdir(save_dir_path)

    # if not os.path.isdir()

    # 绘制测试误差
    for bottom_step in range(bottom_step_num + 1):
        ax = plt.figure()

        for middle_step in range(3):
            plt.plot(x, data_dict[f'loss_{middle_step}_{bottom_step}'][start_index:end_index], alpha=0.5,
                     label=f'{middle_step}-{bottom_step}')

        plt.legend()
        plt.title(f'test loss - bottom step {bottom_step}')

        plt.savefig(os.path.join(save_dir_path, f'test_bottom_step_{bottom_step}.png'))
        plt.close(ax)

    # 绘制训练误差
    for bottom_step in range(bottom_step_num + 1):
        ax = plt.figure()

        for middle_step in range(3):
            plt.plot(x, data_dict[f'train_loss_{middle_step}_{bottom_step}'][start_index:end_index], alpha=0.5,
                     label=f'{middle_step}-{bottom_step}')

        plt.legend()
        plt.title(f'train loss - bottom step {bottom_step}')

        plt.savefig(os.path.join(save_dir_path, f'train_bottom_step_{bottom_step}.png'))
        plt.close(ax)

    if isShow:
        plt.show()

# if not os.path.isdir()


# 自定义的绘图函数
# 下面是绘制损失曲线的分层曲线，但是还有改代码
def redraw_step_acc_hjy(file, start_index, end_index, bottom_step_num=2, isShow=False):
    """
    :param file:
    :param start_index:
    :param end_index:
    :param bottom_step_num:
    :param isShow:
    :return:
    """

    from tensorflow.python.summary.summary_iterator import summary_iterator
    import matplotlib.pyplot as plt
    from fnmatch import fnmatch
    # assert len(summary_iterator) == 1  # 以防把loss之外的东西搞坏了

    bottom_step_0 = []
    bottom_step_1 = []
    bottom_step_2 = []

    train_bottom_step_0 = []
    train_bottom_step_1 = []
    train_bottom_step_2 = []

    data_dict_acc = {'train_acc_0_0': [],
                     'train_acc_0_1': [],
                     'train_acc_0_2': [],
                     'train_acc_1_0': [],
                     'train_acc_1_1': [],
                     'train_acc_1_2': [],
                     'train_acc_2_0': [],
                     'train_acc_2_1': [],
                     'train_acc_2_2': [],
                     'acc_0_0': [],
                     'acc_0_1': [],
                     'acc_0_2': [],
                     'acc_1_0': [],
                     'acc_1_1': [],
                     'acc_1_2': [],
                     'acc_2_0': [],
                     'acc_2_1': [],
                     'acc_2_2': []
                     }
    data_dict = {'train_loss_0_0': [],
                 'train_loss_0_1': [],
                 'train_loss_0_2': [],
                 'train_loss_1_0': [],
                 'train_loss_1_1': [],
                 'train_loss_1_2': [],
                 'train_loss_2_0': [],
                 'train_loss_2_1': [],
                 'train_loss_2_2': [],
                 'loss_0_0': [],
                 'loss_0_1': [],
                 'loss_0_2': [],
                 'loss_1_0': [],
                 'loss_1_1': [],
                 'loss_1_2': [],
                 'loss_2_0': [],
                 'loss_2_1': [],
                 'loss_2_2': []
                 }

    for event in summary_iterator(file):
        for value in event.summary.value:
            # print(value)
            if 'acc' not in value.tag:
                continue
            data_dict_acc[value.tag].append(value.simple_value)

        # exit()

    # print("打印数据字典")
    # print(data_dict_acc)
    # exit()

    step_num = len(data_dict_acc['acc_0_0'])

    if end_index > step_num:
        end_index = step_num

    x = list(range(step_num))[start_index:end_index]
    # plt.plot(x, data_dict['loss_0_2'][start_index:end_index], alpha=0.5, label='02')
    # plt.plot(x, data_dict['loss_1_2'][start_index:end_index], alpha=0.5, label='12')
    # plt.plot(x, data_dict['loss_2_2'][start_index:end_index], alpha=0.5, label='22')
    # plt.legend()
    # plt.title('test loss')

    # 保存的文件夹的名称
    save_dir_name = file.split('/')[-2]
    save_dir_path = os.path.join('../output/tensorboard/compare', root_dir.split('/', -1)[-1], save_dir_name)

    # print(save_dir_name)
    # print(save_dir_path)
    #
    # exit(0)

    if not os.path.isdir(save_dir_path):
        os.mkdir(save_dir_path)

    # if not os.path.isdir()

    # 绘制测试准确率
    for bottom_step in range(bottom_step_num + 1):
        ax = plt.figure()

        for middle_step in range(3):
            plt.plot(x, data_dict_acc[f'acc_{middle_step}_{bottom_step}'][start_index:end_index], alpha=0.5,
                     label=f'{middle_step}-{bottom_step}')

        plt.legend()
        plt.title(f'test acc - bottom step {bottom_step}')

        plt.savefig(os.path.join(save_dir_path, f'test_bottom_step_{bottom_step}.png'))
        plt.close(ax)

    # 绘制训练准确率
    for bottom_step in range(bottom_step_num + 1):
        ax = plt.figure()

        for middle_step in range(3):
            plt.plot(x, data_dict_acc[f'train_acc_{middle_step}_{bottom_step}'][start_index:end_index], alpha=0.5,
                     label=f'{middle_step}-{bottom_step}')

        plt.legend()
        plt.title(f'train acc - bottom step {bottom_step}')

        plt.savefig(os.path.join(save_dir_path, f'train_bottom_step_{bottom_step}.png'))
        plt.close(ax)

    if isShow:
        plt.show()


def draw_compare_step(file1, file2):
    from tensorflow.python.summary.summary_iterator import summary_iterator
    import matplotlib.pyplot as plt
    from fnmatch import fnmatch
    # assert len(summary_iterator) == 1  # 以防把loss之外的东西搞坏了

    bottom_step_0 = []
    bottom_step_1 = []
    bottom_step_2 = []

    train_bottom_step_0 = []
    train_bottom_step_1 = []
    train_bottom_step_2 = []

    data_dict_acc = {'train_acc_0_0': [],
                     'train_acc_0_1': [],
                     'train_acc_0_2': [],
                     'train_acc_1_0': [],
                     'train_acc_1_1': [],
                     'train_acc_1_2': [],
                     'train_acc_2_0': [],
                     'train_acc_2_1': [],
                     'train_acc_2_2': [],
                     'acc_0_0': [],
                     'acc_0_1': [],
                     'acc_0_2': [],
                     'acc_1_0': [],
                     'acc_1_1': [],
                     'acc_1_2': [],
                     'acc_2_0': [],
                     'acc_2_1': [],
                     'acc_2_2': []
                     }
    data_dict_1 = {}

    data_dict_2 = {}

    for event in summary_iterator(file1):
        for value in event.summary.value:
            if 'loss' not in value.tag:
                continue
            if value.tag not in data_dict_1:
                data_dict_1[value.tag] = []
            data_dict_1[value.tag].append(value.simple_value)

    for event in summary_iterator(file2):
        for value in event.summary.value:
            if 'loss' not in value.tag:
                continue
            if value.tag not in data_dict_2:
                data_dict_2[value.tag] = []
            data_dict_2[value.tag].append(value.simple_value)

    step_num = len(data_dict_1['train_loss_0_0'])
    x = list(range(step_num))[100:400]

    # plt.plot(x, data_dict_1['loss_2_0'][100:], alpha=0.5, label='20')
    # plt.plot(x, data_dict_1['loss_2_1'][100:], alpha=0.5, label='21')

    plt.plot(x, data_dict_1['train_loss_0_2'][100:400], alpha=0.5, label='02')
    plt.plot(x, data_dict_1['train_loss_1_2'][100:400], alpha=0.5, label='12')
    plt.plot(x, data_dict_1['train_loss_2_2'][100:400], alpha=0.5, label='22')

    # plt.plot(x, data_dict_2['train_loss_0_0'][100:400], alpha=0.5, label='00')
    # plt.plot(x, data_dict_2['train_loss_0_1'][100:400], alpha=0.5, label='01')
    # plt.plot(x, data_dict_2['train_loss_0_2'][100:400], alpha=0.5, label='02')
    # plt.plot(x, data_dict_2['train_loss_0_3'][100:400], alpha=0.5, label='03')
    # plt.plot(x, data_dict_2['train_loss_0_4'][100:400], alpha=0.5, label='04')
    # plt.plot(x, data_dict_2['train_loss_0_5'][100:400], alpha=0.5, label='05')
    plt.title('train loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    redraw_step_acc('C:/Users/Administrator/Desktop/events.out.tfevents.1668403081.localhost.localdomain')
    # root_dir = '../output/tensorboard/omniglot'
    #
    # l = os.listdir(root_dir)
    # l = list(filter(lambda x: x.find('reproduce') != -1, l))
    # l.sort()
    # print(l)
    #
    # file_path_list = []
    #
    # i = 0
    #
    # start_index = 0
    # end_index = 2000
    #
    # for dir_name in l:
    #     print(i, end=' ')
    #     #
    #     i += 1
    #     #
    #     temp_path = os.path.join(root_dir, dir_name)
    #
    #     # tf.event 文件的路径，文件夹下就只有一个文件，所以是 os.listdir(temp_path)[0]
    #     file_path = os.path.join(root_dir, dir_name, os.listdir(temp_path)[0])
    #
    #     print("开始绘制 {}".format(dir_name))
    #
    #     # redraw_step_acc_hjy(file_path, start_index, end_index, bottom_step_num=1)
    #
    #     #
    #     try:
    #         # 绘制准确率分层曲线
    #         # redraw_step_acc_hjy(file_path, start_index, end_index, bottom_step_num=2)
    #
    #         # 绘制损失分层曲线
    #         redraw_step_loss_hjy(file_path, start_index, end_index, middle_step_num=2,bottom_step_num=5)
    #
    #         print("完成绘制 {}".format(dir_name))
    #     except Exception as e:
    #         print("绘制异常 {}".format(dir_name))
    #
    #         # 打印异常信息
    #         import traceback
    #
    #         traceback.print_exc()
    #     # #
    #
    #     file_path_list.append(file_path)
    #
    # file_path_list.sort()
    #
    # print(len(file_path_list))
