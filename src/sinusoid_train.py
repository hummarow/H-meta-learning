import torch, os
import numpy as np
from sinusoid import sinusoid
from options import Options
from H_MAML import H_MAML
from MAML import MAML
from tqdm import tqdm
from tool import Tensorboard_manager, load_network, save_network


def train_an_epoch(dataset_obejct, model, args, train_or_test):
    if 'hierarchical' in args.sample_mode:
        if args.sample_mode == 'hierarchical_uniform':
            eight_cluster_batch = dataset_obejct.next_hierarchical_uniform(train_or_test)
        elif args.sample_mode == 'hierarchical_concentrate':
            eight_cluster_batch = dataset_obejct.next_hierarchical_concentrate(train_or_test)
        else:
            raise NotImplementedError

        if args.method == 'Hierarchical_MAML':
            eight_cluster_batch = [torch.from_numpy(one_cluster_batch).to(args.device, dtype=torch.float32) for
                                   one_cluster_batch in eight_cluster_batch]

            loss, acc, mid_grad_list = model(eight_cluster_batch, train_or_test)

        elif args.method == 'MAML':
            # MAML用hierarchical采出来的数据需要合并，有几种方案
            # 展开cluster_batch
            eight_cluster_mass = [one_cluster_batch.reshape((-1, one_cluster_batch.shape[-1])) for one_cluster_batch in
                                  eight_cluster_batch]

            if train_or_test == 'train':
                x_spt = np.vstack((eight_cluster_mass[0], eight_cluster_mass[2]))
                x_qry = np.vstack((eight_cluster_mass[1], eight_cluster_mass[3]))
                y_spt = np.vstack((eight_cluster_mass[4], eight_cluster_mass[6]))
                y_qry = np.vstack((eight_cluster_mass[5], eight_cluster_mass[7]))

            else:  # 测试用cluster中的spt_task不是目标对象，在当前方案下，直接舍弃
                x_spt = eight_cluster_mass[2]
                y_spt = eight_cluster_mass[6]
                x_qry = eight_cluster_mass[3]
                y_qry = eight_cluster_mass[7]

            x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(args.device,
                                                                    dtype=torch.float32), torch.from_numpy(
                y_spt).to(args.device, dtype=torch.float32), \
                                         torch.from_numpy(x_qry).to(args.device,
                                                                    dtype=torch.float32), torch.from_numpy(
                y_qry).to(args.device, dtype=torch.float32)

            loss, acc = model(x_spt, y_spt, x_qry, y_qry, train_or_test)

    else:
        assert args.method == 'MAML'
        if args.sample_mode == 'uniform':
            x_spt, y_spt, x_qry, y_qry = dataset_obejct.next_uniform(train_or_test)
        elif args.sample_mode == 'single_batch':
            x_spt, y_spt, x_qry, y_qry = dataset_obejct.next_single_batch_domain(train_or_test)
        else:
            raise NotImplementedError

        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(args.device, dtype=torch.float32), torch.from_numpy(
            y_spt).to(args.device, dtype=torch.float32), \
                                     torch.from_numpy(x_qry).to(args.device, dtype=torch.float32), torch.from_numpy(
            y_qry).to(args.device, dtype=torch.float32)

        loss, acc, _ = model(x_spt, y_spt, x_qry, y_qry, train_or_test)

    if args.method == 'MAML':
        return loss, acc
    else:
        return loss, acc, mid_grad_list


if __name__ == '__main__':
    # init args
    args = Options().parse()

    # init random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # init network structure

    # init tensorboard, checkpoint, and config
    tb_manager = Tensorboard_manager(args.tb_dir)
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    for file in os.listdir(args.model_dir):
        os.remove(os.path.join(args.model_dir, file))
    with open(args.config_dir, 'w') as f:
        f.write(str(args.__dict__))

    # init device
    device = torch.device("cuda:" + args.gpu_id)
    args.device = device

    # init learner
    if args.method == 'MAML':
        method = MAML(args).to(args.device)
    elif args.method == 'Hierarchical_MAML':
        method = H_MAML(args).to(args.device)
    else:
        raise NotImplementedError

    if args.start_epoch != 'no':
        method = load_network(method, args.load_model_dir, args.device)

    # init dataset
    dataset_object = sinusoid(args)

    # train
    if args.start_epoch != 'no':
        begin_epoch = int(args.start_epoch) + 1
    else:
        begin_epoch = 0

    for step in tqdm(range(begin_epoch, args.epoch + 1)):
        if args.method == 'MAML':
            loss_list, acc_list = train_an_epoch(dataset_object, method, args, 'train')
        else:
            loss_list, acc_list, mid_grad_list = train_an_epoch(dataset_object, method, args, 'train')

        if step % 10 == 0:
            # train loss
            print('step:', step, '\tTrain loss:', loss_list)
            if args.method == 'MAML':
                for i, loss in enumerate(loss_list):
                    tb_manager.update_graph(step, {'train_loss_' + str(i): loss})

            else:
                for i, loss_task in enumerate(loss_list):
                    for j, loss in enumerate(loss_task):
                        tb_manager.update_graph(step, {'train_loss_' + str(i) + '_' + str(j): loss})
                for i, mid_grad in enumerate(mid_grad_list):
                    for j, mid_grad_row in enumerate(mid_grad):
                        tb_manager.summary_writer.add_histogram('train_mid_grad_' + str(i + 1) + '_' + str(j),
                                                                mid_grad_row, step)

            # test loss
            if args.method == 'MAML':
                loss_test_list, acc_test_list = train_an_epoch(dataset_object, method, args, 'test')
                for i, loss in enumerate(loss_test_list):
                    tb_manager.update_graph(step, {'loss_' + str(i): loss})
            else:
                loss_test_list, acc_test_list, mid_grad_list_test = train_an_epoch(dataset_object, method, args, 'test')
                for i, loss_task in enumerate(loss_test_list):
                    for j, loss in enumerate(loss_task):
                        tb_manager.update_graph(step, {'loss_' + str(i) + '_' + str(j): loss})
                for i, mid_grad in enumerate(mid_grad_list_test):
                    for j, mid_grad_row in enumerate(mid_grad):
                        tb_manager.summary_writer.add_histogram('mid_grad_' + str(i + 1) + '_' + str(j), mid_grad_row,
                                                                step)

            print('Test loss:', loss_test_list)

        if step % 5000 == 0:
            # save the network
            save_network(method, os.path.join(args.model_dir, args.scheme_name + '_' + str(step) + '.pth'),args)
