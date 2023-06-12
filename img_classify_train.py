import torch, os
import numpy as np
from options import Options
from MAML import MAML
from H_ProtoNet import H_ProtoNet
from H_MAML import H_MAML
from ProtoNet import ProtoNet
from tqdm import tqdm
from img_classify import ImgClassification
from tool import Tensorboard_manager, load_network, save_network
import nni

def train_an_epoch(dataset_obejct: ImgClassification, model, args, train_or_test):
    if 'hierarchical' in args.sample_mode:
        eight_cluster_batch = dataset_obejct.next_hierarchical(mode=train_or_test, sample_mode=args.sample_mode)
        if args.method == 'Hierarchical_MAML' or args.method == 'Hierarchical_ProtoNet':
            eight_cluster_batch[:4] = [torch.from_numpy(one_cluster_batch).to(args.device) for
                                       one_cluster_batch in eight_cluster_batch[:4]]
            eight_cluster_batch[4:] = [torch.from_numpy(one_cluster_batch).to(args.device, dtype=torch.int64) for
                                       one_cluster_batch in eight_cluster_batch[4:]]
            loss, acc, mid_grad_list = model(eight_cluster_batch, train_or_test)


        elif args.method == 'MAML' or 'ProtoNet':
            # MAML用hierarchical采出来的数据需要合并，有几种方案
            # 展开cluster_batch
            x_spt, x_qry, y_spt, y_qry = dataset_obejct.unpack_hie_data(eight_cluster_batch, train_or_test)
            x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(args.device), torch.from_numpy(
                y_spt).to(args.device, dtype=torch.int64), \
                                         torch.from_numpy(x_qry).to(args.device), torch.from_numpy(
                y_qry).to(args.device, dtype=torch.int64)
            loss, acc, mid_grad_list = model(x_spt, y_spt, x_qry, y_qry, train_or_test)

        return loss, acc, mid_grad_list

    # 暂时弃用
    else:
        assert args.method == 'MAML'
        if args.sample_mode == 'uniform':
            x_spt, y_spt, x_qry, y_qry = dataset_obejct.next_uniform(train_or_test)
        elif args.sample_mode == 'single_batch':
            x_spt, y_spt, x_qry, y_qry = dataset_obejct.next_single_batch_domain(train_or_test)
        else:
            raise NotImplementedError

        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(args.device), torch.from_numpy(
            y_spt).to(args.device, dtype=torch.int64), \
                                     torch.from_numpy(x_qry).to(args.device), torch.from_numpy(
            y_qry).to(args.device, dtype=torch.int64)

        loss, acc = model(x_spt, y_spt, x_qry, y_qry, train_or_test)

def report_train_test_metric():
    # train loss
    print('step:', step, '\tTrain acc:', acc_list)
    if args.method == 'MAML':
        for i, acc in enumerate(acc_list):
            tb_manager.update_graph(step, {'train_acc_' + str(i): acc})
        for i, loss in enumerate(loss_list):
            tb_manager.update_graph(step, {'train_loss_' + str(i): loss})

    elif args.method == 'Hierarchical_MAML':
        for i, acc_task in enumerate(acc_list):
            for j, acc in enumerate(acc_task):
                tb_manager.update_graph(step, {'train_acc_' + str(i) + '_' + str(j): acc})
        for i, loss_task in enumerate(loss_list):
            for j, loss in enumerate(loss_task):
                tb_manager.update_graph(step, {'train_loss_' + str(i) + '_' + str(j): loss})
        for i, mid_grad in enumerate(mid_grad_list):
            for j, mid_grad_row in enumerate(mid_grad):
                tb_manager.summary_writer.add_histogram('train_mid_grad_' + str(i + 1) + '_' + str(j),
                                                        mid_grad_row, step)

    elif args.method == 'Hierarchical_ProtoNet' or args.method == 'ProtoNet':
        tb_manager.update_graph(step, {'train_loss': loss_list})
        tb_manager.update_graph(step, {'train_acc': acc_list})

    # test loss
    if step % 10 == 0:
        if args.method == 'MAML':
            loss_test_list, acc_test_list, _ = train_an_epoch(dataset_object, method, args, 'test')
            for i, acc in enumerate(acc_test_list):
                tb_manager.update_graph(step, {'acc_' + str(i): acc})
            for i, loss in enumerate(loss_test_list):
                tb_manager.update_graph(step, {'loss_' + str(i): loss})

        elif args.method == 'Hierarchical_MAML':
            loss_test_list, acc_test_list, mid_grad_list_test = train_an_epoch(dataset_object, method, args, 'test')
            for i, acc_task in enumerate(acc_test_list):
                for j, acc in enumerate(acc_task):
                    tb_manager.update_graph(step, {'acc_' + str(i) + '_' + str(j): acc})
            for i, loss_task in enumerate(loss_test_list):
                for j, loss in enumerate(loss_task):
                    tb_manager.update_graph(step, {'loss_' + str(i) + '_' + str(j): loss})
            for i, mid_grad in enumerate(mid_grad_list_test):
                for j, mid_grad_row in enumerate(mid_grad):
                    tb_manager.summary_writer.add_histogram('mid_grad_' + str(i + 1) + '_' + str(j), mid_grad_row,
                                                            step)
        elif args.method == 'Hierarchical_ProtoNet' or args.method == 'ProtoNet':
            loss_test_list, acc_test_list, _ = train_an_epoch(dataset_object, method, args, 'test')
            tb_manager.update_graph(step, {'loss': loss_test_list})
            tb_manager.update_graph(step, {'acc': acc_test_list})
        else:
            raise NotImplementedError

        # report nni
        acc_test = acc_test_list
        while type(acc_test) == type([]):
            acc_test = acc_test[-1]
        nni.report_intermediate_result(acc_test)
        print('Test acc:', acc_test_list)

if __name__ == '__main__':
    # init args
    args = Options().parse()

    # nni realted
    param = nni.get_next_parameter()
    for key in param.keys():
        args.__dict__[key] = param[key]

    # init random seed
    torch.cuda.cudnn_enabled = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

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
    elif args.method == 'Hierarchical_ProtoNet':
        method = H_ProtoNet(args).to(args.device)
    elif args.method == 'ProtoNet':
        method = ProtoNet(args).to(args.device)
    else:
        raise NotImplementedError

    if args.start_epoch != 'no':
        method = load_network(method, args.load_model_dir, args.device)

    # init dataset
    print("加载数据集")
    dataset_object = ImgClassification(args)

    # train
    if args.start_epoch != 'no':
        begin_epoch = int(args.start_epoch) + 1
    else:
        begin_epoch = 0

    print("开始训练")
    for step in tqdm(range(begin_epoch, args.epoch + 1)):
        loss_list, acc_list, mid_grad_list = train_an_epoch(dataset_object, method, args, 'train')

        if step % 10 == 0:
            report_train_test_metric()

        if step % 5000 == 0:
            # save the network
            save_network(method, os.path.join(args.model_dir, args.scheme_name + '_' + str(step) + '.pth'), args)
