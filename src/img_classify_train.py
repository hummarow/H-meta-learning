import torch, os
import numpy as np
from options import Options
from MAML import MAML
from H_ProtoNet import H_ProtoNet
from H_MAML import H_MAML
from ProtoNet import ProtoNet
from img_classify import ImgClassification
from tool import Tensorboard_manager, load_network, save_network
from pathlib import Path
from meta_generator import MetaDatasetsGenerator
from tqdm import tqdm
from utils import (
    get_basic_expt_info,
    split_support_query,
    cycle,
    batch_split_support_query,
)


class Trainer:
    def __init__(self, args):
        self.args = args
        # init learner
        if args.method == "MAML":
            self.method = MAML(args).to(args.device)
        elif args.method == "Hierarchical_MAML":
            self.method = H_MAML(args).to(args.device)
        elif args.method == "Hierarchical_ProtoNet":
            self.method = H_ProtoNet(args).to(args.device)
        elif args.method == "ProtoNet":
            self.method = ProtoNet(args).to(args.device)
        else:
            raise NotImplementedError
            
        self.trainable = True
        self.start_epoch = args.start_epoch
        if self.start_epoch == "last":
            import glob
            try:
                if args.holdout:
                    # get the model of the last epoch
                    model_dir = os.path.join(args.model_dir, args.test_domain.name)
                    glob.glob(os.path.join(model_dir, "*.pth"))
                    model_path = max(glob.glob(os.path.join(model_dir, "*.pth")), key=os.path.getctime)
                else:
                    model_path = max(glob.glob(os.path.join(args.model_dir, "*.pth")), key=os.path.getctime)
                print("Last model path: ", model_path)
                
                self.method = load_network(self.method, model_path, args.device)
                self.start_epoch = model_path.split("_")[-1].split(".")[0]
                if int(self.start_epoch) >= args.epoch:
                    self.trainable = False
                

            except ValueError:
                print("No model found in the model directory.")
                print("Start from scratch.")
                self.start_epoch = "no"

        if self.start_epoch.isdigit():
            if args.holdout:
                model_path = os.path.join(
                    args.model_dir,
                    args.test_domain.name,
                    f"{args.scheme_name}_{self.start_epoch}.pth",
                )
            else:
                model_path = os.path.join(
                    args.model_dir,
                    f"{args.scheme_name}_{self.start_epoch}.pth",
                )
            self.method = load_network(self.method, model_path, args.device)

        self.dataset_object = MetaDatasetsGenerator(
            args,
            contrastive=False,
            batch=(
                2 if not args.contrastive else 1
            ),  # 1: only query (support tasks are sampled from contrastive loader), 2: support and query
        )
        self.train_loader = cycle(self.dataset_object.m_dataloader["train"])
        self.val_loader = cycle(self.dataset_object.m_dataloader["valid"])
        self.test_loader = cycle(self.dataset_object.m_dataloader["test"])

        self.contrastive = args.contrastive
        if args.contrastive:
            self.contrastive_dataset_object = MetaDatasetsGenerator(
                args, contrastive=True, batch=1
            )
            self.contrastive_train_loader = cycle(
                self.contrastive_dataset_object.m_dataloader["train"]
            )
            self.contrastive_val_loader = cycle(
                self.contrastive_dataset_object.m_dataloader["valid"]
            )
            self.contrastive_test_loader = cycle(
                self.contrastive_dataset_object.m_dataloader["test"]
            )

        self.domains = args.domains
        self.num_domains = len(args.domains)
        self.holdout = args.holdout
        model_dir = os.path.join(args.model_dir, args.test_domain.name) if self.holdout else args.model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if self.holdout:
            self.best_model_path = os.path.join(
                model_dir,
                self.args.scheme_name + "_best_holdout.pth",
            )
        else:
            self.best_model_path = os.path.join(
                model_dir,
                self.args.scheme_name + "_best.pth",
            )
        
        self.max_test_task = args.max_test_task

        self.fast_mode = args.fast_mode

        # self.is_hierarchical = "hierarchical" in args.sample_mode
        self.is_hierarchical = "Hierarchical" in args.method
        self.print_freq = 300 if self.is_hierarchical else 100
        self.avg_train_acc = []
        self.avg_val_acc = []
        self.avg_test_acc = []
        self.avg_train_loss = []
        self.avg_val_loss = []
        self.avg_test_loss = []

    def train(self):
        # train
        if self.start_epoch != "no":
            begin_epoch = int(self.start_epoch) + 1
        else:
            begin_epoch = 0
        loss_U_best = 1000
        loss_M_best = 1000
        for step in range(begin_epoch, self.args.epoch + 1):
            loss_list, acc_list, mid_grad_list = self.epoch(step, mode="train")
            if step % self.print_freq == 0:
                print(f"step: {step}/{self.args.epoch} \tTrain acc: {acc_list}")
                report_train_test_metric(
                    step, acc_list, loss_list, mid_grad_list, self.args.method, "train", self.args
                )
                # Calculate average accuracy and loss
                if self.is_hierarchical:
                    self.avg_train_acc.append(np.mean([acc[-1] for acc in acc_list]))
                    self.avg_train_loss.append(
                        np.mean([loss[-1] for loss in loss_list])
                    )
                else:
                    self.avg_train_acc.append(acc_list[-1])
                    self.avg_train_loss.append(loss_list[-1])

                loss, acc = self.test(step, "valid", load_best=False)
                self.avg_val_acc.append(acc)
                self.avg_val_loss.append(loss)

                if not self.fast_mode:
                    loss, acc = self.test(step, "test", load_best=False)
                    self.avg_test_acc.append(acc)
                    self.avg_test_loss.append(loss)

                print(f"Train Loss:\t{self.avg_train_loss[-1]:.2f}")
                print(f"Valid Loss:\t{self.avg_val_loss[-1]:.2f}")
                if not self.fast_mode:
                    print(f"Test Loss:\t{self.avg_test_loss[-1]:.2f}")
                print(f"Train Accuracy:\t{self.avg_train_acc[-1]:.2f}")
                print(f"Valid Accuracy:\t{self.avg_val_acc[-1]:.2f}")
                if not self.fast_mode:
                    print(f"Test Accuracy:\t{self.avg_test_acc[-1]:.2f}")

                # Save the best model
                # if self.avg_val_acc[-1] == max(self.avg_val_acc):
                #     save_network(
                #         self.method,
                #         self.best_model_path,
                #         self.args,
                #     )
                if self.avg_val_loss[-1] == min(self.avg_val_loss):
                    if self.is_hierarchical:
                        loss_M_best = loss_list[0][-1]
                        loss_U_best = loss_list[-1][-1]
                    save_network(
                        self.method,
                        self.best_model_path,
                        self.args,
                    )

                if self.holdout:
                    model_path = os.path.join(
                        self.args.model_dir,
                        self.args.test_domain.name,
                        f"{self.args.scheme_name}_{step}.pth",
                    )
                else:
                    model_path = os.path.join(
                        self.args.model_dir,
                        f"{self.args.scheme_name}_{step}.pth",
                    )
                save_network(self.method, model_path, self.args)

        # best_val_acc = max(self.avg_val_acc)
        # best_val_acc_index = self.avg_val_acc.index(best_val_acc)

        min_val_loss = min(self.avg_val_loss)
        min_val_loss_index = self.avg_val_loss.index(min_val_loss)
        if not self.fast_mode:
                    
            # best_test_acc = self.avg_test_acc[best_val_acc_index]
            best_test_acc = self.avg_test_acc[min_val_loss_index]

            print(f"Best test accuracy: {best_test_acc}")
            print(f"Test accuracy at best validation: {self.avg_test_acc[-1]}")

            return self.avg_test_acc[-1]
        
        print(f"Best validation loss: {min_val_loss}")
        print(f"Best validation epoch: {min_val_loss_index * self.print_freq}")
        print(f"Best U Loss: {loss_U_best}")
        print(f"Best M Loss: {loss_M_best}")
        return min_val_loss

    def test(self, step, mode, load_best=False):
        # load the best model and test
        if load_best:
            print("Best model path: ", self.best_model_path)
            self.method = load_network(
                self.method, self.best_model_path, self.args.device
            )
        total_loss_list = []
        total_acc_list = []
        num_tasks = self.max_test_task

        num_tasks = 10 if mode == "valid" else num_tasks
        for _ in range(num_tasks):
            loss_list, acc_list, _ = self.epoch(0, mode=mode)
            if self.is_hierarchical:
                total_loss_list.append(np.mean([loss[-1] for loss in loss_list]))
                total_acc_list.append(np.mean([acc[-1] for acc in acc_list]))
            else:
                total_loss_list.append(loss_list[-1])
                total_acc_list.append(acc_list[-1])
        loss = np.mean(total_loss_list)
        acc = np.mean(total_acc_list)
        report_train_test_metric(step, [acc], [loss], [0], self.args.method, mode, self.args)
        return loss, acc

    def epoch(self, step, mode="train"):
        if mode == "train":
            query_task_dataloader = self.train_loader
            support_task_dataloader = (
                query_task_dataloader
                if not self.contrastive
                else self.contrastive_train_loader
            )
            num_domains = self.num_domains - 1 * int(args.holdout)
        elif mode == "valid":
            query_task_dataloader = self.val_loader
            support_task_dataloader = (
                query_task_dataloader
                if not self.contrastive
                else self.contrastive_val_loader
            )
            num_domains = self.num_domains - 1 * int(args.holdout)
        else:
            query_task_dataloader = self.test_loader
            support_task_dataloader = (
                query_task_dataloader
                if not self.contrastive
                else self.contrastive_test_loader
            )
            num_domains = 1

        n_way, n_support, n_query, y_spt, y_qry = get_basic_expt_info(args)
        # get Support Task data
        ST_batch, _ = next(support_task_dataloader)
        ST_x_spt, ST_x_qry = batch_split_support_query(
            ST_batch,
            n_support=n_support if not self.contrastive else int(args.contrastive_batch_size/2),
            n_query=n_query if not self.contrastive else args.contrastive_batch_size-int(args.contrastive_batch_size/2),
            n_way=n_way if not self.contrastive else 1,
            num_domains=num_domains,
        )
        if self.contrastive:
            ST_x_spt = torch.transpose(ST_x_spt, 1, 2)
            ST_x_qry = torch.transpose(ST_x_qry, 1, 2)
        y_spt = y_spt.repeat(num_domains, 1)
        y_qry = y_qry.repeat(num_domains, 1)

        if self.is_hierarchical:
            # get Query Task data
            QT_batch, _ = next(query_task_dataloader)

            QT_x_spt, QT_x_qry = batch_split_support_query(
                QT_batch,
                n_support=n_support,
                n_query=n_query,
                n_way=n_way,
                num_domains=num_domains,
            )

            batch = [
                ST_x_spt.to(args.device),
                ST_x_qry.to(args.device),
                QT_x_spt.to(args.device),
                QT_x_qry.to(args.device),
                y_spt.to(args.device, dtype=torch.int64),
                y_qry.to(args.device, dtype=torch.int64),
            ]
        else:
            batch = [
                ST_x_spt.to(args.device),
                ST_x_qry.to(args.device),
                y_spt.to(args.device, dtype=torch.int64),
                y_qry.to(args.device, dtype=torch.int64),
            ]

        loss_list, acc_list, mid_grad_list = self.method(batch, mode)

        return loss_list, acc_list, mid_grad_list


def report_train_test_metric(step, acc_list, loss_list, mid_grad_list, method, mode, args):
    if method == "MAML":
        for i, acc in enumerate(acc_list):
            tb_manager.update_graph(step, {f"{mode}_acc_{i}": acc})
        for i, loss in enumerate(loss_list):
            tb_manager.update_graph(step, {f"{mode}_loss_{i}": loss})

    elif method == "Hierarchical_MAML":
        prefix = f"{mode}" if not args.holdout else f"{mode}_{args.test_domain.name}"
        for i, acc_task in enumerate(acc_list):
            if isinstance(acc_task, list):
                tb_manager.update_graph(step, {f"{prefix}_M_acc": acc_task[0][-1]})
                tb_manager.update_graph(step, {f"{prefix}_U_acc": acc_task[-1][-1]})
                print(f"{prefix}_M_acc: {acc_task[0][-1]}")
                print(f"{prefix}_U_acc: {acc_task[-1][-1]}")
                for j, acc in enumerate(acc_task):
                    tb_manager.update_graph(step, {f"{prefix}_acc_{i}_{j}": acc})
            else:
                tb_manager.update_graph(step, {f"{prefix}_acc_{i}_avg": acc_task})
        for i, loss_task in enumerate(loss_list):
            if isinstance(loss_task, list):
                tb_manager.update_graph(step, {f"{prefix}_M_loss": loss_task[0][-1]})
                tb_manager.update_graph(step, {f"{prefix}_U_loss": loss_task[-1][-1]})
                print(f"{prefix}_M_loss: {loss_task[0][-1]}")
                print(f"{prefix}_U_loss: {loss_task[-1][-1]}")
                for j, loss in enumerate(loss_task):
                    tb_manager.update_graph(step, {f"{prefix}_loss_{i}_{j}": loss})
            else:
                tb_manager.update_graph(step, {f"{prefix}_loss_{i}_avg": loss_task})
        for i, mid_grad in enumerate(mid_grad_list):
            if isinstance(mid_grad, list):
                for j, mid_grad_row in enumerate(mid_grad):
                    tb_manager.summary_writer.add_histogram(
                        f"{prefix}_mid_grad_{i + 1}_{j}",
                        mid_grad_row,
                        step,
                    )
            else:
                tb_manager.summary_writer.add_histogram(
                    f"{prefix}_mid_grad_{i + 1}_avg",
                    mid_grad,
                    step,
                )

    elif method == "Hierarchical_ProtoNet" or method == "ProtoNet":
        tb_manager.update_graph(step, {f"{mode}_loss": loss_list})
        tb_manager.update_graph(step, {f"{mode}_acc": acc_list})


if __name__ == "__main__":
    # init args
    args = Options().parse()

    # init random seed
    torch.cuda.cudnn_enabled = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    tb_manager = Tensorboard_manager(args.tb_dir)

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    import shutil

    # shutil.rmtree(args.model_dir)
    Path(args.config_dir).mkdir(parents=True, exist_ok=True)

    with open(args.config_path, "w+") as f:
        f.write(str(args.__dict__))

    # init device
    device = torch.device("cuda:" + args.gpu_id)
    args.device = device

    # init dataset
    args.datasets = args.dataset
    args.max_test_task = 600
    args.num_ways = args.n_way
    args.num_shots = args.k_spt
    args.num_shots_test = args.k_qry
    args.batch_size = 1
    from data.dataset_utils import DatasetEnum

    clusters = DatasetEnum[args.datasets].get_clusters()
    if args.domains:
        _cluster_names = {d.name:d for d in clusters}
        print(_cluster_names)
        for i, d in enumerate(args.domains):
            if d in list(_cluster_names.keys()):
                args.domains[i] = _cluster_names[d]
            else:
                raise ValueError(f"Unknown domain: {d}")
    else:
        args.domains = clusters
    clusters = args.domains
    if not args.test_domains:
        args.test_domains = [cl.name for cl in clusters]
    if args.check_clusters:
        print(clusters)
        exit()
    accs_per_domain = {}
    last_test_acc_per_domain = {}

    # Print settings (N-way, K-shot...)
    print(f"N-way: {args.num_ways}")
    print(f"K-shot: {args.num_shots}")
    print(f"K-query: {args.num_shots_test}")

    for i, test_domain in enumerate(clusters):
        if test_domain.name not in args.test_domains:
            continue
        print(f"Test domain: {test_domain.name}")

        args.test_domain = test_domain
        trainer = Trainer(args)
        if (not args.test) and (args.holdout or i == 0) and (trainer.trainable):
            acc = trainer.train()
            if not args.fast_mode:
                last_test_acc_per_domain[test_domain.name] = acc
        else:
            print("Skip training")
        accs_per_domain[test_domain.name] = trainer.test(-1, "test", load_best=True)[1]

    if last_test_acc_per_domain != {}:
        for test_domain in list(last_test_acc_per_domain.keys()):
            print(
                f"Test accuracy (last) for {test_domain}: {last_test_acc_per_domain[test_domain]}"
            )
        print(
            f"Average accuracy (last): {np.mean(list(last_test_acc_per_domain.values()))}"
        )

    for test_domain in clusters:
        if test_domain.name not in accs_per_domain:
            continue
        print(
            f"Test accuracy for {test_domain.name}: {accs_per_domain[test_domain.name]}"
        )
    print(f"Average accuracy: {np.mean(list(accs_per_domain.values()))}")
