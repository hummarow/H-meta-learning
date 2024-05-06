from argparse import ArgumentParser
import argparse

def parse_args(default=False):
    """Command-line argument parser for train."""

    parser = ArgumentParser(
        description='PyTorch implementation of Multi-MAML'
    )

    argparser = argparse.ArgumentParser()
    
    argparser.add_argument("--epoch", type=int, help="number of training tasks", default=60000)
    # argparser.add_argument("--num_maml", type=int, help="n way", default=1)
    argparser.add_argument("--num_ways", type=int, help="n way", default=5)
    argparser.add_argument("--num_shots", type=int, help="k shot for support set", default=5)
    argparser.add_argument("--num_shots_test", type=int, help="k shot for query set", default=15)
    argparser.add_argument("--imgc", type=int, help="imgc", default=3)
    argparser.add_argument("--filter_size", type=int, help="size of filters of convblock", default=64)
    argparser.add_argument("--batch_size", type=int, help="meta batch size: task num", default=2)
    argparser.add_argument("--max_test_task", type=int, help="validation/test task number", default=600)
    argparser.add_argument("--meta_lr", type=float, help="outer-loop learning rate", default=1e-3)
    argparser.add_argument("--multi_meta_lr", type=float, help="multi maml outer-loop learning rate", default=1e-3)
    argparser.add_argument("--update_lr", type=float, help="inner-loop update learning rate", default=1e-2)
    argparser.add_argument("--update_step", type=int, help="inner-loop update steps", default=3)
    argparser.add_argument("--update_step_test", type=int, help="update steps for finetunning", default=8)
    argparser.add_argument("--update", type=str, help="update method: maml, anil, boil", default="maml")
    argparser.add_argument("--temp_scaling", type=float, help="temperature scaling while softmaxing", default=2)
    argparser.add_argument("--dropout", type=float, help="dropout probability", default=0.)
    argparser.add_argument("--regularizer", type=float, help="dot product regularizer", default=5e-4)
    argparser.add_argument("--lam", type=float, help="covariance", default=0.005)
    argparser.add_argument("--max", type=int, help="counting overfitting", default=4)
    argparser.add_argument("--threshold", type=int, help="c", default=500)
    argparser.add_argument("--gpu_id", type=int, help="gpu device number", default=3)
    argparser.add_argument("--maxlen", type=int, help="max length of deque", default=2)
    argparser.add_argument("--resume", type=bool, help="resume training", default=False)
    argparser.add_argument("--model", type=str, help="model architecture", default="conv4")
    argparser.add_argument("--datasets", type=str, help="meta dataset", default="MetaCIO")
    argparser.add_argument("--multi", action='store_true', help="apply XB-MAML")
    argparser.add_argument("--first_order", action='store_true', help="apply XB-MAML")
    argparser.add_argument("--version", type=str, help="version", default="0")
    args = argparser.parse_args()

    if default:
        return parser.parse_args('')
    else:
        return args
    
    
    
    