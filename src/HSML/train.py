from .model import Model
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict
# from torchmeta.utils.gradient_based import gradient_update_parameters

def get_accuracy(logits, targets):
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())

def gradient_update_parameters(model, loss, params=None, step_size=0.5, first_order=False):
    if params is None:
        params = OrderedDict(model.meta_named_parameters())
    grads = torch.autograd.grad(loss,
                                params.values(),
                                create_graph=not first_order,
                                retain_graph=True,
                                allow_unused=True)

    updated_params = OrderedDict()

    for (name, param), grad in zip(params.items(), grads):
        if grad is None:
            grad = 0.0
        updated_params[name] = param - step_size * grad

    return updated_params


class MetaLearning:
    def __init__(self, args):
        self.inner_steps = args.inner_steps
        self.meta_lr = args.meta_lr
        self.inner_lr = args.inner_lr
        self.first_order = args.first_order
        self.model = Model(encoder_type="lstm", out_features=args.n_way)
        self.meta_optim = optim.Adam(self.model.parameters(), lr=self.meta_lr)
        self.gamma = 0.01

    def __call__(self, batch, mode):
        if mode == "train":
            return self.epoch(batch)
        elif mode == "test" or mode == "valid":
            return self.test(batch)
        else:
            raise ValueError("Undefined Action")

    def epoch(self, batch):
        outer_loss = 0.0
        accuracy = 0
#         loss_list = [0.0 for _ in range(self.inner_steps + 1)]
#         accuracy_list = [0.0 for _ in range(self.inner_steps + 1)]

        for i, (spt_x, spt_y, qry_x, qry_y) in enumerate(batch):
#             recon_x, input_task_emb, gate = self.model(spt_x, spt_y)
# 
#             recon_loss = self.model.encoder.reconstruction_loss(recon_x.squeeze(1), input_task_emb)
#             # outer_loss += recon_loss * self.gamma
# 
#             updated_param = OrderedDict()
#             for (name, param), gate_per_param in zip(self.model.meta_learner.meta_named_parameters(), gate):
#                 gate_per_param = gate_per_param.reshape(param.shape)
#                 updated_param[name] = torch.mul(param, gate_per_param)
            updated_param = None
            qry_logit = self.model.meta_learner(qry_x, params=updated_param)
            loss = F.cross_entropy(qry_logit, qry_y)
#             loss_list[0] += loss.detach().cpu().item() / (len(batch))
#             with torch.no_grad():
#                 accuracy_list[0] = get_accuracy(qry_logit, qry_y).detach().cpu() / (len(batch))

            for j in range(self.inner_steps):
                logit = self.model.meta_learner(spt_x, params=updated_param)
                inner_loss = F.cross_entropy(logit, spt_y)
                self.model.meta_learner.zero_grad()
                updated_param = gradient_update_parameters(self.model.meta_learner,
                                                           inner_loss,
                                                           step_size=self.inner_lr,
                                                           first_order=self.first_order
                                                           )

                qry_logit = self.model.meta_learner(qry_x, params=updated_param)
                loss = F.cross_entropy(qry_logit, qry_y)
#                 loss_list[j+1] += loss.clone().detach().cpu().item() / (len(batch))
#                 with torch.no_grad():
#                     accuracy_list[j+1] = get_accuracy(qry_logit, qry_y).detach().cpu() / (len(batch))
            outer_loss += loss

            with torch.no_grad():
                accuracy += get_accuracy(qry_logit, qry_y)

        outer_loss.div_(len(batch))
        accuracy.div_(len(batch))

        self.meta_optim.zero_grad()
        outer_loss.backward()
        self.meta_optim.step()

#         print(loss_list)
        loss_list = [outer_loss.detach().cpu().item()]
        accuracy_list = [accuracy.detach().cpu()]
        return loss_list, accuracy_list, None

    def test(self, batch):
        outer_loss = 0.0
        accuracy = 0.0
        self.model.eval()

        for (spt_x, spt_y, qry_x, qry_y) in batch:
            recon_x, input_task_emb, gate = self.model(spt_x, spt_y)

            recon_loss = self.model.encoder.reconstruction_loss(recon_x.squeeze(1), input_task_emb)
            outer_loss += recon_loss

            updated_param = OrderedDict()
            for (name, param), gate_per_param in zip(self.model.meta_learner.meta_named_parameters(), gate):
                gate_per_param = gate_per_param.reshape(param.shape)
                updated_param[name] = torch.mul(param, gate_per_param)

            # meta_optim = optim.Adam(updated_param, lr=self.meta_lr)

            for _ in range(self.inner_steps):
                logit = self.model.meta_learner(spt_x, params=updated_param)
                inner_loss = F.cross_entropy(logit, spt_y)
                self.model.meta_learner.zero_grad()
                updated_param = gradient_update_parameters(self.model.meta_learner,
                                                           inner_loss,
                                                           step_size=self.inner_lr,
                                                           first_order=self.first_order
                                                           )

            qry_logit = self.model.meta_learner(qry_x, params=updated_param)
            outer_loss += F.cross_entropy(qry_logit, qry_y)

            accuracy += get_accuracy(qry_logit, qry_y)

        outer_loss.div_(len(batch))
        accuracy.div_(len(batch))

        return [outer_loss.detach().cpu().item()], [accuracy.detach().cpu()], None 

