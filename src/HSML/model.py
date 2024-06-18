import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import *
from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d, MetaBatchNorm2d, MetaLinear)
from typing import *


class ImageEmbedding(nn.Module):
    def __init__(self, hidden_num, channels, k=5):
        super(ImageEmbedding, self).__init__()
        self.hidden_num = hidden_num
        self.channels = channels

        self.features = nn.Sequential(
            nn.Conv2d(self.channels, self.hidden_num, kernel_size=k, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(size=4, alpha=0.001 / 9.0, beta=0.75),
            nn.Conv2d(self.hidden_num, self.hidden_num, kernel_size=k, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=4, alpha=0.001 / 9.0, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Fully connected layers setup is deferred to the forward method due to dynamic sizing
        self.fc = None

    def forward(self, images):
        features = self.features(images)

        flat = features.view(images.size(0), -1)
        if self.fc is None:
            self.fc = nn.Sequential(
                nn.Linear(flat.size(1), 384).cuda(),
                nn.ReLU(),
                nn.Linear(384, 64).cuda(),
                nn.ReLU(),
            ).to(images.device)
        out = self.fc(flat)
        return out


class AutoEncoder(nn.Module, metaclass=ABCMeta):
    reconstruction_loss = nn.MSELoss()


class LSTMAutoencoder(AutoEncoder):
    def __init__(self, hidden_num, reverse=True, decode_without_input=False):
        super(LSTMAutoencoder, self).__init__()
        self.hidden_num = hidden_num
        self.reverse = reverse
        self.decode_without_input = decode_without_input

        self.enc_cell = nn.GRU(69, 40, batch_first=True)
        self.dec_cell = nn.GRU(456, 69, batch_first=True)
        self.dec_weight = nn.Parameter(torch.randn(42, 69))
        self.dec_bias = nn.Parameter(torch.zeros(69))

    def forward(self, x):
        _, hidden = self.enc_cell(x)

        if self.decode_without_input:
            dec_input = torch.zeros_like(x)
        else:
            dec_input = x

        outputs, _ = self.dec_cell(dec_input, hidden)

        outputs = torch.matmul(outputs, self.dec_weight) + self.dec_bias

        return outputs, hidden

class LSTMAutoencoder(AutoEncoder):
    def __init__(self, input_num, hidden_num, batch_size, cell=None, reverse=True, decode_without_input=False):
        super(LSTMAutoencoder, self).__init__()
        if cell is None:
            self.enc_cell = nn.GRUCell(input_num, hidden_num)
            self.dec_cell = nn.GRUCell(input_num, hidden_num)
        else:
            self.enc_cell = cell
            self.dec_cell = cell
        self.reverse = reverse
        self.decode_without_input = decode_without_input
        self.hidden_num = hidden_num

        self.elem_num =  input_num #+ args.num_classes

        self.dec_weight = nn.Parameter(torch.randn(self.hidden_num, self.elem_num))
        self.dec_bias = nn.Parameter(torch.ones(self.elem_num) * 0.1)
        self.batch_size = batch_size

    def forward(self, inputs):

        inputs = torch.unsqueeze(inputs, 0)
        inputs = torch.unbind(inputs, dim=1)

        enc_state = torch.zeros(1, self.enc_cell.hidden_size).cuda() # Hidden state

        enc_states = []
        for input_step in inputs:
            input_step = input_step.view(1,-1).cuda()
            enc_state = self.enc_cell(input_step, enc_state)
            enc_states.append(enc_state)
        self.enc_states = enc_states

        dec_states = []
        if self.decode_without_input:
            dec_inputs = torch.zeros_like(inputs[0])
            for enc_state in enc_states:
                dec_inputs, dec_state = self.dec_cell(dec_inputs, enc_state)
                dec_states.append(dec_state)
            if self.reverse:
                dec_states = dec_states[::-1]
            dec_outputs = torch.stack(dec_states, dim=0)
            dec_weight = self.dec_weight.unsqueeze(0).expand(self.batch_size, -1, -1)
            self.output_ = torch.matmul(dec_weight, dec_outputs) + self.dec_bias.unsqueeze(0).expand(self.batch_size, -1)
        else:
            dec_outputs = []
            for step in range(len(inputs)):
                dec_input = torch.zeros_like(inputs[0]).view(1,-1)
                dec_input = self.dec_cell(dec_input, enc_states[step])
                dec_output = torch.matmul(dec_input, self.dec_weight) + self.dec_bias
                dec_outputs.append(dec_output)
            if self.reverse:
                dec_outputs = dec_outputs[::-1]
            self.output_ = torch.stack(dec_outputs, dim=0)

        self.input_ = torch.stack(inputs, dim=1)
        self.loss = torch.mean(torch.square(self.input_ - self.output_))
        self.emb_all = torch.mean(torch.stack(self.enc_states, dim=0), dim=0)

        return self.output_, self.emb_all



class MeanAutoencoder(AutoEncoder):
    def __init__(self, input_dim, hidden_num_mid, hidden_num):
        super(MeanAutoencoder, self).__init__()
#         self.hidden_num_mid = hidden_num_mid
#         self.hidden_num = hidden_num

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_num_mid),
            nn.ReLU(),
            nn.Linear(hidden_num_mid, hidden_num),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_num, hidden_num_mid),
            nn.ReLU(),
            nn.Linear(hidden_num_mid, input_dim),
        )

    def forward(self, x):
        hidden = self.encoder(x)
        recon_x = self.decoder(x)

        return recon_x, hidden
#         emb_pool = torch.mean(enc_dense2, dim=0, keepdim=True)
#         loss = 0.5 * torch.mean((x - reconstructed) ** 2)
#         return emb_pool, loss


def conv3x3(in_channels, out_channels, **kwargs):
    return MetaSequential(
        MetaConv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        MetaBatchNorm2d(out_channels, momentum=1., track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class HierarchicalClustering(nn.Module):
    def __init__(self, num_clusters: List[int], dim: int):
        super(HierarchicalClustering, self).__init__()
        self.encoders = []
        self.prototypes = []
        for i, num_clusters_per_layer in enumerate(num_clusters):
            encoders_per_layer = []
            prototypes_per_layer = []
            for cluster_num in range(num_clusters_per_layer):
                encoders_per_layer.append(
                    nn.Sequential(
                        nn.Linear(dim, dim),
                        nn.Tanh(),
                    )
                )
                prototypes_per_layer.append(nn.Parameter(torch.rand(dim)))
            self.encoders.append(encoders_per_layer)
            self.prototypes.append(prototypes_per_layer)

    def get_proba(self, x, layer_num):
        def get_single_score(x, prototype, sigma=10.0):
            prototype = prototype.to(x.device)
            distances = torch.sum((x.unsqueeze(1) - prototype.unsqueeze(0)) ** 2, dim=2)
            scaled_distances = -distances / (sigma ** 2)
            softmax_scores = F.softmax(scaled_distances, dim=1)
            return softmax_scores

        probs = []
        for prototype in self.prototypes[layer_num]:
            probs.append(get_single_score(x, prototype))

        probs = torch.stack(probs, dim=0)

        return probs

    def get_transform(self, x, layer_num):
        transform = []
        for encoder in self.encoders[layer_num]:
            encoder = encoder.to(x.device)
            transform.append(encoder(x))

        transform = torch.stack(transform, dim=0)
        return transform

    def forward(self, x):
        embedding = x
        for i in range(len(self.encoders)):
            probs = self.get_proba(x, i)
            transform = self.get_transform(x, i)
            embedding = torch.matmul(probs, transform)

        return embedding


class Conv4(MetaModule):
    def __init__(self, in_channels, out_features, hidden_size=64):
        super(Conv4, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size

        self.features = MetaSequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size)
        )

        self.classifier = MetaLinear(hidden_size * 5 * 5, out_features)

    def forward(self, x, params=None):
        features = self.features(x, params=self.get_subdict(params, 'features'))
        features = features.view((features.size(0), -1))
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits


class Model(nn.Module):
    def __init__(self, encoder_type: str, out_features: int):
        super(Model, self).__init__()
        hidden_dim = 128 # As specified in the maml.py of the original github code (FLAG.hidden_dim)
        self.out_features = out_features # n_way
        self.img_size = 84

        self.image_encoder = ImageEmbedding(hidden_num=64, channels=3)

        if encoder_type == "mean":
            hidden_dim_mid = 96 # As specified in the original github code.
            self.encoder = MeanAutoencoder(64, hidden_dim_mid, hidden_dim)
        elif encoder_type == "lstm":
            self.encoder = LSTMAutoencoder(69, hidden_dim, batch_size=25)

        self.cluster = HierarchicalClustering(num_clusters=[4, 2, 1], dim=hidden_dim)

        self.meta_learner = Conv4(3, out_features, hidden_size=32)

        self._gate = [
            nn.Sequential(
                nn.Linear(2 * hidden_dim, param.numel()),
                nn.Sigmoid(),
            ) for param in self.meta_learner.meta_parameters()
        ]

    def get_gate(self, g, h):
        concated_embedding = torch.cat((g, h.squeeze(1)), axis=-1)
        gate = []
        for _g in self._gate:
            _g = _g.to(g.device)
            gate.append(_g(concated_embedding))
        return gate

    def forward(self, x, y):
        image_embedding = self.image_encoder(x.reshape([-1, 3, self.img_size, self.img_size])).cuda()
        one_hot_labels = F.one_hot(y, num_classes=self.out_features)
        input_task_emb = torch.cat([image_embedding, one_hot_labels], dim=-1).cuda()
        # 4.1 Task Representation Learning
        recon_x, hidden = self.encoder(input_task_emb)

        # 4.2 Hierarchical Task Clustering
        embedding = self.cluster(hidden)
        # 4.3 Knowledge Adaptation
        gate = self.get_gate(hidden, embedding)

        return recon_x, input_task_emb, gate
