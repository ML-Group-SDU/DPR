import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torchvision.models as models
import utils


def index_batch(batch, indices):
    indexed = {}
    for key in batch.keys():
        indexed[key] = batch[key][indices, ...]
    return indexed


def gen_net(in_size=1, out_size=1, H=128, n_layers=3, activation='tanh'):
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(in_size, H))
        net.append(nn.LeakyReLU())
        in_size = H
    net.append(nn.Linear(in_size, out_size))
    if activation == 'tanh':
        net.append(nn.Tanh())
    elif activation == 'sig':
        net.append(nn.Sigmoid())
    else:
        pass

    return net


class RewardModel(object):
    def __init__(self, task, observation_dim, action_dim, ensemble_size=3, lr=3e-4, activation="tanh", logger=None,
                 device="cuda"):
        self.task = task
        self.observation_dim = observation_dim  # state: env.observation_space.shape[0]
        self.action_dim = action_dim  # state: env.action_space.shape[0]
        self.ensemble_size = ensemble_size  # ensemble_size
        self.lr = lr  # learning rate
        self.logger = logger  # logger
        self.device = torch.device(device)

        # build network
        self.opt = None
        self.activation = activation
        self.ensemble = []
        self.paramlst = []
        self.construct_ensemble()

    def construct_ensemble(self):
        for i in range(self.ensemble_size):
            model = nn.Sequential(*gen_net(in_size=self.observation_dim + self.action_dim,
                                           out_size=1, H=256, n_layers=3,
                                           activation=self.activation)).float().to(self.device)
            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())

        self.opt = torch.optim.Adam(self.paramlst, lr=self.lr)

    def save_model(self, path):
        state_dicts = [model.state_dict() for model in self.ensemble]
        torch.save(state_dicts, path)

    def load_model(self, path):
        state_dicts = torch.load(path, map_location='cpu')
        for model, state_dict in zip(self.ensemble, state_dicts):
            model.load_state_dict(state_dict)
            model.to(self.device)

    def train(self, n_epochs, pref_dataset, data_size, batch_size):
        interval = int(data_size / batch_size) + 1

        for epoch in range(1, n_epochs + 1):
            ensemble_losses = [[] for _ in range(self.ensemble_size)]
            ensemble_acc = [[] for _ in range(self.ensemble_size)]

            batch_shuffled_idx = []
            for _ in range(self.ensemble_size):
                batch_shuffled_idx.append(np.random.permutation(pref_dataset["observations"].shape[0]))

            for i in range(interval):
                self.opt.zero_grad()
                total_loss = 0
                start_pt = i * batch_size
                end_pt = min((i + 1) * batch_size, pref_dataset["observations"].shape[0])
                for member in range(self.ensemble_size):
                    # get batch
                    batch = index_batch(pref_dataset, batch_shuffled_idx[member][start_pt:end_pt])
                    # compute loss
                    curr_loss, correct = self._train(batch, member)
                    total_loss += curr_loss
                    ensemble_losses[member].append(curr_loss.item())
                    ensemble_acc[member].append(correct)
                total_loss.backward()
                self.opt.step()

            train_metrics = {"epoch": epoch,
                             "avg_loss": np.mean(ensemble_losses),
                             "avg_acc": np.mean(ensemble_acc)}
            for i in range(self.ensemble_size):
                train_metrics.update({f"ensemble_{i}_loss": np.mean(ensemble_losses[i])})
                train_metrics.update({f"ensemble_{i}_acc": np.mean(ensemble_acc[i])})
            self.logger.log(train_metrics)
            if epoch % 50 == 1:
                self.save_model(self.logger._model_dir / f"reward_model_{epoch}.pt")

            # early stop
            if np.mean(ensemble_acc) > 0.968 and "antmaze" not in self.task:
                break

    def _train(self, batch, member):
        # get batch
        obs_1 = batch['observations']  # batch_size * len_query * obs_dim
        act_1 = batch['actions']  # batch_size * len_query * action_dim
        obs_2 = batch['observations_2']
        act_2 = batch['actions_2']
        labels = batch['labels']  # batch_size * 2 (one-hot, for equal label)
        s_a_1 = np.concatenate([obs_1, act_1], axis=-1)
        s_a_2 = np.concatenate([obs_2, act_2], axis=-1)

        # get comparable labels
        comparable_indices = np.where((labels != [0.5, 0.5]).any(axis=1))[0]
        comparable_labels = torch.from_numpy(np.argmax(labels, axis=1)).to(self.device)

        # get logits
        r_hat1 = self.r_hat_member(s_a_1, member)  # batch_size * len_query * 1
        r_hat2 = self.r_hat_member(s_a_2, member)
        r_hat1 = r_hat1.sum(axis=1)  # batch_size * 1
        r_hat2 = r_hat2.sum(axis=1)
        r_hat = torch.cat([r_hat1, r_hat2], axis=1)  # batch_size * 2

        # get labels
        # labels = torch.from_numpy(labels).long().to(self.device)  # TODO
        labels = torch.from_numpy(labels).to(self.device)

        # compute loss
        curr_loss = self.softXEnt_loss(r_hat, labels)

        # compute acc
        _, predicted = torch.max(r_hat.data, 1)

        if not len(comparable_indices):
            correct = 0.7  # TODO, for exception
        else:
            correct = (predicted[comparable_indices] == comparable_labels[comparable_indices]).sum().item() / len(
                comparable_indices)
        return curr_loss, correct

    def r_hat_member(self, x, member):
        return self.ensemble[member](torch.from_numpy(x).float().to(self.device))

    def get_reward_batch(self, x):
        # they say they average the rewards from each member of the ensemble,
        # but I think this only makes sense if the rewards are already normalized.
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.ensemble_size):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)

        return np.mean(r_hats, axis=0)

    def softXEnt_loss(self, input, target):
        logprobs = nn.functional.log_softmax(input, dim=1)
        return -(target * logprobs).sum() / input.shape[0]


class CNNRewardModel(RewardModel):
    def __init__(self, task, observation_dim, action_dim, ensemble_size=3, lr=3e-4, activation=None, logger=None,
                 device="cpu"):
        super().__init__(task, observation_dim, action_dim, ensemble_size, lr, activation, logger, device)
        # observation_dim (84, 84)
        # action_dim 6

    def construct_ensemble(self):
        for i in range(self.ensemble_size):
            model = nn.Sequential(
                # CNN Layers
                nn.Conv2d(1, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),

                # MLP layers
                *gen_net(in_size=7 * 7 * 64,
                         out_size=1, H=256, n_layers=3,
                         activation=None)
            ).float().to(self.device)

            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())

        self.opt = torch.optim.Adam(self.paramlst, lr=self.lr)

    def r_hat_mlp_member(self, x, member):
        # x should have shape: (batch_size, len_query, 7*7*64 + action_dim)
        return self.ensemble[member][7:](x)

    def r_hat_conv_member(self, observation, member):
        # x should have shape: (batch_size, len_query, 1, 84, 84)
        batch_size = observation.shape[0]
        len_query = observation.shape[1]
        obs_dim = observation.shape[-3:]
        observation = torch.from_numpy(observation).float().to(self.device) / 255.0
        observation = observation.view((-1,) + obs_dim)
        embedding = self.ensemble[member][:7](observation).view(batch_size, len_query, -1)
        return embedding

    def _train(self, batch, member):
        # get batch
        batch_size = batch['observations'].shape[0]
        obs_1 = batch['observations']  # batch_size * len_query * 1 * obs_dim
        act_1 = batch['actions']  # batch_size * len_query * action_dim
        obs_2 = batch['observations_2']
        act_2 = batch['actions_2']
        labels = batch['labels']  # batch_size * 2 (one-hot, for equal label)

        s_embedding_1 = self.r_hat_conv_member(obs_1, member)  # batch_size * len_query * (7*7*64)
        s_embedding_2 = self.r_hat_conv_member(obs_2, member)  # batch_size * len_query * (7*7*64)

        # get comparable labels
        comparable_indices = np.where((labels != [0.5, 0.5]).any(axis=1))[0]
        comparable_labels = torch.from_numpy(np.argmax(labels, axis=1)).to(self.device)

        # get logits
        r_hat1 = self.r_hat_mlp_member(s_embedding_1, member)  # batch_size * len_query * 1
        r_hat2 = self.r_hat_mlp_member(s_embedding_2, member)
        r_hat1 = r_hat1.sum(axis=1)  # batch_size * 1
        r_hat2 = r_hat2.sum(axis=1)
        r_hat = torch.cat([r_hat1, r_hat2], axis=1)  # batch_size * 2

        # get labels
        # labels = torch.from_numpy(labels).long().to(self.device)  # TODO
        labels = torch.from_numpy(labels).to(self.device)

        # compute loss
        curr_loss = self.softXEnt_loss(r_hat, labels)

        # compute acc
        _, predicted = torch.max(r_hat.data, 1)
        if not len(comparable_indices):
            correct = 0.7  # TODO, for exception
        else:
            correct = (predicted[comparable_indices] == comparable_labels[comparable_indices]).sum().item() / len(
                comparable_indices)
        return curr_loss, correct


    def split_train(self, n_epochs, pref_dataset, data_size, batch_size):
        N_DATASET_PARTITION = len(pref_dataset)
        for n_epoch in range(n_epochs):
            ensemble_losses = [[] for _ in range(self.ensemble_size)]
            ensemble_acc = [[] for _ in range(self.ensemble_size)]

            for partition_idx in range(N_DATASET_PARTITION):
                pref_dataset_partition = pref_dataset[partition_idx]
                data_size = pref_dataset_partition["observations"].shape[0]

                interval = int(data_size / batch_size) + 1

                batch_shuffled_idx = []
                for _ in range(self.ensemble_size):
                    batch_shuffled_idx.append(np.random.permutation(pref_dataset_partition["observations"].shape[0]))

                for i in range(interval):
                    self.opt.zero_grad()
                    total_loss = 0
                    start_pt = i * batch_size
                    end_pt = min((i + 1) * batch_size, pref_dataset_partition["observations"].shape[0])
                    for member in range(self.ensemble_size):
                        # get batch
                        batch = index_batch(pref_dataset_partition, batch_shuffled_idx[member][start_pt:end_pt])
                        # compute loss
                        curr_loss, correct = self._train(batch, member)
                        total_loss += curr_loss
                        ensemble_losses[member].append(curr_loss.item())
                        ensemble_acc[member].append(correct)

                    total_loss.backward()
                    self.opt.step()

            train_metrics = {"epoch": n_epoch,
                             "avg_loss": np.mean(ensemble_losses),
                             "avg_acc": np.mean(ensemble_acc)}
            for i in range(self.ensemble_size):
                train_metrics.update({f"ensemble_{i}_loss": np.mean(ensemble_losses[i])})
                train_metrics.update({f"ensemble_{i}_acc": np.mean(ensemble_acc[i])})
            self.logger.log(train_metrics)
    
    def get_reward_batch(self, state):
        # they say they average the rewards from each member of the ensemble,
        # but I think this only makes sense if the rewards are already normalized.
        # but I don't understand how the normalization should be happening right now :(
        state = np.expand_dims(state, axis=0)
        r_hats = []
        for member in range(self.ensemble_size):
            s_embedding = self.r_hat_conv_member(state, member)
            r_hats.append(self.r_hat_mlp_member(s_embedding, member).detach().cpu().numpy())
        r_hats = np.array(r_hats)

        return np.mean(r_hats, axis=0)


class TransformerRewardModel(RewardModel):
    def __init__(self,
                 task, observation_dim, action_dim, structure_type="transformer1",
                 ensemble_size=3, lr=0.0003, activation="tanh",
                 d_model=256, nhead=4, num_layers=1, max_seq_len=100,
                 logger=None, device="cuda"):
        self.structure_type = structure_type
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        super().__init__(
            task, observation_dim, action_dim,
            ensemble_size, lr, activation, logger, device)
        
    def construct_ensemble(self):
        for i in range(self.ensemble_size):
            if self.structure_type == "transformer1":
                transformer = utils.PrefTransformer1(
                    self.observation_dim, self.action_dim,
                    self.max_seq_len,
                    self.d_model, self.nhead, self.num_layers
                )
            elif self.structure_type == "transformer2":
                transformer = utils.PrefTransformer2(
                    self.observation_dim, self.action_dim,
                    self.d_model, self.nhead, self.num_layers
                )
            elif self.structure_type == "transformer3":
                transformer = utils.PrefTransformer3(
                    self.observation_dim, self.action_dim,
                    self.d_model, self.nhead, self.num_layers
                )
            elif self.structure_type == "transformer4":
                transformer = utils.PrefTransformer4(
                    self.observation_dim, self.action_dim,
                    self.max_seq_len,
                    self.d_model, self.nhead, self.num_layers
                )
            elif self.structure_type == "transformer5":
                transformer = utils.PrefTransformer5(
                    self.observation_dim, self.action_dim,
                    self.max_seq_len,
                    self.d_model, self.nhead, self.num_layers
                )
            else:
                raise NotImplementedError

            self.ensemble.append(transformer.to(self.device))
            self.paramlst.extend(self.ensemble[-1].parameters())

        self.opt = torch.optim.Adam(self.paramlst, lr=self.lr)

    def _train(self, batch, member):
        # get batch
        obs_1 = batch['observations']  # batch_size * len_query * obs_dim
        act_1 = batch['actions']  # batch_size * len_query * action_dim
        obs_2 = batch['observations_2']
        act_2 = batch['actions_2']
        labels = batch['labels']  # batch_size * 2 (one-hot, for equal label)

        # to_torch
        obs_1 = utils.to_torch(obs_1).to(self.device)
        act_1 = utils.to_torch(act_1).to(self.device)
        obs_2 = utils.to_torch(obs_2).to(self.device)
        act_2 = utils.to_torch(act_2).to(self.device)

        # get comparable labels
        comparable_indices = np.where((labels != [0.5, 0.5]).any(axis=1))[0]
        comparable_labels = torch.from_numpy(np.argmax(labels, axis=1)).to(self.device)

        # get logits
        r_hat1 = self.ensemble[member](obs_1, act_1)  # batch_size * len_query
        r_hat2 = self.ensemble[member](obs_2, act_2)
        
        r_hat1 = r_hat1.mean(-1, keepdim=True)  # batch_size * 1
        r_hat2 = r_hat2.mean(-1, keepdim=True)
        r_hat = torch.cat([r_hat1, r_hat2], axis=-1)  # batch_size * 2
        
        p_1_2 = 1./(1.+torch.exp(r_hat2-r_hat1)) # batch_size * 1
        y = utils.to_torch(labels[:, :1], dtype=torch.float32).to(self.device) # batch_size * 1

        weights = torch.ones_like(y)
        weights[torch.where(y==0.5)] = 0.0
        
        curr_loss = - (weights*(y*torch.log(p_1_2+1e-8) + (1-y)*torch.log(1-p_1_2+1e-8))).mean()
        
        # labels = utils.to_torch(labels, dtype=torch.long).to(self.device)

        # # compute loss
        # curr_loss = self.softXEnt_loss(r_hat, labels)

        # compute acc
        _, predicted = torch.max(r_hat.data, 1)

        if not len(comparable_indices):
            correct = 0.7  # TODO, for exception
        else:
            correct = (predicted[comparable_indices] == comparable_labels[comparable_indices]).sum().item() / len(
                comparable_indices)
            
        return curr_loss, correct

class DdpmRewardModel(RewardModel):
    def __init__(self, task, observation_dim, action_dim, max_action, beta_schedule, n_timesteps, device, disc_hid_dim,
                 disc_momentum, lr=0.0003, clamp_magnitude=10.0, ftb=5, ensemble_size=3, logger=None, activation=None):
        self.max_action = max_action
        self.beta_schedule = beta_schedule
        self.n_timesteps = n_timesteps
        self.disc_hid_dim = disc_hid_dim
        self.clamp_magnitude = clamp_magnitude
        self.disc_momentum = disc_momentum
        self.ftb = ftb
        super().__init__(task, observation_dim, action_dim, ensemble_size, lr, activation, logger, device)

    def construct_ensemble(self):
        for i in range(self.ensemble_size):
            model = utils.DDPM_RM(
                self.observation_dim + self.action_dim, self.action_dim, self.max_action,
                n_timesteps=self.n_timesteps,
                beta_schedule=self.beta_schedule,
                disc_hid_dim=self.disc_hid_dim,
                device=self.device,
                lr=self.lr,
                disc_momentum=self.disc_momentum,
                clamp_magnitude=self.clamp_magnitude,
            )

            self.ensemble.append(model.to(self.device))
            self.paramlst.extend(self.ensemble[-1].parameters())
        self.opt = torch.optim.Adam(self.paramlst, lr=self.lr, betas=(self.disc_momentum, 0.999))

    def disc_reward(self, x, member):
        batch = x.to(self.device)
        disc_cost = self.ensemble[member].diffusion.calc_reward(batch)
        return disc_cost

    def compute_loss(self, x, grad=False):
        batch = x.to(self.device)
        total_loss = 0
        for member in range(self.ensemble_size):
            total_loss += self.ensemble[member].diffusion.compute_loss(batch, grad=grad)

        avg_loss = total_loss.mean()
        return avg_loss

    def train_rm(self, n_epochs, pref_dataset, data_size, batch_size, loss_type='disc', generate_model=None):
        interval = int(data_size / batch_size) + 1
        train_metrics = [
            ['epoch', 'train_loss', 'train_acc']
        ]

        for epoch in range(1, n_epochs + 1):
            ensemble_losses = [[] for _ in range(self.ensemble_size)]
            ensemble_acc = [[] for _ in range(self.ensemble_size)]
            batch_shuffled_idx = []
            len_dataset = pref_dataset["observations"].shape[0]

            for _ in range(self.ensemble_size):
                batch_shuffled_idx.append(np.random.permutation(len_dataset))

            curr_loss_list = []
            curr_acc_list = []

            for i in range(interval):
                total_loss = 0
                self.opt.zero_grad()
                start_pt = i * batch_size
                end_pt = min((i + 1) * batch_size, len_dataset)

                for member in range(self.ensemble_size):
                    batch = index_batch(pref_dataset, batch_shuffled_idx[member][start_pt:end_pt])
                    # print("member: ",member)
                    # print(batch.shape)
                    curr_loss, correct = self._train(batch, member)
                    total_loss += curr_loss
                    ensemble_losses[member].append(curr_loss.item())
                    ensemble_acc[member].append(correct)

                    curr_loss_list.append(curr_loss.item())
                    curr_acc_list.append(correct)

                total_loss.backward()
                self.opt.step()
            train_metrics.append([epoch, np.mean(curr_loss_list), np.mean(curr_acc_list)])

            train_metrics_2 = {"epoch": epoch,
                               "avg_loss": np.mean(ensemble_losses),
                               "avg_acc": np.mean(ensemble_acc)}
            print("epoch", epoch)
            for i in range(self.ensemble_size):
                train_metrics_2.update({f"ensemble_{i}_loss": np.mean(ensemble_losses[i])})
                train_metrics_2.update({f"ensemble_{i}_acc": np.mean(ensemble_acc[i])})
                print({f"ensemble_{i}_loss": np.mean(ensemble_losses[i])})
                print({f"ensemble_{i}_acc": np.mean(ensemble_acc[i])})
            self.logger.log(train_metrics_2)

            if epoch % 50 == 1:
                self.save_model(self.logger._model_dir / f"reward_model_{epoch}.pt")
            # early stop
            if np.mean(ensemble_acc) > 0.968 and "antmaze" not in self.task:
                break
        return train_metrics

    def _train(self, batch, member):
        # get batch
        obs_1 = batch['observations']  # batch_size * len_query * obs_dim
        act_1 = batch['actions']  # batch_size * len_query * action_dim
        s_a_1 = np.concatenate([obs_1, act_1], axis=-1).squeeze().astype('float32')
        s_a_1 = torch.tensor(s_a_1).to(self.device)
        obs_2 = batch['observations_2']
        act_2 = batch['actions_2']
        labels = batch['labels']  # batch_size * 2 (one-hot, for equal label)

        s_a_2 = np.concatenate([obs_2, act_2], axis=-1).squeeze().astype('float32')
        s_a_2 = torch.tensor(s_a_2).to(self.device)
        # get comparable labels
        num_traj = len(labels)
        comparable_indices = np.where((labels != [0.5, 0.5]).any(axis=1))[0]
        comparable_labels = torch.from_numpy(np.argmax(labels, axis=1)).to(self.device)
        labels = torch.from_numpy(labels).to(self.device)
        # print("labels: ", labels)
        curr_loss = 0.0

        for num in range(num_traj):
            if labels[num][0] == 0:
                input_1 = s_a_2[num]
                input_2 = s_a_1[num]
            else:
                input_1 = s_a_1[num]
                input_2 = s_a_2[num]
            disc_1 = self.ensemble[member].diffusion.loss(input_1, disc_ddpm=True).unsqueeze(dim=1)
            disc_2 = self.ensemble[member].diffusion.loss(input_2, disc_ddpm=True).unsqueeze(dim=1)
            # compute loss
            loss_better = torch.nn.BCELoss()(disc_1, torch.ones(disc_1.size(), device=self.device))
            loss_worse = torch.nn.BCELoss()(disc_2, torch.zeros(disc_2.size(), device=self.device))
            traj_loss = loss_better + loss_worse
            curr_loss = curr_loss + traj_loss

        curr_loss = curr_loss / num_traj
        # compute accuracy
        r_1 = torch.zeros_like(labels)[:, 0].reshape(-1, 1)
        r_2 = torch.zeros_like(labels)[:, 0].reshape(-1, 1)
        for num in range(num_traj):
            r_hat_1 = self.disc_reward(s_a_1[num], member).detach()
            r_1[num] = (- torch.log(1 - r_hat_1)).sum()
            r_hat_2 = self.disc_reward(s_a_2[num], member).detach()
            r_2[num] = (- torch.log(1 - r_hat_2)).sum()
        r = torch.cat([r_1, r_2], axis=1)

        _, predicted = torch.max(r, 1)
        if not len(comparable_indices):
            correct = 0.7  # TODO, for exception
        else:
            correct = (predicted[comparable_indices] == comparable_labels[comparable_indices]).sum().item() / len(
                comparable_indices)

        return curr_loss, correct

    def softXEnt_loss(self, input, target):
        logprobs = nn.functional.log_softmax(input, dim=1)
        return -(target * logprobs).sum() / input.shape[0]

    # trajectory mix up

    def save_model(self, path):
        state_dicts = [model.state_dict() for model in self.ensemble]
        torch.save(state_dicts, path)

    def load_model(self, path):
        state_dicts = torch.load(path, map_location='cpu')
        for model, state_dict in zip(self.ensemble, state_dicts):
            model.load_state_dict(state_dict)
            model.to(self.device)

class CondDdpmRewardModel(RewardModel):
    def __init__(self, task, observation_dim, action_dim, n_timesteps, device, num_units,
                 disc_momentum, lr=0.0003, discrim_depth=4, lable_dim=10, ensemble_size=3, logger=None,
                 activation=None, training=True, loss_type=1):

        self.args = {"n_steps": n_timesteps, "device": device,
                     "discrim_depth": discrim_depth, "label_dim": lable_dim}
        self.num_units = num_units
        self.disc_momentum = disc_momentum
        self.training = training

        self.loss_type = loss_type
        super().__init__(task, observation_dim, action_dim, ensemble_size, lr, activation, logger, device)

    def construct_ensemble(self):
        for i in range(self.ensemble_size):
            model = utils.CondDDPM_RM(
                self.observation_dim, self.action_dim, args=self.args, num_units=self.num_units
            )

            self.ensemble.append(model.to(self.device))
            self.paramlst.extend(self.ensemble[-1].parameters())
        if self.training:
            self.opt = torch.optim.Adam(self.paramlst, lr=self.lr, betas=(self.disc_momentum, 0.999))

    def disc_reward(self, x, member):
        batch = x.to(self.device)
        # disc_cost = self.ensemble[member].diffusion.calc_reward(batch)
        s = self._compute_disc_val(x, member)
        return s

    def compute_loss(self, x):
        batch = x.to(self.device)
        total_loss = 0
        for member in range(self.ensemble_size):
            total_loss += self.ensemble[member].diffusion.compute_loss(batch)

        avg_loss = total_loss.mean()
        return avg_loss

    def train_rm(self, n_epochs, pref_dataset, data_size, batch_size, loss_type='disc', generate_model=None):
        interval = int(data_size / batch_size) + 1
        train_metrics = [
            ['epoch', 'train_loss', 'train_acc']
        ]

        for epoch in range(1, n_epochs + 1):
            ensemble_losses = [[] for _ in range(self.ensemble_size)]
            ensemble_acc = [[] for _ in range(self.ensemble_size)]
            batch_shuffled_idx = []
            len_dataset = pref_dataset["observations"].shape[0]

            for _ in range(self.ensemble_size):
                batch_shuffled_idx.append(np.random.permutation(len_dataset))

            curr_loss_list = []
            curr_acc_list = []

            for i in range(interval):
                total_loss = 0
                self.opt.zero_grad()
                start_pt = i * batch_size
                end_pt = min((i + 1) * batch_size, len_dataset)

                for member in range(self.ensemble_size):
                    batch = index_batch(pref_dataset, batch_shuffled_idx[member][start_pt:end_pt])
                    # print("member: ",member)
                    # print(batch.shape)
                    curr_loss, correct = self._train(batch, member)
                    total_loss += curr_loss
                    ensemble_losses[member].append(curr_loss.item())
                    ensemble_acc[member].append(correct)

                    curr_loss_list.append(curr_loss.item())
                    curr_acc_list.append(correct)

                total_loss.backward()
                self.opt.step()
            train_metrics.append([epoch, np.mean(curr_loss_list), np.mean(curr_acc_list)])

            train_metrics_2 = {"epoch": epoch,
                               "avg_loss": np.mean(ensemble_losses),
                               "avg_acc": np.mean(ensemble_acc)}
            print("epoch", epoch)
            for i in range(self.ensemble_size):
                train_metrics_2.update({f"ensemble_{i}_loss": np.mean(ensemble_losses[i])})
                train_metrics_2.update({f"ensemble_{i}_acc": np.mean(ensemble_acc[i])})
                print({f"ensemble_{i}_loss": np.mean(ensemble_losses[i])})
                print({f"ensemble_{i}_acc": np.mean(ensemble_acc[i])})
            self.logger.log(train_metrics_2)

            if epoch % 50 == 1:
                self.save_model(self.logger._model_dir / f"reward_model_{epoch}.pt")
            # early stop
            if np.mean(ensemble_acc) > 0.968 and "antmaze" not in self.task:
                break

        return train_metrics

    def _train(self, batch, member):
        # get batch
        obs_1 = batch['observations']  # batch_size * len_query * obs_dim
        act_1 = batch['actions']  # batch_size * len_query * action_dim
        s_a_1 = np.concatenate([obs_1, act_1], axis=-1).squeeze().astype('float32')
        s_a_1 = torch.tensor(s_a_1).to(self.device)
        obs_2 = batch['observations_2']
        act_2 = batch['actions_2']
        labels = batch['labels']  # batch_size * 2 (one-hot, for equal label)

        s_a_2 = np.concatenate([obs_2, act_2], axis=-1).squeeze().astype('float32')
        s_a_2 = torch.tensor(s_a_2).to(self.device)
        # get comparable labels
        num_traj = len(labels)
        comparable_indices = np.where((labels != [0.5, 0.5]).any(axis=1))[0]
        comparable_labels = torch.from_numpy(np.argmax(labels, axis=1)).to(self.device)
        labels = torch.from_numpy(labels).to(self.device)
        # print("labels: ", labels)
        curr_loss = 0.0
        if self.loss_type == 1:
            for num in range(num_traj):
                if labels[num][0] == 0:
                    input_1 = s_a_2[num]
                    input_2 = s_a_1[num]
                else:
                    input_1 = s_a_1[num]
                    input_2 = s_a_2[num]

                expert_d, agent_d = self._compute_discrim_loss(input_1, input_2, member)
                expert_loss = self._compute_expert_loss(expert_d)
                agent_loss = self._compute_agent_loss(agent_d)

                traj_loss = expert_loss + agent_loss
                curr_loss = curr_loss + traj_loss

            curr_loss = curr_loss / num_traj
        elif self.loss_type == 2:
            for num in range(num_traj):
                if labels[num][0] == 0:
                    input_1 = s_a_2[num]
                    input_2 = s_a_1[num]
                else:
                    input_1 = s_a_1[num]
                    input_2 = s_a_2[num]

                expert_d, agent_d = self._compute_discrim_loss(input_1, input_2, member)
                avg_expert_d = torch.mean(expert_d).unsqueeze(0)
                avg_agent_d = torch.mean(agent_d).unsqueeze(0)
                expert_loss = self._compute_expert_loss(avg_expert_d)
                agent_loss = self._compute_agent_loss(avg_agent_d)

                traj_loss = expert_loss + agent_loss
                curr_loss = curr_loss + traj_loss

            curr_loss = curr_loss / num_traj
        # compute accuracy
        r_1 = torch.zeros_like(labels)[:, 0].reshape(-1, 1)
        r_2 = torch.zeros_like(labels)[:, 0].reshape(-1, 1)
        for num in range(num_traj):
            r_hat_1 = self.disc_reward(s_a_1[num], member).detach()
            r_1[num] = (- torch.log(1 - r_hat_1)).sum()
            r_hat_2 = self.disc_reward(s_a_2[num], member).detach()
            r_2[num] = (- torch.log(1 - r_hat_2)).sum()
        r = torch.cat([r_1, r_2], axis=1)

        _, predicted = torch.max(r, 1)
        if not len(comparable_indices):
            correct = 0.7  # TODO, for exception
        else:
            correct = (predicted[comparable_indices] == comparable_labels[comparable_indices]).sum().item() / len(
                comparable_indices)

        return curr_loss, correct

    def softXEnt_loss(self, input, target):
        logprobs = nn.functional.log_softmax(input, dim=1)
        return -(target * logprobs).sum() / input.shape[0]

    # trajectory mix up

    def save_model(self, path):
        state_dicts = [model.state_dict() for model in self.ensemble]
        torch.save(state_dicts, path)

    def load_model(self, path):
        state_dicts = torch.load(path, map_location='cpu')
        for model, state_dict in zip(self.ensemble, state_dicts):
            model.load_state_dict(state_dict)
            model.to(self.device)

    def _compute_discrim_loss(self, input_1, input_2, member):
        expert_d = self._compute_disc_val(input_1, member)
        agent_d = self._compute_disc_val(input_2, member)

        return expert_d, agent_d

    def _compute_disc_val(self, input_1, member):
        label_one = self.ensemble[member](input_1, 1.)
        label_zero = self.ensemble[member](input_1, 0.)
        output = F.softmax(torch.stack([-label_one, -label_zero]), dim=0)[0]
        return output

    def _compute_expert_loss(self, expert_d):
        return F.binary_cross_entropy(expert_d,
                                      torch.ones(expert_d.shape).to(self.args["device"]))

    def _compute_agent_loss(self, agent_d):
        return F.binary_cross_entropy(agent_d,
                                      torch.zeros(agent_d.shape).to(self.args["device"]))
