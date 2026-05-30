import time

import numpy as np

import torch
from torch import nn
from DPR.diffusion.model import MLP
from DPR.diffusion.diffusion import Diffusion

def index_batch(batch, indices):
    indexed = {}
    for key in batch.keys():
        indexed[key] = batch[key][indices, ...]
    return indexed
def index_flow(dataset, indices):
    # indexed = [dataset[index] for index in indices]
    indexed = []
    for index in indices:
        indexed.append(dataset[index])
    return indexed

def index_batch_all(batch, indices):
    indexed = {}
    for key in batch.keys():
        # indexed[key] = (np.array(batch[key])[indices, ...]).tolist()
        indexed[key] = [batch[key][index] for index in indices]
    return indexed
class DDPM_RM(torch.nn.Module):

    def __init__(self, x_dim, action_dim, max_action, beta_schedule, n_timesteps, device, disc_hid_dim, disc_momentum,
                 lr=0.0003, clamp_magnitude=10.0, ftb=5, logger=None):
        super().__init__()
        self.model = MLP(x_dim=x_dim, hid_dim=disc_hid_dim, device=device)
        self.diffusion = Diffusion(x_dim, action_dim, self.model, max_action, beta_schedule=beta_schedule,
                                   n_timesteps=n_timesteps, clamp_magnitude=clamp_magnitude).to(device)

        self.diff_opti = torch.optim.Adam(self.diffusion.parameters(), lr=lr, betas=(disc_momentum, 0.999))
        self.ftb = ftb

        self.device = device
        self.logger = logger

    def disc_reward(self, x):
        batch = x.to(self.device)
        disc_cost = self.diffusion.calc_reward(batch)
        return disc_cost

    def train_rm(self, n_epochs, pref_dataset, data_size, batch_size, loss_type='disc', generate_model=None):
        interval = int(data_size / batch_size) + 1
        if generate_model is not None:
            flow_dataset = pref_dataset.get_flow(generate_model, self.ftb)
            # print(flow_dataset)
        else:
            flow_batch = None

        for epoch in range(1, n_epochs + 1):

            if generate_model is not None:
                len_dataset = len(pref_dataset.trajs)
            else:
                len_dataset = pref_dataset["observations"].shape[0]
            batch_shuffled_idx = np.random.permutation(len_dataset)

            curr_loss_list = []
            curr_acc_list = []

            for i in range(interval):
                self.diff_opti.zero_grad()
                start_pt = i * batch_size
                end_pt = min((i + 1) * batch_size, len_dataset)
                if generate_model is not None:
                    batch = pref_dataset.get_batch(batch_shuffled_idx[start_pt:end_pt])
                    # flow_batch = index_flow(flow_dataset, batch_shuffled_idx[start_pt:end_pt])
                    flow_batch = flow_dataset[batch_shuffled_idx[start_pt:end_pt]]
                else:
                    batch = index_batch(pref_dataset, batch_shuffled_idx[start_pt:end_pt])
                if loss_type =='bt':
                    curr_loss, correct = self._train_bt(batch)
                else:
                    curr_loss, correct = self._train(batch, flow_batch)
                curr_loss_list.append(curr_loss.item())
                curr_acc_list.append(correct)

                curr_loss.backward()
                self.diff_opti.step()
            train_metrics = {'epoch': epoch, 'train_loss': np.mean(curr_loss_list), 'train_acc': np.mean(curr_acc_list)}

            self.logger.log(train_metrics)
            if epoch % 50 == 1:
                self.save_model(self.logger._model_dir / f"reward_model_{epoch}.pt")
            # early stop
            if np.mean(curr_acc_list) > 0.968:
                break

        return train_metrics

    # trajectory mix up
    def _train_before(self, batch, flow_batch=None):
        # get batch
        obs_1 = batch['observations']  # batch_size * len_query * obs_dim
        act_1 = batch['actions']  # batch_size * len_query * action_dim
        obs_2 = batch['observations_2']
        act_2 = batch['actions_2']
        labels = batch['labels']  # batch_size * 2 (one-hot, for equal label)
        num_traj = len(labels)
        s_a_1 = np.concatenate([obs_1, act_1], axis=-1).squeeze().astype('float32')
        s_a_2 = np.concatenate([obs_2, act_2], axis=-1).squeeze().astype('float32')

        # get comparable labels
        comparable_indices = np.where((labels != [0.5, 0.5]).any(axis=1))[0]
        comparable_labels = torch.from_numpy(np.argmax(labels, axis=1)).to(self.device)

        # compute reward
        s_a_1 = torch.tensor(s_a_1).to(self.device)
        s_a_2 = torch.tensor(s_a_2).to(self.device)

        labels = torch.from_numpy(labels).to(self.device)

        curr_loss = 0.0

        for num in range(num_traj):
            if labels[num][0] == 0:
                input_1 = s_a_2[num]
                input_2 = s_a_1[num]
            else:
                input_1 = s_a_1[num]
                input_2 = s_a_2[num]
            disc_1 = self.diffusion.loss(input_1, disc_ddpm=True).unsqueeze(dim=1)
            disc_2 = self.diffusion.loss(input_2, disc_ddpm=True).unsqueeze(dim=1)
            # compute loss
            loss_better = torch.nn.BCELoss()(disc_1, torch.ones(disc_1.size(), device=self.device))
            loss_worse = torch.nn.BCELoss()(disc_2, torch.zeros(disc_2.size(), device=self.device))
            traj_loss = loss_better + loss_worse
            curr_loss = curr_loss + traj_loss

        curr_loss = curr_loss/num_traj
        # curr_loss = self.softXEnt_loss(r_hat, labels)
        # compute accuracy
        r_1 = torch.zeros_like(labels)[:, 0].reshape(-1, 1)
        r_2 = torch.zeros_like(labels)[:, 0].reshape(-1, 1)
        for num in range(num_traj):
            r_hat_1 = self.disc_reward(s_a_1[num]).detach()
            r_1[num] = (- torch.log(1 - r_hat_1)).sum()
            r_hat_2 = self.disc_reward(s_a_2[num]).detach()
            r_2[num] = (- torch.log(1 - r_hat_2)).sum()
        r = torch.cat([r_1, r_2], axis=1)

        _, predicted = torch.max(r, 1)
        if not len(comparable_indices):
            correct = 0.7  # TODO, for exception
        else:
            correct = (predicted[comparable_indices] == comparable_labels[comparable_indices]).sum().item() / len(
                comparable_indices)
        return curr_loss, correct

    def _train(self, batch, flow_batch=None):

        if flow_batch is not None:
            num_traj = len(flow_batch)
            obs_1, act_1 = batch
            s_a_1 = torch.cat([obs_1, act_1], dim=-1)
            # s_a_1 = flow_batch.reshape(1, -1, s_a_1.shape[-1])
            # s_a_1 = s_a_1.squeeze()
            s_a_2 = flow_batch
            # s_a_2 = flow_batch.reshape(1, -1, flow_batch.shape[-1])
            # s_a_2 = s_a_2.squeeze()
            # labels = np.array([[0., 1.] for _ in range(num_traj)])
            # comparable_indices = np.where((labels != [0.5, 0.5]).any(axis=1))[0]
            # labels = torch.from_numpy(labels).to(self.device)
            # comparable_labels = labels

            curr_loss = 0.0

            for num in range(num_traj):
                input_1 = s_a_2[num]
                input_2 = s_a_1[num]
                disc_1 = self.diffusion.loss(input_1, disc_ddpm=True).unsqueeze(dim=1)
                disc_2 = self.diffusion.loss(input_2, disc_ddpm=True).unsqueeze(dim=1)
                # compute loss
                loss_better = torch.nn.BCELoss()(disc_1, torch.ones(disc_1.size(), device=self.device))
                loss_worse = torch.nn.BCELoss()(disc_2, torch.zeros(disc_2.size(), device=self.device))
                traj_loss = loss_better + loss_worse
                curr_loss = curr_loss + traj_loss

            curr_loss = curr_loss / num_traj

            correct = 0.0

            # disc_1 = self.diffusion.loss(s_a_2, disc_ddpm=True).unsqueeze(dim=1)
            # disc_2 = self.diffusion.loss(s_a_1, disc_ddpm=True).unsqueeze(dim=1)
            # # compute loss
            # loss_better = torch.nn.BCELoss()(disc_1, torch.ones(disc_1.size(), device=self.device))
            # loss_worse = torch.nn.BCELoss()(disc_2, torch.zeros(disc_2.size(), device=self.device))
            # curr_loss = loss_better + loss_worse
            #
            # r_hat_1 = self.disc_reward(s_a_1).detach()
            # r_1 = (- torch.log(1 - r_hat_1)).sum()
            # r_hat_2 = self.disc_reward(s_a_2).detach()
            # r_2 = (- torch.log(1 - r_hat_2)).sum()
            # r = torch.cat([r_1, r_2], axis=1)

        else:
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
            print("labels: ", labels)
            curr_loss = 0.0

            for num in range(num_traj):
                if labels[num][0] == 0:
                    input_1 = s_a_2[num]
                    input_2 = s_a_1[num]
                else:
                    input_1 = s_a_1[num]
                    input_2 = s_a_2[num]
                disc_1 = self.diffusion.loss(input_1, disc_ddpm=True).unsqueeze(dim=1)
                disc_2 = self.diffusion.loss(input_2, disc_ddpm=True).unsqueeze(dim=1)
                # compute loss
                loss_better = torch.nn.BCELoss()(disc_1, torch.ones(disc_1.size(), device=self.device))
                loss_worse = torch.nn.BCELoss()(disc_2, torch.zeros(disc_2.size(), device=self.device))
                traj_loss = loss_better + loss_worse
                curr_loss = curr_loss + traj_loss

            curr_loss = curr_loss / num_traj
            # curr_loss = self.softXEnt_loss(r_hat, labels)
            # compute accuracy
            r_1 = torch.zeros_like(labels)[:, 0].reshape(-1, 1)
            r_2 = torch.zeros_like(labels)[:, 0].reshape(-1, 1)
            for num in range(num_traj):
                r_hat_1 = self.disc_reward(s_a_1[num]).detach()
                r_1[num] = (- torch.log(1 - r_hat_1)).sum()
                r_hat_2 = self.disc_reward(s_a_2[num]).detach()
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
    def _train_bt(self, batch):
        # get batch
        obs_1 = batch['observations']  # batch_size * len_query * obs_dim
        act_1 = batch['actions']  # batch_size * len_query * action_dim
        obs_2 = batch['observations_2']
        act_2 = batch['actions_2']
        labels = batch['labels']  # batch_size * 2 (one-hot, for equal label)
        num_traj = len(labels)
        s_a_1 = np.concatenate([obs_1, act_1], axis=-1).squeeze().astype('float32')
        s_a_2 = np.concatenate([obs_2, act_2], axis=-1).squeeze().astype('float32')

        # get comparable labels
        comparable_indices = np.where((labels != [0.5, 0.5]).any(axis=1))[0]
        comparable_labels = torch.from_numpy(np.argmax(labels, axis=1)).to(self.device)

        # compute reward
        s_a_1 = torch.tensor(s_a_1).to(self.device)
        s_a_2 = torch.tensor(s_a_2).to(self.device)

        labels = torch.from_numpy(labels).to(self.device)

        curr_loss = 0.0

        for num in range(num_traj):
            disc_1 = self.diffusion.loss(s_a_1[num], disc_ddpm=True).unsqueeze(dim=1)
            disc_2 = self.diffusion.loss(s_a_2[num], disc_ddpm=True).unsqueeze(dim=1)
            # compute loss
            r_hat = torch.cat([disc_1.sum().reshape(1, -1), disc_2.sum().reshape(1, -1)], axis=1)
            traj_loss = self.softXEnt_loss(r_hat, labels[num])
            curr_loss = curr_loss + traj_loss

        curr_loss = curr_loss/num_traj
        # curr_loss = self.softXEnt_loss(r_hat, labels)
        # compute accuracy
        r_1 = torch.zeros_like(labels)[:, 0].reshape(-1, 1)
        r_2 = torch.zeros_like(labels)[:, 0].reshape(-1, 1)
        for num in range(num_traj):
            r_hat_1 = self.disc_reward(s_a_1[num]).detach()
            r_1[num] = (- torch.log(1 - r_hat_1)).sum()
            r_hat_2 = self.disc_reward(s_a_2[num]).detach()
            r_2[num] = (- torch.log(1 - r_hat_2)).sum()

        r = torch.cat([r_1, r_2], axis=1)

        _, predicted = torch.max(r, 1)
        if not len(comparable_indices):
            correct = 0.7  # TODO, for exception
        else:
            correct = (predicted[comparable_indices] == comparable_labels[comparable_indices]).sum().item() / len(
                comparable_indices)
        return curr_loss, correct

    def save_model(self, path):
        state_dicts = self.diffusion.state_dict()
        torch.save(state_dicts, path)

    def load_model(self, path):
        state_dicts = torch.load(path, map_location='cpu')
        self.diffusion.load_state_dict(state_dicts)
        self.diffusion.to(self.device)
