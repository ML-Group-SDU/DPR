import collections
import time
import os
import numpy as np
import gym
from tqdm import trange
import torch
import torch.nn as nn
import math
from DPR.diffusion.diffusion import Diffusion
from DPR.diffusion.model import MLP as MLP2
from CDPR.ddpm import MLPConditionDiffusion
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union




Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])


def to_torch(x, dtype=torch.float32):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype)


class Dataset(object):
    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray,
                 size: int):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx])


class D4RLDataset(Dataset):
    def __init__(self,
                 env: gym.Env,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5):
        dataset = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        dones_float = np.zeros_like(dataset['rewards'])

        for i in range(len(dones_float) - 1):
            if np.linalg.norm(dataset['observations'][i + 1] -
                              dataset['next_observations'][i]
                              ) > 1e-5 or dataset['terminals'][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        super().__init__(dataset['observations'].astype(np.float32),
                         actions=dataset['actions'].astype(np.float32),
                         rewards=dataset['rewards'].astype(np.float32),
                         masks=1.0 - dataset['terminals'].astype(np.float32),
                         dones_float=dones_float.astype(np.float32),
                         next_observations=dataset['next_observations'].astype(
                             np.float32),
                         size=len(dataset['observations']))


@torch.no_grad()
def reward_from_preference(
        dataset: D4RLDataset,
        reward_model,
        batch_size: int = 256,
        reward_model_type: str = "transformer",
        device="cuda",
        ensemble_size=3,
):
    data_size = dataset["rewards"].shape[0]
    interval = int(data_size / batch_size) + 1
    new_r = np.zeros_like(dataset["rewards"])
    new_rs = np.zeros((len(reward_model.ensemble), dataset["rewards"].shape[0]), dtype=float)
    if "transformer" in reward_model_type:
        with torch.no_grad():
            max_seq_len = reward_model.max_seq_len
            for each in reward_model.ensemble:
                each.eval()

            obs, act = [], []
            ptr = 0
            for i in trange(data_size):

                if len(obs) < max_seq_len:
                    obs.append(dataset["observations"][i])
                    act.append(dataset["actions"][i])

                if dataset["terminals"][i] > 0 or i == data_size - 1 or len(obs) == max_seq_len:
                    tensor_obs = to_torch(np.array(obs)[None,], dtype=torch.float32).to(device)
                    tensor_act = to_torch(np.array(act)[None,], dtype=torch.float32).to(device)

                    new_reward = 0
                    for each in reward_model.ensemble:
                        new_reward += each(tensor_obs, tensor_act).detach().cpu().numpy()
                    new_reward /= len(reward_model.ensemble)
                    if tensor_obs.shape[1] <= -1:
                        new_r[ptr:ptr + tensor_obs.shape[1]] = dataset["rewards"][ptr:ptr + tensor_obs.shape[1]]
                    else:
                        new_r[ptr:ptr + tensor_obs.shape[1]] = new_reward
                    ptr += tensor_obs.shape[1]
                    obs, act = [], []

    elif reward_model_type == "DPR":
        with torch.no_grad():
            for i in trange(interval):
                start_pt = i * batch_size
                end_pt = (i + 1) * batch_size

                observations = dataset["observations"][start_pt:end_pt]
                actions = dataset["actions"][start_pt:end_pt]
                obs_act = np.concatenate([observations, actions], axis=-1)

                inputs = torch.tensor(obs_act).to(device)
                new_reward = 0
                for each in reward_model.ensemble:
                    r = each.disc_reward(inputs.reshape(-1, inputs.size(1)))

                    new_reward += (- torch.log(1 - r)).unsqueeze(dim=1).reshape(-1).cpu().numpy()
                new_r[start_pt:end_pt] = new_reward
    elif reward_model_type == "C-DPR":
        with torch.no_grad():
            for i in trange(interval):
                start_pt = i * batch_size
                end_pt = (i + 1) * batch_size

                observations = dataset["observations"][start_pt:end_pt]
                actions = dataset["actions"][start_pt:end_pt]
                obs_act = np.concatenate([observations, actions], axis=-1)

                inputs = torch.tensor(obs_act).to(device)
                new_reward = 0
                eps = 1e-20
                for i in range(ensemble_size):
                    if inputs.reshape(-1, inputs.size(1)).shape[0] % 2 == 1:
                        inputs_ = inputs.reshape(-1, inputs.size(1))
                        inputs_ = torch.cat((inputs_, inputs_[-1:, :]), dim=0)
                        r = reward_model.disc_reward(inputs_, 1)[:-1, ]
                        new_reward += (- torch.log(1 - r)).unsqueeze(dim=1).reshape(-1).cpu().numpy()
                    else:

                        r = reward_model.disc_reward(inputs.reshape(-1, inputs.size(1)), i)
                        # new_reward += (- torch.log(1 - r)).unsqueeze(dim=1).reshape(-1).cpu().numpy()
                        new_reward += (- torch.log(1 - r)).unsqueeze(dim=1).reshape(-1).cpu().numpy()
                new_r[start_pt:end_pt] = new_reward

    else:
        with torch.no_grad():
            for i in trange(interval):
                start_pt = i * batch_size
                end_pt = (i + 1) * batch_size

                observations = dataset["observations"][start_pt:end_pt]
                actions = dataset["actions"][start_pt:end_pt]
                obs_act = np.concatenate([observations, actions], axis=-1)

                new_reward = reward_model.get_reward_batch(obs_act).reshape(-1)
                new_r[start_pt:end_pt] = new_reward

    dataset["rewards"] = new_r.copy()
    # dataset["rewards_ensembles"] = new_rs
    return dataset


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class PrefTransformer1(nn.Module):
    ''' Transformer Structure used in Preference Transformer.

    Description:
        This structure holds a causal transformer, which takes in a sequence of observations and actions,
        and outputs a sequence of latent vectors. Then, pass the latent vectors through self-attention to
        get a weight vector, which is used to weight the latent vectors to get the final preference score.

    Args:
        - observation_dim: dimension of observation
        - action_dim: dimension of action
        - max_seq_len: maximum length of sequence
        - d_model: dimension of transformer
        - nhead: number of heads in transformer
        - num_layers: number of layers in transformer
    '''

    def __init__(self,
                 observation_dim: int, action_dim: int,
                 max_seq_len: int = 100,
                 d_model: int = 256, nhead: int = 4, num_layers: int = 1,
                 ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.pos_emb = SinusoidalPosEmb(d_model)

        self.obs_emb = nn.Sequential(
            nn.Linear(observation_dim, d_model),
            nn.LayerNorm(d_model)
        )
        self.act_emb = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.LayerNorm(d_model)
        )

        self.causual_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, batch_first=True),
            num_layers
        )
        self.mask = nn.Transformer.generate_square_subsequent_mask(2 * self.max_seq_len)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.r_proj = nn.Linear(d_model, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor, repre=False):
        if self.mask.device != obs.device: self.mask = self.mask.to(obs.device)
        batch_size, traj_len = obs.shape[:2]

        pos = self.pos_emb(
            torch.arange(traj_len, device=obs.device))[None,]
        obs = self.obs_emb(obs) + pos
        act = self.act_emb(act) + pos

        x = torch.empty((batch_size, 2 * traj_len, self.d_model), device=obs.device)
        x[:, 0::2] = obs
        x[:, 1::2] = act

        x = self.causual_transformer(x, self.mask[:2 * traj_len, :2 * traj_len])[:, 1::2]
        # x: (batch_size, traj_len, d_model)
        if repre:
            return x
        q = self.q_proj(x)  # (batch_size, traj_len, d_model)
        k = self.k_proj(x)  # (batch_size, traj_len, d_model)
        r = self.r_proj(x)  # (batch_size, traj_len, 1)

        w = torch.softmax(q @ k.permute(0, 2, 1) / np.sqrt(self.d_model), -1).mean(-2)
        # w: (batch_size, traj_len)

        z = (w * r.squeeze(-1))  # (batch_size, traj_len)

        return torch.tanh(z)


class PrefTransformer2(nn.Module):
    ''' Preference Transformer with no causal mask and no self-attention but one transformer layer to get the weight vector.

    Description:
        This structure has no causal mask and no self-attention.
        Instead, it uses one transformer layer to get the weight vector.

    Args:
        - observation_dim: dimension of observation
        - action_dim: dimension of action
        - d_model: dimension of transformer
        - nhead: number of heads in transformer
        - num_layers: number of layers in transformer
    '''

    def __init__(self,
                 observation_dim: int, action_dim: int,
                 d_model: int, nhead: int, num_layers: int,
                 ):
        super().__init__()
        while num_layers < 2: num_layers += 1
        self.d_model = d_model
        self.pos_emb = SinusoidalPosEmb(d_model)
        self.obs_emb = nn.Sequential(
            nn.Linear(observation_dim, d_model),
            nn.LayerNorm(d_model)
        )
        self.act_emb = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.LayerNorm(d_model)
        )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, batch_first=True),
            num_layers - 1
        )
        self.value_layer = nn.Sequential(nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, batch_first=True), 1
        ), nn.Linear(d_model, 1))
        self.weight_layer = nn.Sequential(nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, batch_first=True), 1
        ), nn.Linear(d_model, 1))

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        batch_size, traj_len = obs.shape[:2]

        pos = self.pos_emb(
            torch.arange(traj_len, device=obs.device))[None,]
        obs = self.obs_emb(obs) + pos
        act = self.act_emb(act) + pos

        x = torch.empty((batch_size, 2 * traj_len, self.d_model), device=obs.device)
        x[:, 0::2] = obs
        x[:, 1::2] = act

        x = self.transformer(x)[:, 1::2]
        v = self.value_layer(x)
        w = torch.softmax(self.weight_layer(x), 1)
        return (w * v).squeeze(-1)


class PrefTransformer3(nn.Module):
    ''' Preference Transformer with no causal mask and no weight vector.

    Description:
        This structure has no causal mask and even no weight vector.
        Instead, it directly outputs the preference score.

    Args:
        - observation_dim: dimension of observation
        - action_dim: dimension of action
        - d_model: dimension of transformer
        - nhead: number of heads in transformer
        - num_layers: number of layers in transformer
    '''

    def __init__(self,
                 observation_dim: int, action_dim: int,
                 d_model: int, nhead: int, num_layers: int,
                 ):
        super().__init__()

        self.d_model = d_model
        self.pos_emb = SinusoidalPosEmb(d_model)
        self.obs_emb = nn.Sequential(
            nn.Linear(observation_dim, d_model),
            nn.LayerNorm(d_model)
        )
        self.act_emb = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.LayerNorm(d_model)
        )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, batch_first=True),
            num_layers
        )
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        batch_size, traj_len = obs.shape[:2]

        pos = self.pos_emb(
            torch.arange(traj_len, device=obs.device))[None,]
        obs = self.obs_emb(obs) + pos
        act = self.act_emb(act) + pos

        x = torch.empty((batch_size, 2 * traj_len, self.d_model), device=obs.device)
        x[:, 0::2] = obs
        x[:, 1::2] = act

        x = self.transformer(x)[:, 1::2]
        return self.output_layer(x).squeeze(-1)


class PrefTransformer4(nn.Module):
    ''' Transformer Structure used in Preference Transformer with bidirectional attebtion.

    Description:
        This structure holds a causal transformer, which takes in a sequence of observations and actions,
        and outputs a sequence of latent vectors. Then, pass the latent vectors through self-attention to
        get a weight vector, which is used to weight the latent vectors to get the final preference score.

    Args:
        - observation_dim: dimension of observation
        - action_dim: dimension of action
        - max_seq_len: maximum length of sequence
        - d_model: dimension of transformer
        - nhead: number of heads in transformer
        - num_layers: number of layers in transformer
    '''

    def __init__(self,
                 observation_dim: int, action_dim: int,
                 max_seq_len: int = 100,
                 d_model: int = 256, nhead: int = 4, num_layers: int = 1,
                 ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.pos_emb = SinusoidalPosEmb(d_model)

        self.obs_emb = nn.Sequential(
            nn.Linear(observation_dim, d_model),
            nn.LayerNorm(d_model)
        )
        self.act_emb = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.LayerNorm(d_model)
        )

        self.full_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, batch_first=True),
            num_layers
        )
        self.mask = None

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.r_proj = nn.Linear(d_model, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor, repre=False, weight=False):
        # if self.mask.device != obs.device: self.mask = self.mask.to(obs.device)
        batch_size, traj_len = obs.shape[:2]

        pos = self.pos_emb(
            torch.arange(traj_len, device=obs.device))[None,]
        obs = self.obs_emb(obs) + pos
        act = self.act_emb(act) + pos

        x = torch.empty((batch_size, 2 * traj_len, self.d_model), device=obs.device)
        x[:, 0::2] = obs
        x[:, 1::2] = act

        x = self.full_transformer(x, self.mask)[:, 1::2]
        # x: (batch_size, traj_len, d_model)
        if repre:
            return x
        q = self.q_proj(x)  # (batch_size, traj_len, d_model)
        k = self.k_proj(x)  # (batch_size, traj_len, d_model)
        r = self.r_proj(x)  # (batch_size, traj_len, 1)

        w = torch.softmax(q @ k.permute(0, 2, 1) / np.sqrt(self.d_model), -1).mean(-2)
        # w: (batch_size, traj_len)
        if weight:
            return w
        z = (w * r.squeeze(-1))  # (batch_size, traj_len)

        return torch.tanh(z)


class PrefTransformer5(nn.Module):
    ''' Transformer Structure used in Preference Transformer with bidirectional attebtion.

    Description:
        This structure holds a causal transformer, which takes in a sequence of observations and actions,
        and outputs a sequence of latent vectors. Then, pass the latent vectors through self-attention to
        get a weight vector, which is used to weight the latent vectors to get the final preference score.

    Args:
        - observation_dim: dimension of observation
        - action_dim: dimension of action
        - max_seq_len: maximum length of sequence
        - d_model: dimension of transformer
        - nhead: number of heads in transformer
        - num_layers: number of layers in transformer
    '''

    def __init__(self,
                 observation_dim: int, action_dim: int,
                 max_seq_len: int = 100,
                 d_model: int = 256, nhead: int = 4, num_layers: int = 1,
                 ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.pos_emb = SinusoidalPosEmb(d_model)

        self.obs_emb = nn.Sequential(
            nn.Linear(observation_dim, d_model),
            nn.LayerNorm(d_model)
        )
        self.act_emb = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.LayerNorm(d_model)
        )

        self.full_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, batch_first=True),
            num_layers
        )

        self.mask = self.generate_anti_causal_mask(2 * self.max_seq_len)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.r_proj = nn.Linear(d_model, 1)

    def generate_anti_causal_mask(self, seq_len):
        """
        Generates an anti-causal attention mask for a given sequence length.
        Each position can only attend to future positions, excluding itself.
        """
        # Create an upper triangular mask
        mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=-1)  # Lower triangle
        mask = mask.masked_fill(mask == 1, float('-inf'))  # Convert to -inf
        mask = mask.masked_fill(mask == 0, 0)  # Convert others to 0
        return mask

    def forward(self, obs: torch.Tensor, act: torch.Tensor, repre=False):
        # if obs.shape[0] !=self.max_seq_len:
        #     self.mask = self.generate_anti_causal_mask(2 * obs.shape[1]).to(obs.device)

        if self.mask.device != obs.device: self.mask = self.mask.to(obs.device)
        batch_size, traj_len = obs.shape[:2]

        pos = self.pos_emb(
            torch.arange(traj_len, device=obs.device))[None,]
        obs = self.obs_emb(obs) + pos
        act = self.act_emb(act) + pos

        x = torch.empty((batch_size, 2 * traj_len, self.d_model), device=obs.device)
        x[:, 0::2] = obs
        x[:, 1::2] = act

        x = self.full_transformer(x, self.mask)[:, 1::2]
        # self.mask=mask_cp
        if repre:
            return x

        # x: (batch_size, traj_len, d_model)

        q = self.q_proj(x)  # (batch_size, traj_len, d_model)
        k = self.k_proj(x)  # (batch_size, traj_len, d_model)
        r = self.r_proj(x)  # (batch_size, traj_len, 1)

        w = torch.softmax(q @ k.permute(0, 2, 1) / np.sqrt(self.d_model), -1).mean(-2)
        # w: (batch_size, traj_len)

        z = (w * r.squeeze(-1))  # (batch_size, traj_len)

        return torch.tanh(z)


class DDPM_RM(torch.nn.Module):
    def __init__(self, x_dim, action_dim, max_action, beta_schedule, n_timesteps, device, disc_hid_dim, disc_momentum,
                 lr=0.0003, clamp_magnitude=10.0, ftb=5):
        super().__init__()
        self.model = MLP2(x_dim=x_dim, hid_dim=disc_hid_dim, device=device)
        self.diffusion = Diffusion(x_dim, action_dim, self.model, max_action, beta_schedule=beta_schedule,
                                   n_timesteps=n_timesteps, clamp_magnitude=clamp_magnitude).to(device)

        self.diff_opti = torch.optim.Adam(self.diffusion.parameters(), lr=lr, betas=(disc_momentum, 0.999))
        self.ftb = ftb

        self.device = device

    def disc_reward(self, x):
        batch = x.to(self.device)
        disc_cost = self.diffusion.calc_reward(batch)
        return disc_cost


class CondDDPM_RM(torch.nn.Module):
    # def __init__(self, x_dim, action_dim, max_action, beta_schedule, n_timesteps, device, disc_hid_dim, disc_momentum,
    #              lr=0.0003, clamp_magnitude=10.0, ftb=5):
    #     super().__init__()
    #     self.model = MLP(x_dim=x_dim, hid_dim=disc_hid_dim, device=device)
    #     self.diffusion = Diffusion(x_dim, action_dim, self.model, max_action, beta_schedule=beta_schedule,
    #                                n_timesteps=n_timesteps, clamp_magnitude=clamp_magnitude).to(device)
    #
    #     self.diff_opti = torch.optim.Adam(self.diffusion.parameters(), lr=lr, betas=(disc_momentum, 0.999))
    #     self.ftb = ftb
    #
    #     self.device = device

    def __init__(self, state_dim, action_dim, args, re_c=False, num_units=128):
        super(CondDDPM_RM, self).__init__()
        input_dim = state_dim + action_dim
        self.args = args
        self.n_steps = self.args["n_steps"]
        betas = self.cosine_beta_schedule(self.n_steps)
        self.betas = betas.to(self.args["device"])
        alphas = 1 - betas
        alphas_prod = torch.cumprod(alphas, 0)
        self.alphas_bar_sqrt = torch.sqrt(alphas_prod).to(self.args["device"])
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod).to(self.args["device"])

        d_model = MLPConditionDiffusion(self.n_steps, self.args["label_dim"], input_dim, num_units=num_units,
                                        depth=self.args["discrim_depth"]).to(self.args["device"])
        self.re_c = re_c
        self.model = d_model

    def cosine_beta_schedule(self, timesteps, s=0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def diffusion_loss(self, label, sa_pair, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
        batch_size = sa_pair.shape[0]

        # if self.args["sample_strategy"] == "constant":
        #     step = self.args["sample_strategy_value"]
        #     if step >= n_steps:
        #         step = n_steps - 1
        #     t = torch.full((batch_size,), step, device=self.args["device"])
        #     t = t.unsqueeze(-1)
        # else:
        t = torch.randint(0, n_steps, size=(batch_size // 2,)).to(self.args["device"])
        t = torch.cat([t, n_steps - 1 - t], dim=0)  # [batch_size, 1]
        t = t.unsqueeze(-1)

        # coefficient of x0
        a = alphas_bar_sqrt[t]

        # coefficient of eps
        aml = one_minus_alphas_bar_sqrt[t]
        if not self.re_c:
            label_input = torch.full((batch_size, self.args["label_dim"]), label).to(self.args["device"])
        else:
            label_input = label.to(self.args["device"])

        # generate random noise eps
        e = torch.randn_like(sa_pair).to(self.args["device"])

        # model input
        x = sa_pair * a + e * aml

        # get predicted randome noise at time t
        output = self.model(x, label_input, t.squeeze(-1))

        return (e - output).square().mean(dim=1, keepdim=True)
        # return torch.unsqueeze(torch.mean(e - output, dim=1), 1)

    def diffusion_loss_fn(self, label, sa_pair):
        diff_loss = self.diffusion_loss(label, sa_pair, self.alphas_bar_sqrt, self.one_minus_alphas_bar_sqrt,
                                        self.n_steps)

        return diff_loss

    def forward(self, state_action, label):
        # state_action = torch.cat([state, action], dim=1)
        loss = self.diffusion_loss_fn(label, state_action)
        return loss

    def p_sample_loop(self, state, action):
        cond = torch.cat([state, action], dim=1).to(self.args["device"])
        batch_size = cond.shape[0]
        cur_x = torch.randn(batch_size, self.args["label_dim"]).to(self.args["device"])
        x_seq = [cur_x]
        for i in reversed(range(self.n_steps)):
            cur_x = self.p_sample(cur_x, cond, i, self.betas, self.one_minus_alphas_bar_sqrt)
            x_seq.append(cur_x)
        return x_seq

    def p_sample(self, x, c, t, betas, one_minus_alphas_bar_sqrt):
        # sample reconstruction data at time t drom x[T]
        t = torch.tensor([t]).to(self.args["device"])

        coeff = betas[t] / one_minus_alphas_bar_sqrt[t]

        eps_theta = self.model(x, c, t)

        mean = (1 / (1 - betas[t]).sqrt()) * (x - (coeff * eps_theta))

        z = torch.randn_like(x)
        sigma_t = betas[t].sqrt()

        sample = mean + sigma_t * z

        return (sample)


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()
        # Keep track of evaluation time so that total time only includes train time
        self._eval_start_time = 0
        self._eval_time = 0
        self._eval_flag = False

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time - self._eval_time
        return elapsed_time, total_time

    def eval(self):
        if not self._eval_flag:
            self._eval_flag = True
            self._eval_start_time = time.time()
        else:
            self._eval_time += time.time() - self._eval_start_time
            self._eval_flag = False
            self._eval_start_time = 0

    def total_time(self):
        return time.time() - self._start_time - self._eval_time


class D4RLOfflineDataset(torch.utils.data.IterableDataset):
    def __init__(
            self,
            observation_space: gym.Space,
            action_space: gym.Space,
            env: str,
            segment_length: Optional[int] = None,
            batch_size: Optional[int] = None,
            capacity: Optional[int] = None,
            mode: str = "transition",
            padding_mode: str = "right",
            reward_scale: Optional[float] = None,
            reward_shift: Optional[float] = None,
            reward_normalize: bool = False,
    ):
        super().__init__()
        assert mode in {"transition", "trajectory"}, "Supported mode for D4RLOfflineDataset: {transition, trajectory}."
        assert padding_mode in {"left", "right", "none",
                                "Supported padding mode for D4RLOfflineDataset: {left, right, none}."}
        assert reward_scale is None == reward_shift is None, "reward_scale and reward_shift should be set simultaneously."
        assert not reward_normalize or reward_shift is None, "reward scale & shift and reward normalize can not be set simultaneously."

        self.env_name = env
        self.mode = mode
        self.padding_mode = padding_mode
        self.batch_size = 1 if batch_size is None else batch_size
        self.segment_length = segment_length
        self.capacity = capacity

        self.reward_scale = reward_scale
        self.reward_shift = reward_shift
        self.reward_normalize = reward_normalize

        self.load_dataset()

    def __len__(self):
        return self.data_size

    def __iter__(self):
        while True:
            if self.mode == "transition" or (self.mode == "trajectory" and self.sample_full):
                idxs = np.random.randint(0, self.data_size, size=self.batch_size)
                idxs = np.squeeze(idxs)
                yield {
                    "obs": self.data["obs"][idxs],
                    "action": self.data["action"][idxs],
                    "next_obs": self.data["next_obs"][idxs],
                    "reward": self.data["reward"][idxs],
                    "terminal": self.data["terminal"][idxs],
                    "mask": self.data["mask"][idxs]
                }
            else:
                sample = []
                for _ in range(self.batch_size):
                    if self.padding_mode == "right":
                        traj_idx = np.random.choice(self.data_size, p=self.sample_prob)
                        start_idx = np.random.choice(self.traj_len[traj_idx])
                        s = {
                            k: self.pad_along_axis(
                                v[traj_idx, start_idx:min(start_idx + self.segment_length, self.traj_len[traj_idx])],
                                pad_to=self.segment_length) for k, v in self.data.items()
                        }
                        s["timestep"] = np.arange(start_idx, start_idx + self.segment_length)
                    elif self.padding_mode == "left":
                        traj_idx = np.random.choice(self.data_size, p=self.sample_prob)
                        end_idx = np.random.choice(self.traj_len[traj_idx]) + 1
                        s = {
                            k: self.pad_along_axis(v[traj_idx, max(0, end_idx - self.segment_length):end_idx],
                                                   pad_to=self.segment_length, direction="left") for k, v in
                            self.data.items()
                        }
                        s["timestep"] = np.maximum(np.arange(self.segment_length) + 1 - self.segment_length + end_idx,
                                                   0)
                    elif self.padding_mode == "none":
                        traj_idx = np.random.choice(self.data_size, p=self.sample_prob)
                        while self.traj_len[traj_idx] < self.segment_length:
                            traj_idx = np.random.choice(self.data_size, p=self.sample_prob)
                        start_idx = np.random.choice(self.traj_len[traj_idx] - self.segment_length + 1)
                        s = {
                            k: v[traj_idx, start_idx:start_idx + self.segment_length] for k, v in self.data.items()
                        }
                        s["timestep"] = np.arange(start_idx, start_idx + self.segment_length)
                    sample.append(s)
                if len(sample) == 1:
                    yield sample[0]
                else:
                    yield {
                        "obs": np.stack([s["obs"] for s in sample], axis=0),
                        "action": np.stack([s["action"] for s in sample], axis=0),
                        "next_obs": np.stack([s["next_obs"] for s in sample], axis=0),
                        "reward": np.stack([s["reward"] for s in sample], axis=0),
                        "terminal": np.stack([s["terminal"] for s in sample], axis=0),
                        "return": np.stack([s["return"] for s in sample], axis=0),
                        "mask": np.stack([s["mask"] for s in sample], axis=0),
                        "timestep": np.stack([s["timestep"] for s in sample], axis=0),
                    }

    def pad_along_axis(self,
                       arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0, direction="right"
                       ) -> np.ndarray:
        pad_size = pad_to - arr.shape[axis]
        if pad_size <= 0:
            return arr

        npad = [(0, 0)] * arr.ndim
        if direction == "right":
            npad[axis] = (0, pad_size)
        else:
            npad[axis] = (pad_size, 0)
        return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)

    def load_dataset(self):
        env = gym.make(self.env_name)
        dataset = env.get_dataset()

        N = dataset["rewards"].shape[0]
        obs_ = []
        next_obs_ = []
        action_ = []
        reward_ = []
        terminal_ = []
        end_ = []
        ep_reward_ = []

        use_timeouts = "timeouts" in dataset
        episode_step = 0
        episode_reward = 0
        for i in range(N - 1):
            obs = dataset["observations"][i].astype(np.float32)
            next_obs = dataset["observations"][i + 1].astype(np.float32)
            action = dataset["actions"][i].astype(np.float32)
            reward = dataset["rewards"][i].astype(np.float32)
            terminal = bool(dataset["terminals"][i])
            end = False
            episode_step += 1
            episode_reward += reward
            if use_timeouts:
                final_timestep = dataset["timeouts"][i]
            else:
                final_timestep = (episode_step == env._max_episode_steps)
            if final_timestep:
                if not terminal:
                    end_[-1] = True
                    ep_reward_.append(episode_reward - reward)
                    episode_step = 0
                    episode_reward = 0
                    continue
            if final_timestep or terminal:
                end = True
                ep_reward_.append(episode_reward)
                episode_step = 0
                episode_reward = 0
            obs_.append(obs)
            next_obs_.append(next_obs)
            action_.append(action)
            reward_.append(reward)
            terminal_.append(terminal)
            end_.append(end)
        end_[-1] = True

        data = {
            "obs": np.asarray(obs_),
            "action": np.asarray(action_),
            "next_obs": np.asarray(next_obs_),
            "reward": np.asarray(reward_)[..., None],
            "terminal": np.asarray(terminal_)[..., None],
            "mask": np.ones([len(obs_), 1], dtype=np.float32),
            "end": np.asarray(end_)[..., None],
        }

        if self.reward_normalize:
            min_, max_ = min(ep_reward_), max(ep_reward_)
            data["reward"] = data["reward"] * env._max_episode_steps / (max_ - min_)
        if self.reward_shift:
            data["reward"] = data["reward"] * self.reward_scale + self.reward_shift

        if self.mode == "transition":
            self.data_size = len(obs_)
            if self.capacity is not None:
                if self.capacity > self.data_size:
                    print(f"[Warning]: capacity {self.capacity} exceeds dataset size {self.data_size}")
                self.data_size = min(self.data_size, self.capacity)
                data = {
                    k: v[:self.data_size] for k, v in data.items()
                }
            self.data = data
        elif self.mode == "trajectory":
            traj, traj_len = [], []
            traj_start = 0
            for i in range(len(reward_)):
                if end_[i]:
                    episode_data = {
                        k: v[traj_start:i + 1] for k, v in data.items()
                    }
                    episode_data["return"] = self.discounted_cum_sum(episode_data["reward"], 1.0)
                    traj.append(episode_data)
                    traj_len.append(i + 1 - traj_start)
                    traj_start = i + 1
            self.traj_len = np.asarray(traj_len)
            self.data_size = len(self.traj_len)
            if self.capacity is not None:
                if self.capacity > self.data_size:
                    print(f"[Warning]: capacity {self.capacity} exceeds dataset size {self.data_size}")
                self.data_size = min(self.data_size, self.capacity)
                traj = traj[:self.data_size]
                self.traj_len = self.traj_len[:self.data_size]

            if self.segment_length is None:
                self.max_len = self.traj_len.max()
                self.segment_length = self.max_len
                self.sample_full = True
            else:
                self.max_len = self.traj_len.max()
                self.sample_prob = self.traj_len / self.traj_len.sum()
                self.sample_full = False

            for i_traj in range(self.data_size):
                for _key, _value in traj[i_traj].items():
                    traj[i_traj][_key] = self.pad_along_axis(_value, pad_to=self.max_len)
            self.data = {
                "obs": np.asarray([t["obs"] for t in traj]),
                "action": np.asarray([t["action"] for t in traj]),
                "next_obs": np.asarray([t["next_obs"] for t in traj]),
                "reward": np.asarray([t["reward"] for t in traj]),
                "terminal": np.asarray([t["terminal"] for t in traj]),
                "return": np.asarray([t["return"] for t in traj]),
                "mask": np.asarray([t["mask"] for t in traj]),
            }
        del env

    @torch.no_grad()
    def relabel_reward(self, agent):
        assert hasattr(agent, "select_reward"), f"Agent {agent} must support relabel_reward!"
        bs = 64
        for i_batch in range((self.data_size - 1) // bs + 1):
            idx = np.arange(i_batch * bs, min((i_batch + 1) * bs, self.data_size))
            batch = {
                "obs": self.data["obs"][idx],
                "action": self.data["action"][idx],
                "next_obs": self.data["next_obs"][idx],
                "mask": self.data["mask"][idx]
            }
            batch = agent.format_batch(batch)
            reward = agent.select_reward(batch).detach().cpu().numpy()
            reward = reward * self.data["mask"][idx]
            self.data["reward"][idx] = reward

        if self.mode == "trajectory":
            # CHECK: may be bug. the max and min returns are not consistent with those computed in transition mode
            return_ = self.data["reward"].copy()
            for t in reversed(range(return_.shape[1] - 1)):
                return_[:, t] += return_[:, t + 1]
            self.data["return"] = return_
            # normalization
            prev_return_min, prev_return_max = return_[:, 0].min(), return_[:, 0].max()
            max_return = max(abs(return_[:, 0].max()), abs(return_[:, 0].min()),
                             return_[:, 0].max() - return_[:, 0].min(), 1.0)
            norm = 1000. / max_return
            self.data["reward"] *= norm
            self.data["return"] *= norm
            print(
                f"[D4RLOfflineDataset]: return range: [{prev_return_min}, {prev_return_max}], multiplying norm factor {norm}.")
        elif self.mode == "transition":
            ep_reward_ = []
            episode_reward = 0
            N = self.data["reward"].shape[0]
            for i in range(N):
                episode_reward += self.data["reward"][i]
                if self.data["end"][i]:
                    ep_reward_.append(episode_reward)
                    episode_reward = 0
            max_return = max(abs(min(ep_reward_)).item(), abs(max(ep_reward_)).item(),
                             (max(ep_reward_) - min(ep_reward_)).item(), 1.0)
            norm = 1000 / max_return
            self.data["reward"] *= norm
            print(
                f"[D4RLOfflineDataset]: return range: [{min(ep_reward_)}, {max(ep_reward_)}], multiplying norm factor {norm}.")

    def discounted_cum_sum(self, seq, discount):
        seq = seq.copy()
        for t in reversed(range(len(seq) - 1)):
            seq[t] += discount * seq[t + 1]
        return seq


class D4RLOfflineDataset2(torch.utils.data.IterableDataset):
    def __init__(
            self,
            pref_dataset,
            observation_space: gym.Space,
            action_space: gym.Space,
            env: str,
            segment_length: Optional[int] = None,
            batch_size: Optional[int] = None,
            capacity: Optional[int] = None,
            mode: str = "transition",
            padding_mode: str = "right",
            reward_scale: Optional[float] = None,
            reward_shift: Optional[float] = None,
            reward_normalize: bool = False,
            traj_length: int = 200,
    ):
        super().__init__()
        assert mode in {"transition", "trajectory"}, "Supported mode for D4RLOfflineDataset: {transition, trajectory}."
        assert padding_mode in {"left", "right", "none",
                                "Supported padding mode for D4RLOfflineDataset: {left, right, none}."}
        assert reward_scale is None == reward_shift is None, "reward_scale and reward_shift should be set simultaneously."
        assert not reward_normalize or reward_shift is None, "reward scale & shift and reward normalize can not be set simultaneously."

        self.env_name = env
        self.mode = mode
        self.padding_mode = padding_mode
        self.batch_size = 1 if batch_size is None else batch_size
        self.segment_length = segment_length
        self.capacity = capacity

        self.reward_scale = reward_scale
        self.reward_shift = reward_shift
        self.traj_length = traj_length
        self.reward_normalize = reward_normalize
        self.load_dataset(pref_dataset)

    def __len__(self):
        return self.data_size

    def __iter__(self):
        while True:
            sample = []
            for _ in range(self.batch_size):
                if self.padding_mode == "right":
                    traj_idx = np.random.choice(self.data_size, p=self.sample_prob)
                    start_idx = np.random.choice(self.traj_len[traj_idx])
                    s = {
                        k: self.pad_along_axis(
                            v[traj_idx, start_idx:min(start_idx + self.segment_length, self.traj_len[traj_idx])],
                            pad_to=self.segment_length) for k, v in self.data.items()
                    }
                    s["timestep"] = np.arange(start_idx, start_idx + self.segment_length)
                elif self.padding_mode == "left":
                    traj_idx = np.random.choice(self.data_size, p=self.sample_prob)
                    end_idx = np.random.choice(self.traj_len[traj_idx]) + 1
                    s = {
                        k: self.pad_along_axis(v[traj_idx, max(0, end_idx - self.segment_length):end_idx],
                                               pad_to=self.segment_length, direction="left") for k, v in
                        self.data.items()
                    }
                    s["timestep"] = np.maximum(np.arange(self.segment_length) + 1 - self.segment_length + end_idx,
                                               0)
                elif self.padding_mode == "none":
                    # traj_idx = np.random.choice(self.data_size, p=self.sample_prob)
                    # while self.traj_len[traj_idx] <= self.segment_length:
                    #     traj_idx = np.random.choice(self.data_size, p=self.sample_prob)
                    traj_idx = np.random.choice(self.data_size)
                    start_idx = np.random.choice(self.traj_len[traj_idx] - self.segment_length + 1)
                    s = {
                        k: v[traj_idx, start_idx:start_idx + self.segment_length] for k, v in self.data.items()
                    }
                    # s["labels"] = self.data["labels"][traj_idx]
                    s["timestep"] = np.arange(start_idx, start_idx + self.segment_length)
                sample.append(s)
            if len(sample) == 1:
                yield sample[0]
            else:
                yield {
                    "obs": np.stack([s["obs"] for s in sample], axis=0),
                    "action": np.stack([s["action"] for s in sample], axis=0),
                    "obs_2": np.stack([s["obs_2"] for s in sample], axis=0),
                    "action_2": np.stack([s["action_2"] for s in sample], axis=0),
                    "labels": np.stack([s["labels"] for s in sample], axis=0),
                    "mask": np.stack([s["mask"] for s in sample], axis=0),
                    "timestep": np.stack([s["timestep"] for s in sample], axis=0),
                }

    def pad_along_axis(self,
                       arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0, direction="right"
                       ) -> np.ndarray:
        pad_size = pad_to - arr.shape[axis]
        if pad_size <= 0:
            return arr

        npad = [(0, 0)] * arr.ndim
        if direction == "right":
            npad[axis] = (0, pad_size)
        else:
            npad[axis] = (pad_size, 0)
        return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)

    def load_dataset(self, pref_dataset):
        self.data_size = len(pref_dataset["observations"])
        self.traj_len = np.full((len(pref_dataset["observations"]),), self.traj_length)
        self.sample_prob = self.traj_len / self.traj_len.sum()
        self.data = pref_dataset
        # terminal = np.zeros((pref_dataset["observations"].shape[0], pref_dataset["observations"].shape[1], 1),
        #                     dtype=int)
        self.data = {
            "obs": pref_dataset["observations"],
            "action": pref_dataset["actions"],
            "obs_2": pref_dataset["observations_2"],
            "action_2": pref_dataset["actions_2"],
            "labels": pref_dataset["labels"],
            "mask": np.ones([pref_dataset["observations"].shape[0], pref_dataset["observations"].shape[1], 1],
                            dtype=np.float32),
        }

    @torch.no_grad()
    def relabel_reward(self, agent):
        assert hasattr(agent, "select_reward"), f"Agent {agent} must support relabel_reward!"
        bs = 64
        for i_batch in range((self.data_size - 1) // bs + 1):
            idx = np.arange(i_batch * bs, min((i_batch + 1) * bs, self.data_size))
            batch = {
                "obs": self.data["obs"][idx],
                "action": self.data["action"][idx],
                "next_obs": self.data["next_obs"][idx],
                "mask": self.data["mask"][idx]
            }
            batch = agent.format_batch(batch)
            reward = agent.select_reward(batch).detach().cpu().numpy()
            reward = reward * self.data["mask"][idx]
            self.data["reward"][idx] = reward

        if self.mode == "trajectory":
            # CHECK: may be bug. the max and min returns are not consistent with those computed in transition mode
            return_ = self.data["reward"].copy()
            for t in reversed(range(return_.shape[1] - 1)):
                return_[:, t] += return_[:, t + 1]
            self.data["return"] = return_
            # normalization
            prev_return_min, prev_return_max = return_[:, 0].min(), return_[:, 0].max()
            max_return = max(abs(return_[:, 0].max()), abs(return_[:, 0].min()),
                             return_[:, 0].max() - return_[:, 0].min(), 1.0)
            norm = 1000. / max_return
            self.data["reward"] *= norm
            self.data["return"] *= norm
            print(
                f"[D4RLOfflineDataset]: return range: [{prev_return_min}, {prev_return_max}], multiplying norm factor {norm}.")
        elif self.mode == "transition":
            ep_reward_ = []
            episode_reward = 0
            N = self.data["reward"].shape[0]
            for i in range(N):
                episode_reward += self.data["reward"][i]
                if self.data["end"][i]:
                    ep_reward_.append(episode_reward)
                    episode_reward = 0
            max_return = max(abs(min(ep_reward_)).item(), abs(max(ep_reward_)).item(),
                             (max(ep_reward_) - min(ep_reward_)).item(), 1.0)
            norm = 1000 / max_return
            self.data["reward"] *= norm
            print(
                f"[D4RLOfflineDataset]: return range: [{min(ep_reward_)}, {max(ep_reward_)}], multiplying norm factor {norm}.")

    def discounted_cum_sum(self, seq, discount):
        seq = seq.copy()
        for t in reversed(range(len(seq) - 1)):
            seq[t] += discount * seq[t + 1]
        return seq
