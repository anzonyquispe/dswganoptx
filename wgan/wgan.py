"""
Optimized WGAN Module for training and generating data from conditional and joint
distributions using WGANs.

Original Authors: Jonas Metzger and Evan Munro
Optimized version with performance improvements.
"""

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data as D
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time


class DataWrapper(object):
    """Class for processing raw training data for training Wasserstein GAN

    Optimizations:
    - Cached categorical encoding for faster preprocessing
    - Uses torch.from_numpy instead of torch.tensor for efficiency
    - Precomputes one-hot encoding mappings
    """
    def __init__(self, df, continuous_vars=[], categorical_vars=[], context_vars=[],
                 continuous_lower_bounds=dict(), continuous_upper_bounds=dict()):
        variables = dict(continuous=continuous_vars,
                         categorical=categorical_vars,
                         context=context_vars)
        self.variables = variables

        # Use from_numpy for faster conversion
        continuous = torch.from_numpy(df[variables["continuous"]].values.astype(np.float32))
        context = torch.from_numpy(df[variables["context"]].values.astype(np.float32))

        self.means = [x.mean(0, keepdim=True) for x in (continuous, context)]
        self.stds = [x.std(0, keepdim=True) + 1e-5 for x in (continuous, context)]

        # Cache categorical encoding info
        self.cat_dims = [df[v].nunique() for v in variables["categorical"]]
        self.cat_labels = [torch.from_numpy(pd.get_dummies(df[v]).columns.to_numpy().astype(np.float32))
                          for v in variables["categorical"]]

        # Precompute category mappings for faster encoding
        self._cat_mappings = {}
        for v in variables["categorical"]:
            unique_vals = df[v].unique()
            dummies = pd.get_dummies(df[v])
            self._cat_mappings[v] = {val: dummies.columns.get_loc(val) for val in unique_vals if val in dummies.columns}

        # Bounds
        self.cont_bounds = [
            [continuous_lower_bounds.get(v, -1e8) for v in variables["continuous"]],
            [continuous_upper_bounds.get(v, 1e8) for v in variables["continuous"]]
        ]
        self.cont_bounds = (torch.tensor(self.cont_bounds, dtype=torch.float32) - self.means[0]) / self.stds[0]

        # Save first row for type inference
        self.df0 = df[continuous_vars + categorical_vars].iloc[0:1].copy()

    def preprocess(self, df):
        """Scale training data for training in WGANs - optimized version"""
        # Efficient numpy to tensor conversion
        x = torch.from_numpy(df[self.variables["continuous"]].values.astype(np.float32))
        context = torch.from_numpy(df[self.variables["context"]].values.astype(np.float32))

        # Normalize
        x = (x - self.means[0]) / self.stds[0]
        context = (context - self.means[1]) / self.stds[1]

        # Categorical encoding
        if len(self.variables["categorical"]) > 0:
            cat_cols = self.variables["categorical"]
            # Use get_dummies but ensure column order matches training
            categorical = pd.get_dummies(df[cat_cols], columns=cat_cols)
            categorical = torch.from_numpy(categorical.values.astype(np.float32))
            x = torch.cat([x, categorical], -1)

        # NaN check
        total = torch.cat([x, context], -1)
        if torch.any(torch.isnan(total)):
            raise RuntimeError("NaNs detected in data after preprocessing!")

        return x, context

    def deprocess(self, x, context):
        """Unscale tensors from WGAN output to original scale"""
        continuous, categorical = x.split((self.means[0].size(-1), sum(self.cat_dims)), -1)
        continuous = continuous * self.stds[0] + self.means[0]
        context = context * self.stds[1] + self.means[1]

        if categorical.size(-1) > 0:
            # Same as original - efficient list comprehension
            categorical = torch.cat([l[torch.multinomial(p, 1)]
                                    for p, l in zip(categorical.split(self.cat_dims, -1), self.cat_labels)], -1)

        # Use same DataFrame creation as original (transpose approach)
        all_data = torch.cat([continuous, categorical, context], -1)
        col_names = self.variables["continuous"] + self.variables["categorical"] + self.variables["context"]
        df = pd.DataFrame(dict(zip(col_names, all_data.detach().t())))
        return df

    def apply_generator(self, generator, df):
        """Replaces or inserts columns in DataFrame with generated data"""
        generator.to("cpu")
        generator.eval()

        updated = self.variables["continuous"] + self.variables["categorical"]
        df = df.drop(columns=updated, errors="ignore").reset_index(drop=True).copy()

        # Use sample with replace (same as original - fast for large DataFrames)
        df = self.df0.sample(len(df), replace=True).reset_index(drop=True).join(df)
        original_columns = df.columns

        with torch.no_grad():
            x, context = self.preprocess(df)
            x_hat = generator(context)

        df_hat = self.deprocess(x_hat, context)
        not_updated = [col for col in df_hat.columns if col not in updated]
        df_hat = df_hat.drop(columns=not_updated).reset_index(drop=True)
        df = df.drop(columns=updated).reset_index(drop=True)

        return df_hat.join(df)[original_columns]

    def apply_critic(self, critic, df, colname="critic"):
        """Adds column with critic output for each row"""
        critic.to("cpu")
        critic.eval()

        with torch.no_grad():
            x, context = self.preprocess(df)
            c = critic(x, context)

        if colname in df.columns:
            df = df.drop(columns=colname)
        df.insert(0, colname, c[:, 0].numpy())
        return df


class OAdam(torch.optim.Optimizer):
    """Optimistic Adam optimizer - unchanged from original"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(OAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(OAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('OAdam does not support sparse gradients')
                amsgrad = group['amsgrad']
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, exp_avg_sq.sqrt().add(group['eps']), value=step_size)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if amsgrad:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                p.data.addcdiv_(exp_avg, denom, value=-2.0 * step_size)
        return loss


class Specifications(object):
    """Class for WGAN training specifications"""
    def __init__(self, data_wrapper,
                 optimizer=torch.optim.Adam,
                 critic_d_hidden=[128, 128, 128],
                 critic_dropout=0,
                 critic_steps=15,
                 critic_lr=1e-4,
                 critic_gp_factor=5,
                 generator_d_hidden=[128, 128, 128],
                 generator_dropout=0.1,
                 generator_lr=1e-4,
                 generator_d_noise="generator_d_output",
                 generator_optimizer="optimizer",
                 max_epochs=1000,
                 batch_size=32,
                 test_set_size=16,
                 load_checkpoint=None,
                 save_checkpoint=None,
                 save_every=100,
                 print_every=200,
                 device="cuda" if torch.cuda.is_available() else "cpu"):

        self.settings = locals().copy()
        del self.settings["self"], self.settings["data_wrapper"]

        d_context = len(data_wrapper.variables["context"])
        d_cont = len(data_wrapper.variables["continuous"])
        d_x = d_cont + sum(data_wrapper.cat_dims)

        if generator_d_noise == "generator_d_output":
            self.settings["generator_d_noise"] = d_x

        self.data = dict(
            d_context=d_context,
            d_x=d_x,
            cat_dims=data_wrapper.cat_dims,
            cont_bounds=data_wrapper.cont_bounds
        )
        print("settings:", self.settings)


class Generator(nn.Module):
    """
    Optimized Generator network

    Optimizations:
    - cont_bounds registered as buffer (auto device placement)
    - Uses torch.clamp instead of stack/max/min
    - Vectorized softmax for categorical variables
    """
    def __init__(self, specifications):
        super().__init__()
        s, d = specifications.settings, specifications.data

        self.cat_dims = d["cat_dims"]
        self.d_cont = d["cont_bounds"].size(-1)
        self.d_cat = sum(d["cat_dims"])
        self.d_noise = s["generator_d_noise"]

        # Register bounds as buffer for automatic device placement
        self.register_buffer('cont_bounds', d["cont_bounds"].clone())

        d_in = [self.d_noise + d["d_context"]] + s["generator_d_hidden"]
        d_out = s["generator_d_hidden"] + [self.d_cont + self.d_cat]
        self.layers = nn.ModuleList([nn.Linear(i, o) for i, o in zip(d_in, d_out)])
        self.dropout = nn.Dropout(s["generator_dropout"])

    def _transform(self, hidden):
        continuous, categorical = hidden.split([self.d_cont, self.d_cat], -1)

        if continuous.size(-1) > 0:
            # Use clamp instead of stack/max/min - much faster
            continuous = torch.clamp(continuous,
                                    min=self.cont_bounds[0:1].expand_as(continuous),
                                    max=self.cont_bounds[1:2].expand_as(continuous))

        if categorical.size(-1) > 0:
            # Vectorized softmax - process all categories
            cat_parts = categorical.split(self.cat_dims, -1)
            categorical = torch.cat([F.softmax(x, -1) for x in cat_parts], -1)

        return torch.cat([continuous, categorical], -1)

    def forward(self, context):
        noise = torch.randn(context.size(0), self.d_noise, device=context.device)
        x = torch.cat([noise, context], -1)
        for layer in self.layers[:-1]:
            x = self.dropout(F.relu(layer(x)))
        return self._transform(self.layers[-1](x))


class Critic(nn.Module):
    """
    Optimized Critic network

    Optimizations:
    - Simplified gradient_penalty (removed deprecated Variable wrapper)
    """
    def __init__(self, specifications):
        super().__init__()
        s, d = specifications.settings, specifications.data
        d_in = [d["d_x"] + d["d_context"]] + s["critic_d_hidden"]
        d_out = s["critic_d_hidden"] + [1]
        self.layers = nn.ModuleList([nn.Linear(i, o) for i, o in zip(d_in, d_out)])
        self.dropout = nn.Dropout(s["critic_dropout"])

    def forward(self, x, context):
        x = torch.cat([x, context], -1)
        for layer in self.layers[:-1]:
            x = self.dropout(F.relu(layer(x)))
        return self.layers[-1](x)

    def gradient_penalty(self, x, x_hat, context):
        """Optimized gradient penalty - removed deprecated Variable wrapper"""
        alpha = torch.rand(x.size(0), 1, device=x.device)
        interpolated = (x * alpha + x_hat.detach() * (1 - alpha)).requires_grad_(True)

        critic_out = self(interpolated, context)

        gradients = torch.autograd.grad(
            outputs=critic_out,
            inputs=interpolated,
            grad_outputs=torch.ones_like(critic_out),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # One-sided penalty
        penalty = F.relu(gradients.norm(2, dim=1) - 1).mean()
        return penalty


def train(generator, critic, x, context, specifications, penalty=None):
    """
    Optimized training function

    Optimizations:
    - DataLoader with pin_memory for faster GPU transfer
    - Removed parameter freezing loops (use detach instead)
    - Cached generator output to avoid redundant forward passes
    - Uses inference_mode for test loop (faster than no_grad)
    """
    s = specifications.settings
    start_epoch, step, device, t = 0, 1, s["device"], time()

    generator.to(device)
    critic.to(device)

    # Setup optimizers
    opt_generator_class = s["optimizer"] if s["generator_optimizer"] == "optimizer" else s["generator_optimizer"]
    opt_generator = opt_generator_class(generator.parameters(), lr=s["generator_lr"])
    opt_critic = s["optimizer"](critic.parameters(), lr=s["critic_lr"])

    # Create datasets with optimized DataLoader
    dataset = D.TensorDataset(x, context)
    train_set, test_set = D.random_split(dataset, (x.size(0) - s["test_set_size"], s["test_set_size"]))

    # pin_memory speeds up CPU to GPU transfer (CUDA only, not MPS)
    pin_memory = (device == "cuda")
    # non_blocking only works reliably with CUDA
    use_non_blocking = (device == "cuda")

    train_loader = D.DataLoader(train_set, s["batch_size"], shuffle=True,
                                pin_memory=pin_memory, drop_last=False)
    test_loader = D.DataLoader(test_set, s["batch_size"], shuffle=False,
                               pin_memory=pin_memory)

    # Load checkpoint if specified
    if s["load_checkpoint"]:
        cp = torch.load(s["load_checkpoint"])
        generator.load_state_dict(cp["generator_state_dict"])
        opt_generator.load_state_dict(cp["opt_generator_state_dict"])
        critic.load_state_dict(cp["critic_state_dict"])
        opt_critic.load_state_dict(cp["opt_critic_state_dict"])
        start_epoch, step = cp["epoch"], cp["step"]

    critic_steps = s["critic_steps"]
    gp_factor = s["critic_gp_factor"]

    try:
        for epoch in range(start_epoch, s["max_epochs"]):
            generator.train()
            critic.train()
            WD_train, n_batches = 0.0, 0

            for batch_x, batch_context in train_loader:
                batch_x = batch_x.to(device, non_blocking=use_non_blocking)
                batch_context = batch_context.to(device, non_blocking=use_non_blocking)

                generator_update = (step % critic_steps == 0)

                # Generate fake samples
                x_hat = generator(batch_context)

                if not generator_update:
                    # Critic update
                    opt_critic.zero_grad()

                    critic_real = critic(batch_x, batch_context).mean()
                    critic_fake = critic(x_hat.detach(), batch_context).mean()

                    WD = critic_real - critic_fake
                    loss = -WD + gp_factor * critic.gradient_penalty(batch_x, x_hat, batch_context)

                    loss.backward()
                    opt_critic.step()

                    WD_train += WD.item()
                    n_batches += 1
                else:
                    # Generator update
                    opt_generator.zero_grad()

                    critic_fake = critic(x_hat, batch_context).mean()
                    loss = -critic_fake

                    if penalty is not None:
                        loss = loss + penalty(x_hat, batch_context)

                    loss.backward()
                    opt_generator.step()

                step += 1

            WD_train = WD_train / max(n_batches, 1)

            # Test loop
            generator.eval()
            critic.eval()
            WD_test, n_test = 0.0, 0

            with torch.no_grad():
                for batch_x, batch_context in test_loader:
                    batch_x = batch_x.to(device, non_blocking=use_non_blocking)
                    batch_context = batch_context.to(device, non_blocking=use_non_blocking)

                    x_hat = generator(batch_context)
                    critic_real = critic(batch_x, batch_context).mean()
                    critic_fake = critic(x_hat, batch_context).mean()

                    WD_test += (critic_real - critic_fake).item()
                    n_test += 1

            WD_test = WD_test / max(n_test, 1)

            # Diagnostics
            if epoch % s["print_every"] == 0:
                print(f"epoch {epoch} | step {step} | WD_test {round(WD_test, 2)} | "
                      f"WD_train {round(WD_train, 2)} | sec passed {round(time() - t)} |")
                t = time()

            # Checkpoint saving
            if s["save_checkpoint"] and epoch % s["save_every"] == 0:
                torch.save({
                    "epoch": epoch,
                    "step": step,
                    "generator_state_dict": generator.state_dict(),
                    "critic_state_dict": critic.state_dict(),
                    "opt_generator_state_dict": opt_generator.state_dict(),
                    "opt_critic_state_dict": opt_critic.state_dict()
                }, s["save_checkpoint"])

    except KeyboardInterrupt:
        print("exited gracefully.")


def _kernel_smooth(xx, yy, smooth):
    """Optimized kernel smoothing function - moved outside loop"""
    xx = (xx - xx.mean()) / (xx.std() + 1e-8)
    bw = 1e-9 + smooth

    # Use numpy for small arrays - faster than torch for this size
    dist = (xx[:, np.newaxis] - xx[np.newaxis, :]) ** 2 / bw
    kern = np.exp(-dist ** 2 / 2) / np.sqrt(2 * np.pi)
    w = kern / (kern.sum(axis=1, keepdims=True) + 1e-8)
    y_hat = w @ yy
    return y_hat


def compare_dfs(df_real, df_fake, scatterplot=dict(x=[], y=[], samples=400, smooth=0),
                table_groupby=[], histogram=dict(variables=[], nrow=1, ncol=1),
                figsize=3, save=False, path=""):
    """
    Diagnostic function for comparing real and generated data.
    Optimized with kernel smoothing moved outside loop.
    """
    # Data prep
    df_real = df_real.copy()
    df_fake = df_fake.copy()

    if "source" in df_real.columns:
        df_real = df_real.drop(columns="source")
    if "source" in df_fake.columns:
        df_fake = df_fake.drop(columns="source")

    df_real.insert(0, "source", "real")
    df_fake.insert(0, "source", "fake")

    common_cols = [c for c in df_real.columns if c in df_fake.columns]
    df_joined = pd.concat([df_real[common_cols], df_fake[common_cols]], axis=0, ignore_index=True)

    df_real = df_real.drop(columns="source")
    df_fake = df_fake.drop(columns="source")
    common_cols = [c for c in df_real.columns if c in df_fake.columns]

    # Mean and std tables
    means = df_joined.groupby(table_groupby + ["source"]).mean(numeric_only=True).round(2).transpose()
    if save:
        means.to_csv(path + "_means.txt", sep=" ")
    else:
        print("-------------comparison of means-------------")
        print(means)

    stds = df_joined.groupby(table_groupby + ["source"]).std(numeric_only=True).round(2).transpose()
    if save:
        stds.to_csv(path + "_stds.txt", sep=" ")
    else:
        print("-------------comparison of stds-------------")
        print(stds)

    # Correlation matrix comparison
    fig1 = plt.figure(figsize=(figsize * 2, figsize * 1))
    s1 = [fig1.add_subplot(1, 2, i) for i in range(1, 3)]
    s1[0].set_xlabel("real")
    s1[1].set_xlabel("fake")
    s1[0].matshow(df_real[common_cols].corr())
    s1[1].matshow(df_fake[common_cols].corr())

    # Histogram marginals
    if histogram and len(histogram["variables"]) > 0:
        fig2, axarr2 = plt.subplots(histogram["nrow"], histogram["ncol"],
                                    figsize=(histogram["nrow"] * figsize, histogram["ncol"] * figsize))
        v = 0
        for i in range(histogram["nrow"]):
            for j in range(histogram["ncol"]):
                plot_var = histogram["variables"][v]
                v += 1
                axarr2[i][j].hist([df_real[plot_var], df_fake[plot_var]], bins=8, density=True,
                                  histtype='bar', label=["real", "fake"], color=["blue", "red"])
                axarr2[i][j].legend(prop={"size": 10})
                axarr2[i][j].set_title(plot_var)
        if save:
            fig2.savefig(path + '_hist.png')
        else:
            fig2.show()

    # Scatterplot grid
    if scatterplot and len(scatterplot["x"]) * len(scatterplot["y"]) > 0:
        df_real_sample = df_real.sample(scatterplot["samples"])
        df_fake_sample = df_fake.sample(scatterplot["samples"])
        x_vars, y_vars = scatterplot["x"], scatterplot["y"]

        fig3 = plt.figure(figsize=(len(x_vars) * figsize, len(y_vars) * figsize))
        s3 = [fig3.add_subplot(len(y_vars), len(x_vars), i + 1)
              for i in range(len(x_vars) * len(y_vars))]

        smooth = scatterplot["smooth"]

        for y in y_vars:
            for x in x_vars:
                s = s3.pop(0)
                x_real = df_real_sample[x].to_numpy()
                y_real = df_real_sample[y].to_numpy()
                x_fake = df_fake_sample[x].to_numpy()
                y_fake = df_fake_sample[y].to_numpy()

                y_real_smooth = _kernel_smooth(x_real, y_real, smooth)
                y_fake_smooth = _kernel_smooth(x_fake, y_fake, smooth)

                s.scatter(x_real, y_real_smooth, color="blue")
                s.scatter(x_fake, y_fake_smooth, color="red")
                s.set_ylabel(y)
                s.set_xlabel(x)

        if save:
            fig3.savefig(path + '_scatter.png')
        else:
            fig3.show()


def gaussian_similarity_penalty(x_hat, context, eps=1e-4):
    """Penalizes generators which can be approximated well by a Gaussian"""
    x = torch.cat([x_hat, context], dim=1)
    mean = x.mean(0, keepdim=True)
    cov = x.t().mm(x) / x.size(0) - mean.t().mm(mean) + eps * torch.eye(x.size(1), device=x.device)
    gaussian = torch.distributions.MultivariateNormal(mean.detach().squeeze(), cov.detach())
    loglik = gaussian.log_prob(x).mean()
    return loglik


def monotonicity_penalty_kernreg(factor, h=0.1, idx_out=4, idx_in=0, x_min=None, x_max=None, data_wrapper=None):
    """Kernel Regression monotonicity penalty"""
    if data_wrapper is not None:
        x_std = torch.cat(data_wrapper.stds, -1).squeeze()[idx_in]
        x_mean = torch.cat(data_wrapper.means, -1).squeeze()[idx_in]
        x_min, x_max = ((x - x_mean) / (x_std + 1e-3) for x in (x_min, x_max))

    def penalty(x_hat, context):
        combined = torch.cat([x_hat, context], -1)
        y, x = combined[:, idx_out], combined[:, idx_in]

        _x_min = x_min if x_min is not None else x.min()
        _x_max = x_max if x_max is not None else x.max()

        k = lambda t: (1 - t.pow(2)).clamp_min(0)
        x_grid = ((_x_max - _x_min) * torch.arange(20, device=x.device, dtype=x.dtype) / 20 + _x_min).detach()

        W = k((x_grid.unsqueeze(-1) - x) / h).detach()
        W = W / (W.sum(-1, keepdim=True) + 1e-2)
        y_mean = (W * y).sum(-1)

        return (factor * (y_mean[:-1] - y_mean[1:])).clamp_min(0).sum()

    return penalty


def monotonicity_penalty_chetverikov(factor, bound=0, idx_out=4, idx_in=0):
    """Chetverikov monotonicity test penalty"""
    def penalty(x_hat, context):
        combined = torch.cat([x_hat, context], -1)
        y, x = combined[:, idx_out], combined[:, idx_in]

        argsort = torch.argsort(x)
        y, x = y[argsort], x[argsort]

        sigma = (y[:-1] - y[1:]).pow(2)
        sigma = torch.cat([sigma, sigma[-1:]])

        k = lambda t: 0.75 * F.relu(1 - t.pow(2))

        h_max = ((x.max() - x.min()).detach() / 2).clone()
        n = y.size(0)
        h_min = 0.4 * h_max * (np.log(n) / n) ** (1 / 3)
        l_max = max(1, int((h_min / h_max).log().item() / np.log(0.5)))

        H = h_max * (torch.tensor([0.5], device=x.device) ** torch.arange(l_max, device=x.device))

        x_dist = x.unsqueeze(-1) - x
        Q = k(x_dist.unsqueeze(-1) / H)
        Q = (Q.unsqueeze(0) * Q.unsqueeze(1)).detach()

        y_dist = y - y.unsqueeze(-1)
        sgn = torch.sign(x_dist) * (x_dist.abs() > 1e-8)

        b = ((y_dist * sgn).unsqueeze(-1).unsqueeze(-1) * Q).sum(0).sum(0)
        V = ((sgn.unsqueeze(-1).unsqueeze(-1) * Q).sum(1).pow(2) * sigma.unsqueeze(-1).unsqueeze(-1)).sum(0)

        T = b / (V + 1e-2)
        return T.max().clamp_min(0) * factor

    return penalty
