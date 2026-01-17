import gymnasium as gym
import torch
import numpy as np
import wandb
import os
from os.path import join
from glob import glob
import datetime
import module
import argparse
import yaml
from types import SimpleNamespace

GREEN = "\033[92m"
BOLD = "\033[1m"
RESET = "\033[0m"

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="BipedalWalker-v3")
parser.add_argument("--resume_run", type=str, default=None)
parser.add_argument("--load_epoch", type=int, default=None)
args = parser.parse_args()

# load configuration
cfg_path = f"config/{args.env}.yaml"
assert os.path.exists(cfg_path), f"config file not found: {cfg_path}"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
cfg = SimpleNamespace(**cfg)

def compute_gae(rewards: list, values: list, gamma: float, lam: float):
    """
    Compute Generalized Advantage Estimation (GAE) for a single trajectory.

    Args:
        rewards (list of length T): Sequence of rewards for the trajectory.
        values (list of length T): Sequence of value estimates for each state.
        gamma (float): Discount factor.
        lam (float): GAE lambda parameter for bias-variance tradeoff.

    Returns:
        advantages (torch.Tensor with shape [T]): Estimated advantage at each timestep.
    """
    advantages = []
    gae = 0
    values = values.copy()
    values.append(0)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return torch.tensor(advantages, dtype=torch.float32)


# device setup.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# parallel environments setup.
make_env = lambda: gym.make(args.env)
envs = gym.vector.SyncVectorEnv([make_env for _ in range(cfg.num_workers)])

obs_dim = envs.single_observation_space.shape[0]
act_dim = envs.single_action_space.shape[0]
max_episode_steps = envs.envs[0]._max_episode_steps

# initialize policy and optimizer.
actor_critic = getattr(module, cfg.actor_critic)
net: torch.nn.Module = actor_critic(obs_dim, act_dim).to(device)
if args.resume_run: # load checkpoint.
    ckpt_dir = join(cfg.ckpt_root_dir, args.env, args.resume_run)
    if args.load_epoch:
        target_ckpt_path = glob(join(ckpt_dir, f"epoch{args.load_epoch}_*.pt"))
        assert target_ckpt_path, f"No checkpoint found for epoch {args.load_epoch} in {args.resume_run}."
        [target_ckpt_path] = target_ckpt_path
    else:
        # locate the best checkpoint
        extract_reward = lambda f: float(f.split("reward")[-1].replace(".pt", "")) 
        ckpt_paths = sorted(glob(join(ckpt_dir, "epoch*.pt")))
        assert ckpt_paths, "No checkpoints found in directory."
        target_ckpt_path = max(ckpt_paths, key=extract_reward)
    net.load_state_dict(torch.load(target_ckpt_path, map_location=device))
    net.eval()
    start_epoch = int(target_ckpt_path.split("epoch")[-1].split("_")[0]) + 1
    print(f"\n{GREEN}{BOLD}Resuming from checkpoint {args.resume_run}. Starting epoch {start_epoch}.\n{RESET}")
else: # train new policy.
    start_epoch = 0
optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)

# initialize WandB logging and checkpoint folder.
run_time = datetime.datetime.now()
run_name = run_time.strftime("%m-%d-%y_%H:%M:%S")
wandb.init(project=f"PPO_{args.env}", name=run_name)

ckpt_run_dir = join(cfg.ckpt_root_dir, args.env, run_name)
os.makedirs(ckpt_run_dir, exist_ok=True)

for epoch in range(start_epoch, start_epoch + cfg.training_epochs):

    # initialize per-episode trajectory buffers.
    # each list stores a single trajectory and is cleared when the episode ends.
    obs_traj_buf = [[] for _ in range(cfg.num_workers)]    # observations
    act_traj_buf = [[] for _ in range(cfg.num_workers)]    # actions
    logp_traj_buf = [[] for _ in range(cfg.num_workers)]   # action log probabilities
    rew_traj_buf = [[] for _ in range(cfg.num_workers)]    # rewards
    val_traj_buf = [[] for _ in range(cfg.num_workers)]    # value estimates

    # initialize rollout data accumulators.
    obs_buf = []     # observations
    act_buf = []     # actions
    logp_buf = []    # action log probabilities
    adv_buf = []     # advantage estimates.
    ret_buf = []     # returns (real state values)

    # WandB outputs.
    episode_lengths = []
    episode_rewards = []

    # reset all environments at the start of the epoch.
    obs, info = envs.reset()

    # collect rollout data across parallel environments.
    for rollout_step in range(0, cfg.rollout_steps_per_epoch, cfg.num_workers):
        # sample actions and value estimates from current policy.
        with torch.no_grad():
            obs_cuda = torch.tensor(obs, dtype=torch.float32, device=device)
            action_dist, value = net(obs_cuda)
            action = action_dist.sample()
            action_logp = action_dist.log_prob(action).sum(dim=-1)
            action = action.cpu()

        # step environments with selected actions.
        next_obs, reward, terminated, truncated, info = envs.step(action.numpy())

        # save data by environments.
        for env in range(cfg.num_workers):
            obs_traj_buf[env].append(torch.from_numpy(obs[env]))
            act_traj_buf[env].append(action[env])
            logp_traj_buf[env].append(action_logp[env].item())
            rew_traj_buf[env].append(reward[env].item())
            val_traj_buf[env].append(value[env].item())

            # check if episode finished for this environment.
            done = terminated[env] or truncated[env]
            last_step = rollout_step + cfg.num_workers == cfg.rollout_steps_per_epoch

            if done or last_step:
                # episode finished or rollout ends — compute advantages and returns.
                rewards, values = rew_traj_buf[env], val_traj_buf[env]
                advantages = compute_gae(rewards, values, cfg.gamma, cfg.gae_lambda)
                returns = advantages + torch.tensor(values, dtype=torch.float32)

                # append collected trajectory to data accumulators.
                adv_buf.append(advantages)
                ret_buf.append(returns)
                act_buf.append(torch.stack(act_traj_buf[env]))
                obs_buf.append(torch.stack(obs_traj_buf[env]))
                logp_buf.append(torch.tensor(logp_traj_buf[env]))

                # record stats and clear trajectory buffers for this env.
                if done:
                    episode_lengths.append(len(rewards))
                    episode_rewards.append(sum(rewards))
                obs_traj_buf[env], act_traj_buf[env], logp_traj_buf[env], \
                    rew_traj_buf[env], val_traj_buf[env] = [], [], [], [], []

        # update current observations.
        obs = next_obs

        # reset terminated or truncated environments.
        done_mask = np.logical_or(terminated, truncated)
        if np.any(done_mask):
            obs, info = envs.reset(options={"reset_mask": done_mask})

    # consolidate data accumulators and move to GPU. N is rollout_step_per_epoch.
    # shapes: obs_buf [N, obs_dim]; act_buf [N, act_dim]; rest [N].
    obs_buf = torch.cat(obs_buf, dim=0).to(dtype=torch.float32, device=device) 
    act_buf = torch.cat(act_buf, dim=0).to(dtype=torch.float32, device=device)
    logp_buf = torch.cat(logp_buf, dim=0).to(dtype=torch.float32, device=device) 
    adv_buf = torch.cat(adv_buf, dim=0).to(dtype=torch.float32, device=device)
    ret_buf = torch.cat(ret_buf, dim=0).to(dtype=torch.float32, device=device)

    # update policy and value networks.
    for grad_step in range(cfg.grad_steps_per_update):
        # forward pass through policy network.
        action_dist, value = net(obs_buf)
        new_action_logp = action_dist.log_prob(act_buf).sum(dim=-1)

        # compute importance sampling ratio (new / old policy).
        action_prob_ratio = torch.exp(new_action_logp - logp_buf)

        # compute surrogate loss.
        adv_buf = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)
        surr1 = action_prob_ratio * adv_buf
        surr2 = torch.clamp(action_prob_ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_buf
        policy_loss = -torch.min(surr1, surr2).mean()

        # value function loss (mean squared error).
        ret_buf = (ret_buf - ret_buf.mean()) / (ret_buf.std() + 1e-8)
        value = (value - value.mean()) / (value.std() + 1e-8)
        value_loss = ((ret_buf - value.squeeze()) ** 2).mean()

        # entropy bonus to encourage exploration.
        entropy_bonus = action_dist.entropy().sum(-1).mean()

        # combined loss.
        loss = policy_loss + cfg.value_loss_coef * value_loss - cfg.entropy_coef * entropy_bonus

        # backpropagation and optimizer step.
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
        optimizer.step()
    
    # log metrics to wandb
    reward_mean = sum(episode_rewards) / len(episode_rewards)
    episode_length_mean = sum(episode_lengths) / len(episode_lengths)

    wandb.log({
        "reward_mean": reward_mean,
        "episode_length_mean": episode_length_mean,
        "total_loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy_bonus": entropy_bonus.item(),
    }, step=epoch)

    print(f"{BOLD}epoch {epoch}/{start_epoch + cfg.training_epochs}:{RESET}")
    print(f"reward_mean {reward_mean:.2f};\t total_loss {loss.item():.2f}.\n")

    # save model checkpoint.
    if epoch % cfg.ckpt_save_interval == 0:
        ckpt_paths = sorted(glob(join(ckpt_run_dir, "epoch*.pt")))
        ckpt_filename = f"epoch{epoch}_reward{reward_mean:.2f}.pt"
        ckpt_path = join(ckpt_run_dir, ckpt_filename)
        if len(ckpt_paths) == cfg.num_ckpts: # only keep 5 checkpoints with highest rewards.
            extract_reward = lambda f: float(f.split("reward")[-1].replace(".pt", ""))
            worst_ckpt_path = min(ckpt_paths, key=extract_reward)
            if reward_mean > extract_reward(worst_ckpt_path):
                os.remove(worst_ckpt_path)
                torch.save(net.state_dict(), ckpt_path)
        else:
              torch.save(net.state_dict(), ckpt_path)
        # save the current checkpoint as the most recent checkpoint.
        ckpt_path = join(ckpt_run_dir, "final_epoch.pt")
        torch.save(net.state_dict(), ckpt_path)