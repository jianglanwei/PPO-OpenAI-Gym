import gymnasium as gym
import torch
import os
from os.path import join
from glob import glob
import imageio
import module
import argparse
import yaml
import datetime
from types import SimpleNamespace

GREEN = "\033[92m"
BOLD = "\033[1m"
RESET = "\033[0m"

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="BipedalWalker-v3")
parser.add_argument("--run", type=str, default="06-05-25_03:09:03")
parser.add_argument("--epoch", type=str, default=None)
parser.add_argument("--render_mode", type=str, default="human")
args = parser.parse_args()

# load configuration
cfg_path = f"config/{args.env}.yaml"
assert os.path.exists(cfg_path), f"config file not found: {cfg_path}"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
cfg = SimpleNamespace(**cfg)


# locate the best checkpoint
ckpt_dir = join(cfg.ckpt_root_dir, args.env, args.run) 
if args.epoch:
    target_ckpt_path = glob(join(ckpt_dir, f"epoch{args.epoch}*_.pt"))
    assert target_ckpt_path, f"No checkpoint found for epoch {args.epoch} in {args.run}."
    [target_ckpt_path] = target_ckpt_path
else:
    extract_reward = lambda f: float(f.split("reward")[-1].replace(".pt", ""))
    ckpt_paths = sorted(glob(join(ckpt_dir, "epoch*.pt")))
    assert ckpt_paths, "No checkpoints found in directory."
    target_ckpt_path = max(ckpt_paths, key=extract_reward)
print(f"\n{GREEN}{BOLD}Loading checkpoint: {target_ckpt_path}\n{RESET}")

# environment setup
env = gym.make(args.env, render_mode=args.render_mode)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load model
actor_critic = getattr(module, cfg.actor_critic)
net: torch.nn = actor_critic(obs_dim, act_dim).to(device)
net.load_state_dict(torch.load(target_ckpt_path, map_location=device))
net.eval()

# rollout and capture frames
obs, _ = env.reset()
frames = []
episode_reward = 0

with torch.no_grad():
    terminated, truncated = False, False
    while not (terminated or truncated):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action_dist, _ = net(obs_tensor)
        action = action_dist.mean[0].cpu().numpy()  # Deterministic action

        obs, reward, terminated, truncated, _ = env.step(action)
        frame = env.render()
        frames.append(frame)
        episode_reward += reward

env.close()

# save gif
if args.render_mode == "rgb_array":
    gif_run_dir = join(cfg.gif_root_dir, args.env, args.run)
    os.makedirs(gif_run_dir, exist_ok=True)

    save_time = datetime.datetime.now()
    save_time = save_time.strftime("%m-%d-%y_%H:%M:%S")
    epoch = target_ckpt_path.split("epoch")[-1].split("_")[0]
    gif_filename = f"{save_time}_epoch{epoch}_reward{episode_reward:.2f}.gif"
    gif_path = join(gif_run_dir, gif_filename)
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"\n{GREEN}{BOLD}Saved rollout to {gif_path}.\n{RESET}")
    print(f"The reward achieved in this rollout is {episode_reward:.2f}")