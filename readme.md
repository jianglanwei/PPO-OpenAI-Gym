# PPO for OpenAI Gym

An implementation of Proximal Policy Optimization (PPO) designed for OpenAI Gymnasium. This repository is built for immediate use on `BipedalWalker-v3` and `Humanoid-v5`. Its modular structure allows fast integration on new gym environments.

**BipedalWalker-v3**  
A bipedal agent learning to walk on uneven terrain.  
![BipedalWalker](gif/BipedalWalker-v3/06-05-25_03:09:03/teaser_epoch960_reward310.37.gif)

**Humanoid-v5**  
A humanoid robot learning basic locomotion.  
![Humanoid](gif/Humanoid-v5/06-06-25_19:08:43/teaser_epoch3730_reward6089.30.gif)


## 0. Installation

__Step 1:__ Create a new conda environment `gym`:
```
conda create -n gym python=3.10
conda activate gym
```


__Step 2:__ Clone this repository:

``` 
git clone https://github.com/jianglanwei/PPO-OpenAI-Gym
cd PPO-OpenAI-Gym
```

__Step 3:__ Install dependencies:

```
pip install -r requirements.txt
```

## 1. Getting Started: Train PPO on `BipedalWalker-v3`

### 1.1 Train From Scratch

To train a `BipedalWalker-v3` policy from scratch, execute:

```
python3 train.py --env BipedalWalker-v3
```

- Checkpoints are saved to `policy_ckpt/BipedalWalker-v3/<train_start_time>`. 
Only the top 5 checkpoints will be retained per run.
- Training hyperparameters can be customized in `config/BipedalWalker-v3.yaml`.
- Real-time training metrics are logged to **Weights & Biases** (`wandb`).

### 1.2 Resume Training

Use the `--resume_run` flag to load a checkpoint from a previous session and continue training:
```
python3 train.py --env BipedalWalker-v3 --resume_run <train_start_time>
```
<br>

This repository includes pretrained `BipedalWalker-v3` checkpoints from session
`06-05-25_03:09:03` (located [here](policy_ckpt/BipedalWalker-v3/06-05-25_03:09:03)). 
To resume training from the highest-reward checkpoint of that session, use:

```
python3 train.py --env BipedalWalker-v3 --resume_run 06-05-25_03:09:03
```
To resume from a specific epoch, add the `--load_epoch` flag:
```
python3 train.py --env BipedalWalker-v3 --resume_run 06-05-25_03:09:03 --load_epoch 990
```


### 1.3 Visualize Policy Rollout

Use `play.py` to render a trained agents. This script supports rendering real-time (`human` mode, default) or by generating GIF files (`rgb_array` mode, suitable for headless execution). The general command is
```
python3 play.py --env BipedalWalker-v3 --run <train_start_time> --epoch <epoch_number> --render_mode <human|rgb_array>
```
<br> 


For example, to visualize `BipedalWalker-v3` session `06-05-25_03:09:03` in a Gymnasium window, run:
```
python3 play.py --env BipedalWalker-v3 --run 06-05-25_03:09:03 --render_mode human
```
By default, this loads the checkpoint with the highest reward. Use `--epoch` to target a specific checkpoint:
```
python3 play.py --env BipedalWalker-v3 --run 06-05-25_03:09:03 --epoch 990 --render_mode human
```
<br>


> **The `Humanoid-v5` environment:**  
> This repository also includes tuned hyperparameters and pretrained checkpoints for `Humanoid-v5`.  
> 
> **Train:**
> 
> ```
> python3 train.py --env Humanoid-v5
> ```
> 
> **Virtualize Pretrained Policy:**  
> 
> ```
> python3 play.py --env Humanoid-v5 --run 06-06-25_19:08:43
> ```

## 2. Train PPO on Any Gym Environment

This repository is designed to be easily extensible. To train a PPO agent on a new
OpenAI Gym environment:

### 2.1 Create a Configuration File
Create a new YAML file in the `config/` directory named exactly after your
target environment ID (e.g., `LunarLander-v2.yaml`). Copy an existing config file
(`BipedalWalker-v3.yaml` or `Humanoid-v5.yaml`) as a template. This file defines all hyperparameters, such as learning rate, batch size, and the actor-critic network.

### 2.2 (Optional) Define Custom Actor-Critic Architecture

If your environment requires a specialized neural network (e.g., a CNN for pixel-based inputs):

1. Add a new Actor-Critic class in `module.py`. Refer to the existing classes in that file to ensure the input/output matches. 
  
2. Update the `actor_critic` field in the environment's config file (from Section 2.1) to match the name of your new class. 

### 2.3 Start Training

Start a new training session:

```
python3 train.py --env <env_name>
```
<br>

Resume from a previous session:
```
python3 train.py --env <env_name> --resume_run <train_start_time> --load_epoch <epoch_number>
```

### 2.4 Visualize Policy Behavior

Render the agent's performance and save the rollout as a GIF:

```
python3 play.py --env <env_name> --run <train_start_time> --epoch <epoch_number> --render_mode <human|rgb_array>
```