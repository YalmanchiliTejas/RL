import argparse
import csv
import os
import random
from dataclasses import dataclass

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.wrappers import RecordVideo
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm



@dataclass
class Config:
    total_steps: int = 300_000
    learning_starts: int = 5_000
    buffer_size: int = 100_000
    batch_size: int = 64
    gamma: float = 0.99
    lr: float = 1e-3
    train_freq: int = 1
    target_update_freq: int = 1_000

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 120_000

    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_eps: float = 1e-6

class QNetwork(nn.Module):
    
    def __init__(self,state_dim, action_dim):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        transision = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transision)
        else:

            self.buffer[self.position] = transision
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size, beta=None):
        batch_indexes = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in batch_indexes]

        state, action, reward, next_state, done = zip(*batch)

        weights = np.ones((batch_size, 1), dtype=np.float32)

        return (
            torch.tensor(np.array(state), dtype=torch.float32),
            torch.tensor(action, dtype=torch.long).view(-1, 1),
            torch.tensor(reward, dtype=torch.float32).view(-1, 1),
            torch.tensor(np.array(next_state), dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32).view(-1, 1),
            torch.tensor(weights, dtype=torch.float32),
            batch_indexes
        )
    def update_priorities(self, batch_indexes, priorities):
        pass
    def __len__(self):
        return len(self.buffer)




class PrioritizedReplayBuffer(ReplayBuffer):

    def __init__(self, capacity, alpha=0.6, eps=1e-6):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
        self.eps = eps



    #During insertion, we can assign a maximum priority to new transitions to ensure they are sampled at least once.
    
    def push(self, state, action, reward, next_state, done):
        transision = (state, action, reward, next_state, done)

        if len(self.buffer) == 0:
            max_priority = 1.0
        else:
            max_priority = self.priorities[:len(self.buffer)].max()

            if max_priority <= 0:
                max_priority = 1.0
        

        if len(self.buffer) < self.capacity:
            self.buffer.append(transision)
        else:

            self.buffer[self.position] = transision
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    #Sample it based on the priorities. The probability of sampling a transition is proportional to its priority raised to the power of alpha. We can use numpy's random.choice with probabilities to sample the batch indexes.
    def sample(self, batch_size, beta):
        only_priorities = self.priorities[:len(self.buffer)]
        scaled_priorities = only_priorities ** self.alpha
        sample_probabilities = scaled_priorities / scaled_priorities.sum()
        batch_indexes = np.random.choice(len(self.buffer), batch_size, p=sample_probabilities)

        batch = [self.buffer[i] for i in batch_indexes]
        state, action, reward, next_state, done= zip(*batch)
        weights = (len(self.buffer) * sample_probabilities[batch_indexes]) ** (-beta) # we are using beta here to correct the bias introduced by prioritized sampling. The importance-sampling weight for each sampled transition is calculated as (N * P(i))^(-beta), where N is the total number of transitions in the buffer and P(i) is the probability of sampling transition i. This weight is then normalized by dividing by the maximum weight in the batch to ensure stability during training.
        weights /= weights.max()  # Normalize for stability
        weights = weights.reshape(-1, 1).astype(np.float32)  # Reshape for broadcasting

        return (
            torch.tensor(np.array(state), dtype=torch.float32),
            torch.tensor(action, dtype=torch.long).view(-1, 1),
            torch.tensor(reward, dtype=torch.float32).view(-1, 1),
            torch.tensor(np.array(next_state), dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32).view(-1, 1),
            torch.tensor(weights, dtype=torch.float32),
            batch_indexes
        )
    def update_priorities(self, batch_indexes, priorities):

        priorities = np.abs(priorities) + self.eps
        self.priorities[batch_indexes] = priorities
    def __len__(self):
        return len(self.buffer)

# helps with the decay of the hyperparameters such as epsilon and beta. It takes the initial value, final value, total duration (in steps), and the current step as input and returns the linearly interpolated value based on the current step.
def linear_schedule(start, end, duration, step):
    ratio = min(step / duration, 1.0)
    return start + ratio * (end - start)

# This function creates and initializes the environment. It takes the environment ID, an optional seed for reproducibility, and an optional render mode. The environment is created using gym.make, and if a seed is provided, it resets the environment with that seed and also seeds the action space to ensure consistent behavior across runs.
def make_env(env_id, seed=None, render_mode=None):
    env = gym.make(env_id, render_mode=render_mode)

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    return env


def save_reward_plot(csv_path, output_path, title):
    episodes = []
    rewards = []
    moving_avgs = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
            moving_avgs.append(float(row["moving_avg_reward"]))

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, alpha=0.35, label="Episode reward")
    plt.plot(episodes, moving_avgs, label="100-episode moving average")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def record_video(env_id, q_net, device, output_dir, seed=999, episodes=3):
    os.makedirs(output_dir, exist_ok=True)

    env = gym.make(env_id, render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=output_dir,
        episode_trigger=lambda ep: ep < episodes,
        name_prefix="trained_agent",
    )

    q_net.eval()

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False

        while not done:
            state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                action = q_net(state).argmax(dim=1).item()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    env.close()


def train(env_id, algo, total_steps, seed):

    cfg = Config(total_steps=total_steps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    use_double_dqn = algo in ["ddqn", "ddqn_per"]
    use_per = algo in ["dqn_per", "ddqn_per"]
    
    run_name = f"{algo}_seed{seed}_{env_id}"
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(env_id, seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = QNetwork(state_dim, action_dim).to(device)
    target_q_net = QNetwork(state_dim, action_dim).to(device)
    target_q_net.load_state_dict(q_net.state_dict())

    optimizer = torch.optim.Adam(q_net.parameters(), lr=cfg.lr)

    if use_per:
        replay_buffer = PrioritizedReplayBuffer(cfg.buffer_size, alpha=cfg.per_alpha, eps=cfg.per_eps)
    else:
        replay_buffer = ReplayBuffer(cfg.buffer_size)
    writer = SummaryWriter(run_dir)

    csv_path = os.path.join(run_dir, "training_log.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        [
            "episode",
            "step",
            "reward",
            "moving_avg_reward",
            "epsilon",
            "loss",
            "mean_q",
            "mean_td_error",
        ]
    )
    obs, _ = env.reset(seed=seed)
    episode_reward = 0.0
    episode = 0
    episode_rewards = []

    last_loss = 0.0
    last_mean_q = 0.0
    last_mean_td_error = 0.0

    progress = tqdm(range(1, cfg.total_steps + 1), desc=f"{env_id} | {algo}")

    #starting of the training process
    for step in progress:
        epsilon = linear_schedule(
            cfg.epsilon_start,
            cfg.epsilon_end,
            cfg.epsilon_decay_steps,
            step,
        )
        # epsilon-greedy
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = q_net(state).argmax(dim=1).item() #action without exploration

        next_obs, reward, terminated, truncated, info = env.step(action) #next state, reward, and done signal from the environment after taking the action
        done = terminated or truncated #done signal indicates whether the episode has ended, either due to termination or truncation

        replay_buffer.push(obs, action, reward, next_obs, done) #store the transition in the replay buffer for later sampling during training

        obs = next_obs #update the current observation to the next observation for the next step in the loop
        episode_reward += reward #accumulate the reward for the current episode

        if len(replay_buffer) >= cfg.learning_starts and step % cfg.train_freq == 0: # check if we went over enough learning steps to start sampling from the replay buffer and if it's time to train based on the specified training frequency
            beta = linear_schedule(
                cfg.per_beta_start,
                cfg.per_beta_end,
                cfg.total_steps,
                step,
            ) #new beta value

            (
                states,
                actions,
                rewards,
                next_states,
                dones,
                weights,
                indices,
            ) = replay_buffer.sample(cfg.batch_size, beta=beta) #sample a batch of transitions from the replay buffer, including the states, actions, rewards, next states, done signals, importance-sampling weights (if using PER), and the indices of the sampled transitions for later priority updates

            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dones = dones.to(device)
            weights = weights.to(device)
            
            actions = actions.long().view(-1, 1)
            rewards = rewards.view(-1, 1)
            dones = dones.view(-1, 1)
            weights = weights.view(-1, 1)

            if states.dim() > 2:
                states = states.view(states.size(0), -1)

            if next_states.dim() > 2:
                next_states = next_states.view(next_states.size(0), -1)

            current_q = q_net(states).gather(1, actions) #compute the current Q-values for the sampled states and actions using the Q-network. The gather function is used to select the Q-values corresponding to the taken actions from the output of the Q-network.

            with torch.no_grad():
                if use_double_dqn:
                    next_actions = q_net(next_states).argmax(dim=1, keepdim=True) # we use the online q network to select the action with the highest Q-value for the next states, and then we use the target network to evaluate the Q-value of those selected actions. This helps to reduce overestimation bias in Q-learning.
                    next_q = target_q_net(next_states).gather(1, next_actions) #obsertved q value for the next states and the selected actions using the target network. This is the Double DQN update step, where we use the online network to select the action and the target network to evaluate it, which helps to mitigate overestimation bias in Q-learning.
                else:
                    next_q = target_q_net(next_states).max(dim=1, keepdim=True).values # we simply take the maximum Q-value across all actions for the next states using the target network, which is the standard DQN update step. This can lead to overestimation bias, which is why Double DQN was introduced as an improvement.

                target_q = rewards + cfg.gamma * (1.0 - dones) * next_q #compute the target Q-values using the Bellman equation. The target Q-value is calculated as the reward plus the discounted maximum Q-value of the next state, adjusted for terminal states using the done signal.

            td_error = target_q - current_q #compute the temporal difference (TD) error, which is the difference between the target Q-values and the current Q-values. This TD error is used to update the priorities in the replay buffer if using PER, and it also serves as the basis for the loss calculation during training.

            per_sample_loss = F.smooth_l1_loss(
                current_q,
                target_q,
                reduction="none",
            ) #compute the per-sample loss using the Huber loss (smooth L1 loss) between the current Q-values and the target Q-values. The reduction is set to "none" to keep the loss for each individual sample, which is necessary for applying the importance-sampling weights when using PER.

            loss = (weights * per_sample_loss).mean() #compute the final loss by multiplying the per-sample loss with the importance-sampling weights (if using PER) and taking the mean across the batch. This weighted loss helps to correct for the bias introduced by prioritized sampling, giving more weight to samples with higher TD errors.

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
            optimizer.step()

            replay_buffer.update_priorities(
                indices,
                td_error.detach().abs().cpu().numpy().flatten(),
            ) #update the priorities based on the td error that has been seen after taking the actions

            last_loss = loss.item()
            last_mean_q = current_q.detach().mean().item()
            last_mean_td_error = td_error.detach().abs().mean().item()

            writer.add_scalar("train/loss", last_loss, step)
            writer.add_scalar("train/mean_q", last_mean_q, step)
            writer.add_scalar("train/mean_td_error", last_mean_td_error, step)
            writer.add_scalar("train/epsilon", epsilon, step)

        if step % cfg.target_update_freq == 0:
            target_q_net.load_state_dict(q_net.state_dict()) #update the target network with the weights of the online Q-network at regular intervals defined by target_update_freq. This helps to stabilize training by providing a fixed target for a certain number of steps before updating it again.

        if done:
            episode += 1
            episode_rewards.append(episode_reward)
            moving_avg = float(np.mean(episode_rewards[-100:]))

            writer.add_scalar("episode/reward", episode_reward, episode)
            writer.add_scalar("episode/moving_avg_reward", moving_avg, episode)

            csv_writer.writerow(
                [
                    episode,
                    step,
                    episode_reward,
                    moving_avg,
                    epsilon,
                    last_loss,
                    last_mean_q,
                    last_mean_td_error,
                ]
            )
            csv_file.flush()

            progress.set_postfix(
                {
                    "ep": episode,
                    "reward": round(episode_reward, 1),
                    "avg100": round(moving_avg, 1),
                    "eps": round(epsilon, 3),
                }
            )

            obs, _ = env.reset()
            episode_reward = 0.0

    env.close()
    csv_file.close()
    writer.close()

    model_path = os.path.join(run_dir, "q_network.pt")
    torch.save(q_net.state_dict(), model_path)

    plot_path = os.path.join(run_dir, "reward_curve.png")
    save_reward_plot(
        csv_path,
        plot_path,
        title=f"{algo} on {env_id}",
    )

    video_dir = os.path.join(run_dir, "videos")
    record_video(env_id, q_net, device, video_dir, seed=seed + 999, episodes=3)

    print()
    print(f"Done: {env_id} | {algo}")
    print(f"Run directory: {run_dir}")
    print(f"Model: {model_path}")
    print(f"CSV: {csv_path}")
    print(f"Plot: {plot_path}")
    print(f"Videos: {video_dir}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env",
        type=str,
        default="LunarLander-v3",
        choices=[
            "CartPole-v1",
            "MountainCar-v0",
            "Acrobot-v1",
            "LunarLander-v3",
        ],
    )

    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=[
            "dqn",
            "ddqn",
            "dqn_per",
            "ddqn_per",
        ],
    )

    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    train(
        env_id=args.env,
        algo=args.algo,
        total_steps=args.steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()



    