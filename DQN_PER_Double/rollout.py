import argparse
import csv
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium.wrappers import RecordVideo


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        # Must match the exact architecture used in train.py
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def safe_load_state_dict(model_path, device):
    """
    Loads a state_dict saved using:
        torch.save(q_net.state_dict(), model_path)

    The try/except keeps this compatible across PyTorch versions.
    """
    try:
        return torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(model_path, map_location=device)


def run_rollout(env_id, model_path, video_dir, episodes, video_episodes, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(video_dir, exist_ok=True)

    env = gym.make(env_id, render_mode="rgb_array")

    # Record only the first few episodes to avoid creating too many videos.
    env = RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda episode_id: episode_id < video_episodes,
        name_prefix="policy_rollout",
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = QNetwork(state_dim, action_dim).to(device)

    state_dict = safe_load_state_dict(model_path, device)
    q_net.load_state_dict(state_dict)
    q_net.eval()

    rewards = []
    steps_list = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)

        done = False
        episode_reward = 0.0
        steps = 0

        while not done:
            state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            # Extra safety in case an environment gives a weird shape
            if state.dim() > 2:
                state = state.view(state.size(0), -1)

            with torch.no_grad():
                q_values = q_net(state)
                action = q_values.argmax(dim=1).item()

            obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            episode_reward += reward
            steps += 1

        rewards.append(episode_reward)
        steps_list.append(steps)

        print(
            f"Episode {ep + 1:03d}: "
            f"reward={episode_reward:.2f}, "
            f"steps={steps}"
        )

    env.close()

    rewards_np = np.array(rewards, dtype=np.float32)
    steps_np = np.array(steps_list, dtype=np.float32)

    avg_reward = float(np.mean(rewards_np))
    median_reward = float(np.median(rewards_np))
    best_reward = float(np.max(rewards_np))
    worst_reward = float(np.min(rewards_np))
    std_reward = float(np.std(rewards_np))
    avg_steps = float(np.mean(steps_np))
    success_rate_200 = float(np.mean(rewards_np >= 200.0) * 100.0)

    rewards_csv_path = os.path.join(video_dir, "rollout_rewards.csv")
    summary_csv_path = os.path.join(video_dir, "rollout_summary.csv")

    with open(rewards_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "steps", "success_ge_200"])

        for i, (reward, steps) in enumerate(zip(rewards, steps_list), start=1):
            writer.writerow([i, reward, steps, int(reward >= 200.0)])

    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model_path",
                "env_id",
                "episodes",
                "video_episodes",
                "seed",
                "avg_reward",
                "median_reward",
                "best_reward",
                "worst_reward",
                "std_reward",
                "avg_steps",
                "success_rate_ge_200",
            ]
        )
        writer.writerow(
            [
                model_path,
                env_id,
                episodes,
                video_episodes,
                seed,
                avg_reward,
                median_reward,
                best_reward,
                worst_reward,
                std_reward,
                avg_steps,
                success_rate_200,
            ]
        )

    print()
    print("Rollout summary")
    print("----------------")
    print(f"Model: {model_path}")
    print(f"Environment: {env_id}")
    print(f"Episodes: {episodes}")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Median reward: {median_reward:.2f}")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Worst reward: {worst_reward:.2f}")
    print(f"Reward std: {std_reward:.2f}")
    print(f"Average steps: {avg_steps:.2f}")
    print(f"Success rate >= 200: {success_rate_200:.1f}%")
    print(f"Videos saved to: {video_dir}")
    print(f"Rewards CSV saved to: {rewards_csv_path}")
    print(f"Summary CSV saved to: {summary_csv_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="LunarLander-v3")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--video-dir", type=str, default="rollouts")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--video-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=100)

    args = parser.parse_args()

    run_rollout(
        env_id=args.env,
        model_path=args.model,
        video_dir=args.video_dir,
        episodes=args.episodes,
        video_episodes=args.video_episodes,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()