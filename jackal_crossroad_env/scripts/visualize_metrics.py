#!/usr/bin/env python3
"""
Generate metric visualizations (plots and tables) from SAC training CSV logs.

Expected input files (created by train_sac_agent.py):
- training_metrics.csv
- episode_metrics.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    raise RuntimeError("matplotlib is required for visualization") from exc


def _read_csv(path: Path):
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(row)
    return rows


def _to_float(value, default=np.nan):
    if value is None:
        return default
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _save_summary_table(output_dir: Path, episode_rows, training_rows):
    rewards = [_to_float(row.get("episode/reward"), default=np.nan) for row in episode_rows]
    rewards = [x for x in rewards if not np.isnan(x)]

    goal_rate_100 = [_to_float(row.get("episode/goal_reach_rate_100"), default=np.nan) for row in episode_rows]
    goal_rate_100 = [x for x in goal_rate_100 if not np.isnan(x)]

    illegal_rate_100 = [_to_float(row.get("episode/illegal_crossing_rate_100"), default=np.nan) for row in episode_rows]
    illegal_rate_100 = [x for x in illegal_rate_100 if not np.isnan(x)]

    actor_loss = np.nan
    critic_loss = np.nan
    for row in reversed(training_rows):
        if np.isnan(actor_loss):
            actor_loss = _to_float(row.get("train/actor_loss"), default=np.nan)
        if np.isnan(critic_loss):
            critic_loss = _to_float(row.get("train/critic_loss"), default=np.nan)
        if not np.isnan(actor_loss) and not np.isnan(critic_loss):
            break

    summary_rows = [
        ("total_episodes", len(episode_rows)),
        ("avg_episode_reward", "n/a" if not rewards else round(float(np.mean(rewards)), 4)),
        ("goal_reach_rate_100", "n/a" if not goal_rate_100 else round(float(goal_rate_100[-1]), 4)),
        ("illegal_crossing_rate_100", "n/a" if not illegal_rate_100 else round(float(illegal_rate_100[-1]), 4)),
        ("latest_actor_loss", "n/a" if np.isnan(actor_loss) else round(float(actor_loss), 6)),
        ("latest_critic_loss", "n/a" if np.isnan(critic_loss) else round(float(critic_loss), 6)),
    ]

    csv_path = output_dir / "metrics_summary_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        for key, value in summary_rows:
            writer.writerow([key, value])

    md_path = output_dir / "metrics_summary_table.md"
    with open(md_path, "w", encoding="utf-8") as file:
        file.write("| metric | value |\n")
        file.write("|---|---|\n")
        for key, value in summary_rows:
            file.write(f"| {key} | {value} |\n")


def _plot_episode_metrics(output_dir: Path, episode_rows):
    if not episode_rows:
        return

    timesteps = [_to_float(row.get("train/timestep"), default=float(i + 1)) for i, row in enumerate(episode_rows)]
    rewards = [_to_float(row.get("episode/reward"), default=np.nan) for row in episode_rows]
    goal_rate = [_to_float(row.get("episode/goal_reach_rate_100"), default=np.nan) for row in episode_rows]
    illegal_rate = [_to_float(row.get("episode/illegal_crossing_rate_100"), default=np.nan) for row in episode_rows]

    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, rewards, label="episode_reward", alpha=0.75)
    plt.xlabel("timestep")
    plt.ylabel("reward")
    plt.title("Episode Reward over Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "episode_reward_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, goal_rate, label="goal_reach_rate_100")
    plt.plot(timesteps, illegal_rate, label="illegal_crossing_rate_100")
    plt.xlabel("timestep")
    plt.ylabel("rate")
    plt.title("Safety and Success Rates (100-episode window)")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "episode_rate_curves.png", dpi=150)
    plt.close()


def _plot_training_metrics(output_dir: Path, training_rows):
    if not training_rows:
        return

    timesteps = [_to_float(row.get("train/timestep"), default=float(i + 1)) for i, row in enumerate(training_rows)]
    actor_loss = [_to_float(row.get("train/actor_loss"), default=np.nan) for row in training_rows]
    critic_loss = [_to_float(row.get("train/critic_loss"), default=np.nan) for row in training_rows]

    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, actor_loss, label="actor_loss")
    plt.plot(timesteps, critic_loss, label="critic_loss")
    plt.xlabel("timestep")
    plt.ylabel("loss")
    plt.title("SAC Loss Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curves.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate plots and tables from SAC metric CSVs")
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default="./sac_logs/visualizations",
        help="Directory containing training_metrics.csv and episode_metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for generated plots/tables (default: same as metrics-dir)",
    )
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    output_dir = Path(args.output_dir) if args.output_dir else metrics_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    training_csv = metrics_dir / "training_metrics.csv"
    episode_csv = metrics_dir / "episode_metrics.csv"

    training_rows = _read_csv(training_csv)
    episode_rows = _read_csv(episode_csv)

    if not training_rows and not episode_rows:
        raise FileNotFoundError(
            f"No metric CSV files found in {metrics_dir}. Expected training_metrics.csv and/or episode_metrics.csv"
        )

    _save_summary_table(output_dir, episode_rows, training_rows)
    _plot_episode_metrics(output_dir, episode_rows)
    _plot_training_metrics(output_dir, training_rows)

    print(f"Metric visualization generated in: {output_dir}")


if __name__ == "__main__":
    main()
