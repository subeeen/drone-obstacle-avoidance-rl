"""
학습된 모델 평가 및 시각화 (gym-pybullet-drones v2.x / gymnasium)

사용법:
    python evaluate.py --model logs/ObstacleAvoidance_0101_1230/best_model.zip
    python evaluate.py --model logs/.../best_model.zip --episodes 20 --gui
"""
import argparse
import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from gymnasium.wrappers import FlattenObservation

from obstacle_avoidance_env import ObstacleAvoidanceAviary, GOAL_POS, GOAL_RADIUS, START_POS, OBSTACLE_CONFIGS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--gui",      action="store_true")
    return parser.parse_args()


def make_flat_env(gui=False):
    def _make():
        env = ObstacleAvoidanceAviary(gui=gui)
        return FlattenObservation(env)
    return _make


def run_episode(env, model, gui=False):
    obs = env.reset()
    done       = False
    total_rew  = 0.0
    steps      = 0
    trajectory = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done_arr, info = env.step(action)
        done = bool(done_arr[0])

        total_rew += float(reward[0])
        steps     += 1

        try:
            state = env.envs[0].env._getDroneStateVector(0)
            trajectory.append(state[0:3].copy())
        except Exception:
            pass

        if gui:
            time.sleep(1 / 60)

    final_pos    = trajectory[-1] if trajectory else np.zeros(3)
    dist_to_goal = float(np.linalg.norm(GOAL_POS - final_pos))
    return {
        "total_reward": total_rew,
        "steps":        steps,
        "dist_to_goal": dist_to_goal,
        "success":      dist_to_goal < GOAL_RADIUS,
        "trajectory":   trajectory,
    }


def print_stats(results):
    rewards   = [r["total_reward"] for r in results]
    steps_arr = [r["steps"]        for r in results]
    dists     = [r["dist_to_goal"] for r in results]
    successes = [r["success"]      for r in results]
    print("\n" + "=" * 50)
    print(f"  에피소드 수:        {len(results)}")
    print(f"  성공률:            {np.mean(successes)*100:.1f}%")
    print(f"  평균 보상:          {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  평균 스텝:          {np.mean(steps_arr):.0f}")
    print(f"  목표까지 평균 거리:  {np.mean(dists):.3f} m")
    print("=" * 50)


def plot_trajectories(results, save_path="trajectories.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("[WARN] matplotlib 없음, 플롯 건너뜀")
        return

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    for ox, oy, oz, r in OBSTACLE_CONFIGS:
        theta = np.linspace(0, 2 * np.pi, 30)
        z_cyl = np.linspace(0, oz * 2, 10)
        T, Z  = np.meshgrid(theta, z_cyl)
        ax.plot_surface(ox + r * np.cos(T), oy + r * np.sin(T), Z,
                        alpha=0.3, color="red")

    for res in results:
        traj = np.array(res["trajectory"])
        if len(traj) == 0:
            continue
        c = "green" if res["success"] else "gray"
        a = 0.8 if res["success"] else 0.3
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=c, alpha=a, linewidth=1.5)

    ax.scatter(*START_POS, color="blue",  s=120, zorder=5, label="시작")
    ax.scatter(*GOAL_POS,  color="green", s=200, zorder=5, label="목표", marker="*")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title("드론 궤적 (초록=성공, 회색=실패)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[INFO] 궤적 저장: {save_path}")
    plt.close()


def main():
    args    = parse_args()
    log_dir = os.path.dirname(args.model)

    # ── 환경 ──────────────────────────────────────
    raw_env = DummyVecEnv([make_flat_env(gui=args.gui)])

    vec_norm_path = os.path.join(log_dir, "vec_normalize.pkl")
    if os.path.exists(vec_norm_path):
        env = VecNormalize.load(vec_norm_path, raw_env)
        env.training   = False
        env.norm_reward = False
        print(f"[INFO] VecNormalize 로드: {vec_norm_path}")
    else:
        env = raw_env

    # ── 모델 ──────────────────────────────────────
    model = PPO.load(args.model, env=env, device="auto")
    print(f"[INFO] 모델 로드: {args.model}")

    # ── 평가 ──────────────────────────────────────
    results = []
    for ep in range(args.episodes):
        r   = run_episode(env, model, gui=args.gui)
        results.append(r)
        tag = "✓" if r["success"] else "✗"
        print(f"  에피소드 {ep+1:3d} [{tag}]  "
              f"보상={r['total_reward']:7.2f}  "
              f"스텝={r['steps']:4d}  "
              f"목표={r['dist_to_goal']:.3f}m")

    print_stats(results)
    plot_trajectories(results, os.path.join(log_dir, "trajectories.png"))
    env.close()


if __name__ == "__main__":
    main()
