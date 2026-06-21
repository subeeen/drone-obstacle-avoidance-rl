"""
PPO로 장애물 회피 드론 학습 (gym-pybullet-drones v2.x / SB3 2.x / gymnasium)

사용법:
    python train.py
    python train.py --timesteps 500000
    python train.py --n-envs 4
"""
import argparse
import os
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecTransposeImage
from gymnasium.wrappers import FlattenObservation

from obstacle_avoidance_env import ObstacleAvoidanceAviary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs",    type=int, default=4)
    parser.add_argument("--resume",    type=str, default=None,
                        help="이어서 학습할 모델 .zip 경로")
    return parser.parse_args()


def make_env():
    """환경 팩토리 — observation을 1D flat으로 래핑."""
    def _make():
        env = ObstacleAvoidanceAviary(gui=False)
        env = FlattenObservation(env)   # (1, N) → (N,) 으로 flatten
        return env
    return _make


def main():
    args = parse_args()

    run_name = f"ObstacleAvoidance_{datetime.now().strftime('%m%d_%H%M')}"
    log_dir  = os.path.join("logs", run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] 로그: {log_dir}")

    # ── 학습 환경 ────────────────────────────────────
    train_env = make_vec_env(make_env(), n_envs=args.n_envs)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # ── 평가 환경 ────────────────────────────────────
    eval_env = make_vec_env(make_env(), n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            clip_obs=10.0, training=False)

    # ── 콜백 ────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000 // args.n_envs,
        save_path=log_dir,
        name_prefix="ckpt",
        verbose=1,
    )
    stop_cb  = StopTrainingOnRewardThreshold(reward_threshold=150.0, verbose=1)
    eval_cb  = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=20_000 // args.n_envs,
        n_eval_episodes=10,
        deterministic=True,
        callback_on_new_best=stop_cb,
        verbose=1,
    )

    # ── 모델 ────────────────────────────────────────
    if args.resume:
        print(f"[INFO] 이어서 학습: {args.resume}")
        model = PPO.load(args.resume, env=train_env, device="auto")
        # lr 제대로 변경 (optimizer까지 반영)
        new_lr = 5e-5
        model.learning_rate = new_lr
        model.lr_schedule = lambda _: new_lr
        for pg in model.policy.optimizer.param_groups:
            pg['lr'] = new_lr
        print(f"[INFO] learning_rate → {new_lr}")
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            learning_rate=3e-4,
            tensorboard_log=log_dir,
            policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
            verbose=1,
            device="auto",
        )

    # ── 학습 ────────────────────────────────────────
    print(f"[INFO] 학습 시작 — 총 {args.timesteps:,} 스텝")
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_cb, eval_cb],
        reset_num_timesteps=(args.resume is None),
    )

    # ── 저장 ────────────────────────────────────────
    model.save(os.path.join(log_dir, "final_model"))
    train_env.save(os.path.join(log_dir, "vec_normalize.pkl"))
    print(f"[INFO] 저장 완료: {log_dir}/final_model.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
