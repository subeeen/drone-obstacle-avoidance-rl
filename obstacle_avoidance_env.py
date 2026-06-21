"""
장애물 회피 드론 강화학습 환경 (gym-pybullet-drones v2.x 호환)

시나리오:
  시작: [0, 0, 0.5]  →  목표: [3, 0, 1.0]
  장애물 3개 (빨간 원기둥, 경로 위에 배치)

관측 (flat 1D):
  12  운동학 (위치, RPY, 속도, 각속도)
  + ACTION_BUFFER_SIZE*4  행동 버퍼 (BaseRLAviary 기본 포함)
  + 3  목표 방향 단위 벡터
  + 3  장애물까지 정규화 거리
  = 12 + buf + 6  차원

행동: VEL (vx, vy, vz, yaw_rate), shape (1, 4)
"""
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


# ── 시나리오 상수 ──────────────────────────────────────
GOAL_POS        = np.array([3.0, 0.0, 1.0])
START_POS       = np.array([0.0, 0.0, 0.5])
GOAL_RADIUS     = 0.5

# (x, y, z_center, radius)
OBSTACLE_CONFIGS = [
    (1.0,  0.0, 0.5, 0.15),
    (2.0,  0.3, 0.7, 0.12),
    (2.5, -0.3, 0.4, 0.12),
]

MAX_XY          = 4.5
MAX_Z           = 2.5


class ObstacleAvoidanceAviary(BaseRLAviary):
    """
    단일 드론이 장애물을 피해 목표 지점에 도달하는 RL 환경.
    gym-pybullet-drones v2.x (gymnasium API) 기반.
    """

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui: bool = False,
                 record: bool = False):

        self.EPISODE_LEN_SEC = 20
        self._obstacle_ids   = []
        self._prev_dist      = None

        super().__init__(
            drone_model=drone_model,
            num_drones=1,
            initial_xyzs=np.array([START_POS]),
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=ObservationType.KIN,
            act=ActionType.VEL,
        )

    # ── 장애물 생성 ────────────────────────────────────
    def _addObstacles(self):
        self._obstacle_ids = []
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self.CLIENT
        )

        for (ox, oy, oz, r) in OBSTACLE_CONFIGS:
            height = oz * 2
            col = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=r, height=height,
                physicsClientId=self.CLIENT,
            )
            vis = p.createVisualShape(
                p.GEOM_CYLINDER, radius=r, length=height,
                rgbaColor=[0.85, 0.2, 0.2, 0.85],
                physicsClientId=self.CLIENT,
            )
            bid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[ox, oy, oz],
                physicsClientId=self.CLIENT,
            )
            self._obstacle_ids.append(bid)

        # 목표 지점 시각화 (초록 구체, 충돌 없음)
        vis_goal = p.createVisualShape(
            p.GEOM_SPHERE, radius=GOAL_RADIUS,
            rgbaColor=[0.2, 0.9, 0.2, 0.5],
            physicsClientId=self.CLIENT,
        )
        p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis_goal,
            basePosition=GOAL_POS.tolist(),
            physicsClientId=self.CLIENT,
        )

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        state = self._getDroneStateVector(0)
        self._prev_dist = float(np.linalg.norm(GOAL_POS - state[0:3]))
        return obs, info

    # ── 관측 공간: base(12+buf) + 6 ───────────────────
    def _observationSpace(self):
        base_space = super()._observationSpace()          # shape (1, 12+buf)
        base_dim   = base_space.shape[1]
        extra_dim  = 6                                    # 목표 벡터 3 + 장애물 거리 3
        total_dim  = base_dim + extra_dim

        low  = np.full((1, total_dim), -np.inf, dtype=np.float32)
        high = np.full((1, total_dim),  np.inf, dtype=np.float32)
        # 거리(양수) 구간 제한
        low[0, base_dim + 3:] = 0.0
        high[0, base_dim + 3:] = 1.0

        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _computeObs(self):
        base_obs = super()._computeObs()                  # shape (1, base_dim)
        state    = self._getDroneStateVector(0)
        pos      = state[0:3]

        # 목표 방향 단위 벡터
        to_goal = GOAL_POS - pos
        dist    = np.linalg.norm(to_goal)
        dir_vec = (to_goal / (dist + 1e-6)).astype(np.float32)

        # 장애물까지 정규화 거리 (0~1)
        obs_dists = np.array([
            float(np.clip(np.linalg.norm(pos - np.array([ox, oy, oz])) / MAX_XY, 0.0, 1.0))
            for ox, oy, oz, _ in OBSTACLE_CONFIGS
        ], dtype=np.float32)

        extra = np.hstack([dir_vec, obs_dists]).reshape(1, 6)
        return np.hstack([base_obs, extra]).astype(np.float32)

    # ── 보상 ─────────────────────────────────────────
    def _computeReward(self):
        state = self._getDroneStateVector(0)
        pos   = state[0:3]
        dist  = float(np.linalg.norm(GOAL_POS - pos))

        # 1) 진행 보상
        progress = (self._prev_dist - dist) * 10.0
        if self._prev_dist is not None:
            self._prev_dist = dist

        # 2) 도착 보너스
        arrival = 200.0 if dist < GOAL_RADIUS else 0.0

        # 3) 충돌 패널티
        col_pen = -50.0 if self._isCollision() else 0.0

        # 4) 경계 이탈 패널티
        bounds_pen = -20.0 if self._isOutOfBounds(pos) else 0.0

        # 5) 시간 패널티
        time_pen = -0.05

        # 6) 장애물 근접 소프트 패널티
        soft_pen = 0.0
        for ox, oy, oz, r in OBSTACLE_CONFIGS:
            d = float(np.linalg.norm(pos - np.array([ox, oy, oz])))
            margin = r + 0.3
            if d < margin:
                soft_pen -= (margin - d) * 2.0

        return float(progress + arrival + col_pen + bounds_pen + time_pen + soft_pen)

    # ── 종료: gymnasium은 terminated / truncated 분리 ──
    def _computeTerminated(self):
        """목표 도달 또는 충돌 → terminated (에피소드 실패/성공으로 끝)."""
        state = self._getDroneStateVector(0)
        pos   = state[0:3]
        if np.linalg.norm(GOAL_POS - pos) < GOAL_RADIUS:
            return True
        if self._isCollision():
            return True
        return False

    def _computeTruncated(self):
        """시간 초과 또는 경계 이탈 → truncated (에피소드 잘림)."""
        state = self._getDroneStateVector(0)
        pos   = state[0:3]
        if self._isOutOfBounds(pos):
            return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    def _computeInfo(self):
        state = self._getDroneStateVector(0)
        return {
            "dist_to_goal": float(np.linalg.norm(GOAL_POS - state[0:3])),
            "collision":    self._isCollision(),
        }

    # ── 헬퍼 ─────────────────────────────────────────
    def _isCollision(self) -> bool:
        drone_id = self.DRONE_IDS[0]
        for obs_id in self._obstacle_ids:
            if p.getContactPoints(bodyA=drone_id, bodyB=obs_id,
                                  physicsClientId=self.CLIENT):
                return True
        state = self._getDroneStateVector(0)
        if state[2] < 0.05:
            return True
        return False

    def _isOutOfBounds(self, pos) -> bool:
        return (abs(pos[0]) > MAX_XY or abs(pos[1]) > MAX_XY
                or pos[2] > MAX_Z or pos[2] < 0.0)
