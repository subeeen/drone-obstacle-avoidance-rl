#!/bin/bash
# ──────────────────────────────────────────────────────────
# 설치 및 실행 가이드
# 실행 전: conda 또는 venv 가상환경 활성화 권장
# ──────────────────────────────────────────────────────────

set -e  # 오류 발생 시 중단

echo "=== 1. gym-pybullet-drones 설치 ==="
pip install pybullet gymnasium stable-baselines3[extra] matplotlib

# gym-pybullet-drones는 pip 패키지 없이 소스에서 설치
git clone https://github.com/utiasDSL/gym-pybullet-drones.git
cd gym-pybullet-drones
pip install -e .
cd ..

echo ""
echo "=== 2. 설치 확인 ==="
python -c "
import gym_pybullet_drones
import stable_baselines3
print('gym-pybullet-drones:', gym_pybullet_drones.__version__)
print('stable-baselines3:',   stable_baselines3.__version__)
print('설치 완료!')
"

echo ""
echo "=== 3. 학습 시작 (100만 스텝, 병렬 4개 환경) ==="
echo "    중단하려면 Ctrl+C (중간 체크포인트는 logs/ 에 저장됩니다)"
python train.py --timesteps 1000000 --n-envs 4

echo ""
echo "=== 4. 평가 (GUI 시각화) ==="
# 학습 완료 후 가장 최근 로그 디렉터리의 best_model 자동 탐색
BEST=$(ls -td logs/ObstacleAvoidance_*/best_model.zip 2>/dev/null | head -1)
if [ -z "$BEST" ]; then
    echo "[ERROR] best_model.zip 을 찾을 수 없습니다. logs/ 디렉터리를 확인하세요."
else
    echo "    모델: $BEST"
    python evaluate.py --model "$BEST" --episodes 10 --gui
fi
