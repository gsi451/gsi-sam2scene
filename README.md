# gsi-sam2scene
gsi-street2sim은 이미지 세그멘테이션(Segment Anything 2.1)과 3D Gaussian Splatting(DreamGaussian)을 결합하여, 거리(Street) 사진을 Unity/Unreal Engine에서 바로 사용 가능한 3D 씬으로 변환하는 오픈소스 프로젝트입니다. 강화학습 연구자가 실제 도심 환경을 손쉽게 시뮬레이터에 반영할 수 있도록 지원합니다.

# 이미지 기반 3D 환경 생성 기술 현황 및 프로젝트 가이드

## 1. 핵심 기술 현황 요약

### 1.1 이미지 세그먼트 분리 (Instance / Semantic Segmentation)

- **Segment Anything Model (SAM)**: Meta AI에서 개발한 SAM 시리즈는 2023년 초기 모델부터 2025년 2월 SAM 2.1까지 발전하며 "클릭 한 번으로 객체 분리" 목표를 달성했습니다. 11M 이미지와 11억 마스크로 학습되어 처음 보는 객체도 안정적으로 분리합니다.

- **SEEM ("Segment Everything Everywhere All at Once")**: CVPR 2024에서 발표된 SEEM과 OneFormer 2024는 단일 트랜스포머로 인스턴스/파노픽/비디오 세그먼트를 통합 처리하여 프롬프트 없는 실시간 분할을 가능하게 합니다.

- **현재 상태**: 정지 화면의 객체 분리는 이미 실시간 수준으로 구현 가능하며, 강화학습 파이프라인에 바로 통합할 수 있을 정도로 성숙한 기술입니다.

### 1.2 분리된 객체의 3D 모델 생성

- **OpenAI Shap-E (2023)**: 텍스트나 이미지를 입력받아 3D implicit function을 생성하고 메시(mesh) 변환까지 자동으로 처리합니다.

- **Gaussian Splatting 기반 기술 (2024~현재)**: 
  - **DreamGaussian/GaussianDreamer**: 2D diffusion과 3D Gaussian splat representation을 결합하여 몇 분 내에 포토리얼리스틱 3D 모델을 생성합니다.
  - **2025년 초 연구**: 단일 이미지에서 구조화된 3DGS로 변환하는 파이프라인을 제안하고 있습니다.

- **한계점**: 큰 건물 전체를 완벽한 CAD 수준 메시로 재현하기는 아직 어렵습니다. 그러나 차량, 가로수, 표지판과 같은 작은 물체는 DreamGaussian 계열 기술로 AR/게임용 중간 디테일 수준의 모델을 확보할 수 있습니다.

## 2. 기술 단계별 가능/제한 사항

| 단계 | 이미 이용 가능한 수준 | 주요 한계 |
|------|----------------------|-----------|
| ① 객체 분리 | SAM 2.1 · SEEM 등으로 1초 내 마스크 출력 | 드론/차량 영상처럼 모션 블러·고각도에서 정확도 저하 |
| ② 클래스 라벨링 | CLIP · BLIP-2 등으로 텍스트 라벨 자동추정 | 동일 카테고리 내 세부 종류(소나무 vs 벚나무) 구분은 미흡 |
| ③ 3D 모델 생성 | Shap-E, DreamGaussian으로 저~중간 디테일 메시 생성 | 큰 구조물·복잡 기하가 아직 부정확, 스케일 정합 필요 |
| ④ 실시간 파이프라인 | CUDA + RTX 3090 기준 단일 이미지 1-2분 | 다수 객체 대량 변환 시 연산량 급증 |

## 3. 프로젝트 아이디어

| 프로젝트 | 핵심 목표 | 기술 스택 제안 |
|---------|----------|--------------|
| **1. "거리 → 가상 놀이터" 변환 툴** | 휴대폰 사진 한 장을 Unity/Unreal 씬(.fbx/.usdz)으로 자동 변환 | SAM 2.1 + CLIP → DreamGaussian → Blender Python API → Unity import |
| **2. 강화학습 자율주행 시뮬레이터** | 실제 골목 사진들로 만든 3D 맵에서 PPO/TD3 에이전트 훈련 | PyTorch-RL, Gazebo Classic or Isaac Sim + 3DGS 에셋 |
| **3. "AR 안내판" 생성기** | 거리 표지판·건물 간판만 자동 분리 후 ARCore에 배치 | MobileSAM(SAM Lite) → Shap-E → ARCore/ARKit |
| **4. 디지털 트윈 ESG 지표 시각화** | 나무 개수·종류를 추정해 탄소저감량 계산, 3D 뷰어로 인터랙션 | SEEM + tree-specific classifier → Shap-E-Tree-Dataset → Three.js |

## 4. 시작 로드맵

### 4.1 데이터 수집
- 스마트폰으로 거리 사진 약 100장 확보

### 4.2 객체 분리 데모
```bash
pip install segment-anything==2.1 clip-by-openai
python demo_sam.py --input street.jpg --output masks/
```

### 4.3 라벨 자동추정
- 각 마스크 → CLIP image-text similarity 최대값으로 라벨링

### 4.4 3D 모델 생성
- **작은 물체**:
  ```bash
  dreamgaussian --img tree_mask.png --steps 800
  ```
- **큰 구조물**: multi-view 촬영 후 GaussianDreamer/NeRF 훈련 권장

### 4.5 씬 통합
- Blender Python 스크립트로 위치·스케일 정렬 → GLB export

### 4.6 강화학습
- Unity ML-Agents나 Isaac Gym에서 환경 로드 후 정책 학습

## 5. 결론 및 제언

- **객체 분할-라벨링**: 이미 "오픈소스 + GPU 한 장"으로 충분히 실용 단계에 도달했습니다.

- **단일 이미지 → 3D 변환**: Gaussian Splatting 기반 접근으로 소규모 물체는 실용적 결과물을 얻을 수 있습니다.

- **개발 전략**: 작은 범위(예: 교차로 하나)를 대상으로 end-to-end 파이프라인을 먼저 완성한 뒤, 멀티-뷰·LiDAR 융합으로 확장하면 연구와 서비스 두 목표를 효과적으로 달성할 수 있습니다.
