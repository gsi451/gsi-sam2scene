# CLIP 기반 세그먼트 자동 라벨링 가이드

이 문서는 SAM(Segment Anything Model)으로 분리된 이미지 세그먼트에 CLIP(Contrastive Language-Image Pre-training)을 사용하여 자동으로 라벨을 지정하는 방법을 설명합니다.

## 1. 개요

이 파이프라인은 두 가지 주요 단계로 구성됩니다:
1. SAM을 사용하여 이미지에서 객체를 분리(세그멘테이션)
2. CLIP을 사용하여 각 세그먼트에 적절한 라벨 지정

## 2. 필요한 패키지 설치

```bash
# SAM 관련 패키지 (이미 설치되어 있다면 건너뛰세요)
pip install segment-anything

# CLIP 설치
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

## 3. 사용 방법

### 3.1 기본 사용법

```bash
# script_path: 스크립트 경로
# image_path: 처리할 이미지 경로
# checkpoint_path: SAM 모델 체크포인트 경로
python segment_labeling.py --image path/to/image.jpg --checkpoint path/to/sam_checkpoint.pth
```

### 3.2 사용자 정의 라벨

스크립트 내의 `CANDIDATE_LABELS` 리스트를 수정하여 특정 도메인에 맞는 라벨을 추가할 수 있습니다. 예를 들어:

```python
# 도시 거리 장면에 특화된 라벨
CANDIDATE_LABELS = [
    "car", "bus", "truck", "motorcycle", "bicycle", 
    "pedestrian", "traffic light", "street sign", "road", 
    "sidewalk", "building", "tree", "fence", "pole", 
    "fire hydrant", "bench", "trash can", "pothole",
    "manhole cover", "construction site"
]
```

## 4. 주요 기능

### 4.1 자동 세그멘테이션

SAM의 `SamAutomaticMaskGenerator`를 사용하여 이미지 내의 모든 객체를 자동으로 세그먼트합니다. 이 과정에서 다음 매개변수를 조정할 수 있습니다:

- `points_per_side`: 높을수록 더 많은 세그먼트 생성 (기본값: 32)
- `pred_iou_thresh`: 예측된 IoU 임계값 (기본값: 0.86)
- `stability_score_thresh`: 안정성 점수 임계값 (기본값: 0.92)
- `min_mask_region_area`: 최소 마스크 영역 크기 (기본값: 100)

### 4.2 CLIP 기반 라벨링

OpenAI의 CLIP 모델을 사용하여 세그먼트된 각 객체와 후보 라벨 간의 유사도를 계산합니다:

1. 각 세그먼트를 개별 이미지로 추출
2. CLIP 모델을 통해 이미지 특성 추출
3. 각 후보 라벨("a photo of a [label]" 형식)에 대한 텍스트 특성 추출
4. 이미지-텍스트 유사도 계산
5. 가장 높은 유사도를 가진 라벨 선택

## 5. 출력 결과

스크립트는 다음과 같은 출력을 생성합니다:

1. **개별 세그먼트 이미지**: `[OUTPUT_DIR]/[label]_[id]_[confidence].png` 형식으로 각 세그먼트 저장
2. **시각화된 결과 이미지**: 원본 이미지 위에 세그먼트 윤곽선과 라벨이 표시된 이미지
3. **라벨별 통계**: 각 라벨의 발생 횟수, 총 면적, 평균 면적 등을 포함한 통계 정보

## 6. 고급 활용 예시

### 6.1 사용자 정의 후처리

세그먼트 결과를 특정 용도에 맞게 후처리할 수 있습니다:

```python
# 라벨별로 분류하여 폴더에 저장
for segment in labeled_segments:
    label = segment["label"]
    os.makedirs(f"sorted/{label}", exist_ok=True)
    # 이미지 복사 또는 저장 코드...
```

### 6.2 특정 객체만 추출

특정 라벨의 객체만 추출할 수 있습니다:

```python
# 특정 라벨(예: 'car')만 추출
car_segments = [s for s in labeled_segments if s["label"] == "car"]
```

## 7. 문제 해결

- **세그먼트가 너무 많거나 적음**: `points_per_side`, `pred_iou_thresh`, `stability_score_thresh` 매개변수 조정
- **라벨링 정확도 향상**: `CANDIDATE_LABELS`에 더 구체적인 라벨 추가
- **메모리 오류**: 이미지 크기를 줄이거나 작은 SAM 모델 사용(vit_b)
- **속도 개선**: GPU 사용 또는 모델 양자화 고려

## 8. 참고 자료

- [Segment Anything 공식 저장소](https://github.com/facebookresearch/segment-anything)
- [CLIP 공식 저장소](https://github.com/openai/CLIP)
- [SAM 논문](https://arxiv.org/abs/2304.02643)
- [CLIP 논문](https://arxiv.org/abs/2103.00020)