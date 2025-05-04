import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

# 이미지 경로 (자신의 이미지 경로로 변경하세요)
image_path = "../sample2.jpg"  # 샘플 이미지 경로

# 이미지 로드
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 모델 로드 (자신의 체크포인트 경로로 변경하세요)
sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"  # vit_h, vit_l, vit_b 중 선택

# M1 맥에서는 MPS 사용
device = "mps" if torch.backends.mps.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
predictor.set_image(image)

# 점 프롬프트 방식
# 이미지 중앙에 점 하나 찍기
h, w = image.shape[:2]
input_point = np.array([[w//2, h//2]])
input_label = np.array([1])  # 1: 포함, 0: 제외

# 마스크 예측
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True  # 여러 마스크 후보 생성
)

# 결과 시각화
plt.figure(figsize=(10, 10))
plt.imshow(image)
show_mask(masks[0], plt.gca())
show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.savefig('segmentation_result.png')
plt.show()

print(f"마스크 갯수: {len(masks)}")
print(f"가장 높은 점수의 마스크 점수: {scores[np.argmax(scores)]:.3f}")

# 마스크 저장
best_mask = masks[np.argmax(scores)]
cv2.imwrite('mask.png', best_mask * 255)

# 마스크 적용 이미지 생성
masked_image = image.copy()
masked_image[~best_mask] = [0, 0, 0]  # 배경을 검은색으로
cv2.imwrite('masked_result.jpg', cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))