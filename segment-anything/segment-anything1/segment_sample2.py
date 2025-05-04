import numpy as np
import torch  # 이 라인 추가
import matplotlib.pyplot as plt
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# 이미지 로드
image_path = "../sample3.png"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 모델 로드
sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"
#device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# 자동 마스크 생성기 설정
mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    min_mask_region_area=100,
)

# 자동으로 모든 마스크 생성
masks = mask_generator.generate(image)

print(f"생성된 마스크 수: {len(masks)}")

# 각 마스크를 개별 색상으로 표시하는 함수
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
    
    ax.imshow(img)

# 마스크 시각화
plt.figure(figsize=(10, 10))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.savefig('all_segments.png')
plt.show()

# 각 마스크 개별적으로 저장하기
for i, mask_data in enumerate(masks):
    # 마스크 이미지 생성
    mask = mask_data['segmentation']
    masked_img = image.copy()
    # 마스크 밖은 검은색으로
    masked_img[~mask] = [0, 0, 0]
    # 이미지 저장
    cv2.imwrite(f'segment_{i}.jpg', cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))