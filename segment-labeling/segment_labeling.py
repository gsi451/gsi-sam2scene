import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import clip
from typing import List, Dict

CANDIDATE_LABELS = [
    "car", "tree", "building", "person", "road", "sidewalk",
    "sign", "traffic light", "bicycle", "motorcycle", "bus",
    "truck", "bench", "fire hydrant", "street light", "fence",
    "wall", "grass", "sky", "mountain", "river", "lake", "flower",
    "bird", "cat", "dog", "trash can", "pole", "bridge"
]

IMAGE_PATH = "../segment-anything/sample3.png"
CHECKPOINT_PATH = "../segment-anything/segment-anything1/checkpoints/sam_vit_h_4b8939.pth"
OUTPUT_DIR = "./segmented_objects"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_models():
    print("SAM 모델 로딩 중...")
    sam = sam_model_registry["vit_h"](checkpoint=CHECKPOINT_PATH)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )

    print("CLIP 모델 로딩 중...")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
    return mask_generator, clip_model, clip_preprocess

def segment_and_label_image(image_path: str, mask_generator, clip_model, clip_preprocess):
    print(f"이미지 로드 중: {image_path}")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image, dtype=np.uint8)


    # monkey patch 적용 (bool → uint8 강제 변환)
    from segment_anything.utils import amg
    original_remove = amg.remove_small_regions

    def patched_remove_small_regions(mask, min_area, mode="holes"):
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        return original_remove(mask, min_area, mode)

    amg.remove_small_regions = patched_remove_small_regions


    print("객체 세그먼트 생성 중...")
    masks = mask_generator.generate(image)
    print(f"{len(masks)} 개의 세그먼트가 생성되었습니다.")

    print("CLIP 모델을 위한 텍스트 토큰화 중...")
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {label}") for label in CANDIDATE_LABELS]).to(clip_model.device)

    labeled_segments = []
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        masked_img = np.zeros_like(image)
        masked_img[mask] = image[mask]

        y_indices, x_indices = np.where(mask)
        if len(y_indices) == 0 or len(x_indices) == 0:
            continue

        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        cropped_obj = masked_img[y_min:y_max, x_min:x_max]

        if cropped_obj.shape[0] < 10 or cropped_obj.shape[1] < 10:
            continue

        try:
            pil_image = Image.fromarray(cropped_obj)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            clip_image = clip_preprocess(pil_image).unsqueeze(0).to(clip_model.device)

            with torch.no_grad():
                image_features = clip_model.encode_image(clip_image)
                text_features = clip_model.encode_text(text_inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                values, indices = similarity[0].topk(3)

                top_labels = [(CANDIDATE_LABELS[idx], values[i].item()) for i, idx in enumerate(indices)]
                best_label = top_labels[0][0]
                confidence = top_labels[0][1]

                segment_info = {
                    "id": i,
                    "label": best_label,
                    "confidence": confidence,
                    "top_labels": top_labels,
                    "mask": mask,
                    "bbox": [x_min, y_min, x_max, y_max],
                    "area": mask_data["area"]
                }
                labeled_segments.append(segment_info)

                output_filename = f"{OUTPUT_DIR}/{best_label}_{i:03d}_{confidence:.2f}.png"
                cv2.imwrite(output_filename, cv2.cvtColor(cropped_obj, cv2.COLOR_RGB2BGR))
                print(f"세그먼트 {i}: 라벨 '{best_label}' (신뢰도: {confidence:.2f})")

        except Exception as e:
            print(f"세그먼트 {i} 처리 중 오류 발생: {e}")
            continue

    return labeled_segments, image

def visualize_results(image, labeled_segments, output_path="labeled_segments.png"):
    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(labeled_segments)))

    for i, segment in enumerate(labeled_segments):
        mask = segment["mask"]
        label = segment["label"]
        confidence = segment["confidence"]

        try:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except ValueError:
            _, contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            contour = contour.reshape(-1, 2)
            plt.plot(contour[:, 0], contour[:, 1], color=colors[i], linewidth=2)

        y, x = np.where(mask)
        if len(y) > 0 and len(x) > 0:
            center_y, center_x = int(np.mean(y)), int(np.mean(x))
            plt.text(center_x, center_y, f"{label} ({confidence:.2f})", color='white', fontsize=8,
                     bbox=dict(facecolor=colors[i], alpha=0.7))

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"결과 이미지가 {output_path}에 저장되었습니다.")
    plt.close()

def generate_label_statistics(labeled_segments):
    label_counts = {}
    label_areas = {}

    for segment in labeled_segments:
        label = segment["label"]
        area = segment["area"]

        label_counts[label] = label_counts.get(label, 0) + 1
        label_areas[label] = label_areas.get(label, 0) + area

    print("\n라벨별 통계:")
    print("-" * 40)
    print(f"{'라벨':<15} {'개수':<8} {'총 면적':<12} {'평균 면적':<12}")
    print("-" * 40)

    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        total_area = label_areas[label]
        avg_area = total_area / count
        print(f"{label:<15} {count:<8} {total_area:<12.0f} {avg_area:<12.0f}")


def patch_segment_anything_remove_small_regions():
    import segment_anything.utils.amg as amg
    import numpy as np
    import cv2

    original_func = amg.remove_small_regions

    def safe_remove_small_regions(mask, min_area, mode="holes"):
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        return original_func(mask, min_area, mode)

    amg.remove_small_regions = safe_remove_small_regions
    print("[패치 완료] remove_small_regions가 안전하게 래핑되었습니다.")
    

def main():
    patch_segment_anything_remove_small_regions()  # ✅ 반드시 먼저 호출
    mask_generator, clip_model, clip_preprocess = load_models()
    labeled_segments, image = segment_and_label_image(IMAGE_PATH, mask_generator, clip_model, clip_preprocess)
    visualize_results(image, labeled_segments)
    generate_label_statistics(labeled_segments)
    print(f"\n총 {len(labeled_segments)}개의 객체가 라벨링되었습니다.")
    print(f"세그먼트된 객체 이미지는 {OUTPUT_DIR} 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main()
