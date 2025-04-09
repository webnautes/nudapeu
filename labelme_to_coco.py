
import os
import json
import numpy as np
import glob
import shutil
from PIL import Image
import datetime

def create_coco_structure(output_dir):
    """COCO 디렉토리 구조 생성"""
    os.makedirs(os.path.join(output_dir, "train2017"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val2017"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)

def init_coco_json():
    """COCO JSON 구조 초기화"""
    coco_json = {
        "info": {
            "description": "Converted from LabelMe",
            "url": "",
            "version": "1.0",
            "year": datetime.datetime.now().year,
            "contributor": "",
            "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [{
            "id": 1,
            "name": "Unknown",
            "url": ""
        }],
        "images": [],
        "annotations": [],
        "categories": []
    }
    return coco_json

def add_categories(coco_json, json_files):
    """JSON 파일에서 모든 카테고리(라벨) 추출하여 COCO 형식으로 추가"""
    categories = set()
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for shape in data["shapes"]:
            categories.add(shape["label"])
    
    for i, category in enumerate(sorted(categories), 1):
        coco_json["categories"].append({
            "id": i,
            "name": category,
            "supercategory": "none"
        })
    
    return {cat["name"]: cat["id"] for cat in coco_json["categories"]}

def convert_labelme_to_coco(input_dir, output_dir, train_ratio=0.8, val_ratio=0.2):
    """labelme 디렉토리를 COCO 형식으로 변환 (YOLOX 학습용)"""
    # COCO 디렉토리 구조 생성
    create_coco_structure(output_dir)
    
    # JSON 파일 목록
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    # COCO JSON 초기화
    train_coco = init_coco_json()
    val_coco = init_coco_json()
    
    # 카테고리 추가
    category_map = add_categories(train_coco, json_files)
    
    # val에도 동일한 카테고리 설정
    val_coco["categories"] = train_coco["categories"]
    
    # 파일 목록 랜덤 분할
    file_names = [os.path.splitext(os.path.basename(json_file))[0] for json_file in json_files]
    np.random.shuffle(file_names)
    
    # 각 세트의 크기 계산
    train_size = int(len(file_names) * train_ratio)
    
    # 데이터 분할
    train_files = file_names[:train_size]
    val_files = file_names[train_size:]
    
    # 변환 작업 수행
    train_image_id = 1
    val_image_id = 1
    train_annotation_id = 1
    val_annotation_id = 1
    
    for json_file in json_files:
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        
        # 이미지 파일 확인
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            potential_img = os.path.join(input_dir, base_name + ext)
            if os.path.exists(potential_img):
                img_path = potential_img
                break
        
        if img_path is None:
            print(f"이미지를 찾을 수 없습니다: {base_name}")
            continue
        
        # 데이터셋 결정
        if base_name in train_files:
            target_set = "train2017"
            coco_json = train_coco
            image_id = train_image_id
            annotation_id = train_annotation_id
        else:
            target_set = "val2017"
            coco_json = val_coco
            image_id = val_image_id
            annotation_id = val_annotation_id
        
        # 이미지 JPG로 변환
        jpg_filename = f"{image_id:012d}.jpg"  # COCO 형식에 맞게 12자리 숫자 형식
        img_dest = os.path.join(output_dir, target_set, jpg_filename)
        
        # 이미지 열고 JPG로 저장
        img = Image.open(img_path)
        width, height = img.size
        
        # RGB 형식으로 변환 (PNG 등에서 알파 채널이 있을 경우)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(img_dest, 'JPEG', quality=95)
        
        # COCO 이미지 정보 추가
        coco_json["images"].append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": jpg_filename,
            "license": 1,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # 어노테이션 정보 처리
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for shape in data["shapes"]:
            category_id = category_map[shape["label"]]
            
            # points를 바운딩 박스로 변환
            points = np.array(shape["points"]).astype(float)
            xmin = np.min(points[:, 0])
            ymin = np.min(points[:, 1])
            xmax = np.max(points[:, 0])
            ymax = np.max(points[:, 1])
            
            # 이미지 경계 내로 좌표 제한
            xmin = max(0, min(width-1, xmin))
            ymin = max(0, min(height-1, ymin))
            xmax = max(0, min(width-1, xmax))
            ymax = max(0, min(height-1, ymax))
            
            # COCO 형식의 bbox: [x, y, width, height]
            bbox_width = xmax - xmin
            bbox_height = ymax - ymin
            
            # COCO 어노테이션 정보 추가
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [float(xmin), float(ymin), float(bbox_width), float(bbox_height)],
                "area": float(bbox_width * bbox_height),
                "segmentation": [],  # 세그먼테이션 정보가 필요하면 여기에 추가
                "iscrowd": 0
            }
            
            # 라벨메 데이터가 폴리곤인 경우 세그먼테이션 정보 추가
            if shape["shape_type"] == "polygon":
                # 폴리곤 점들을 COCO 형식으로 변환 (1차원 배열)
                segmentation = []
                for point in shape["points"]:
                    segmentation.extend([float(point[0]), float(point[1])])
                annotation["segmentation"] = [segmentation]
            
            coco_json["annotations"].append(annotation)
            annotation_id += 1
        
        # ID 증가
        if base_name in train_files:
            train_image_id += 1
            train_annotation_id = annotation_id
        else:
            val_image_id += 1
            val_annotation_id = annotation_id
    
    # COCO JSON 파일 저장
    with open(os.path.join(output_dir, "annotations", "instances_train2017.json"), 'w', encoding='utf-8') as f:
        json.dump(train_coco, f, indent=2)
    
    with open(os.path.join(output_dir, "annotations", "instances_val2017.json"), 'w', encoding='utf-8') as f:
        json.dump(val_coco, f, indent=2)
    
    print(f"변환 완료! 총 {len(file_names)}개 이미지가 COCO 형식으로 변환되었습니다.")
    print(f"- 훈련 세트: {len(train_files)}개")
    print(f"- 검증 세트: {len(val_files)}개")
    print(f"COCO 구조가 {output_dir}에 생성되었습니다.")

# 사용 예시
if __name__ == "__main__":
    input_dir = "labelme_data"  # labelme JSON 파일이 있는 폴더
    output_dir = "datasets/coco128"  # YOLOX 설정 파일에 맞춘 출력 폴더
    
    # 기본 비율: 80% 훈련, 20% 검증
    convert_labelme_to_coco(input_dir, output_dir)
