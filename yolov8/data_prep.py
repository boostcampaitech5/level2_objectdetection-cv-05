import os,shutil
from pathlib import Path
import argparse
##################주소값#########################
# 1. raw 데이터를 copy_from_path에서 복사에서 copy_to_path로 복사
# 2. YOLO data set 폴더 트리를 만들고 COCO --> YOLO data set 형태로 변형
# 3. val.json에 의해 train/ validation set 나누기
# 4. data.yaml 파일 생성
#
# raw data를 working folder로 복사해서 train, val, test셋 COCO-->YOLO 포멧으로 변환
#데이터 복사
# copy_from_path = "/opt/ml/data_v1"
# copy_to_path = "/opt/ml/level2_objectdetection-cv-05/yolov8/dataset"
##################주소값#########################

def data_prep(copy_from_path,copy_to_path,):
    data_base_path = copy_to_path
    train_json_file_path=data_base_path+"train.json" # 파일로드 train 은 전체 데이터 
    val_json_file_path=data_base_path+"val.json" # 파일로드 val 은 val 데이터
    # 데이터 복사해오기
    shutil.copytree(copy_from_path,copy_to_path)
    shutil.rmtree(data_base_path+'test')
    os.remove(data_base_path+'test.json')
    # . 파일 지우기
    train_list =os.listdir(data_base_path+"train")
    base_list =os.listdir(data_base_path)
    for f_name in train_list:
        if f_name.startswith('.'):
            file_dir = data_base_path+"train/"+f_name
            os.remove(file_dir)
    for f_name in base_list:
        if f_name.startswith('.'):
            file_dir = data_base_path+f_name
            os.remove(file_dir)

    # 폴더 트리 재정의
    train_list =os.listdir(data_base_path+"train")
    os.mkdir(data_base_path+"train/images",)
    os.mkdir(data_base_path+"train/labels")
    for f_name in train_list:
        file_dir = data_base_path+"train/"+f_name
        to_dir = data_base_path+"train/images/"
        shutil.move(file_dir,to_dir)

    # annotation --> labels 폴더에 정리
    import json  

    with open(train_json_file_path,'r') as j:
        contents=json.loads(j.read())  # open : r - 읽기모드, w-쓰기모드, a-추가모드 
    image_list = contents['images']
    anno_list = contents['annotations']
    base = data_base_path+"train/"
    l_dir = base+"labels/"
    for anno in anno_list:
        img_id = anno['image_id']
        img_w = image_list[img_id]['width']
        img_h = image_list[img_id]['height']
        f_name = image_list[img_id]['file_name']
        f_name = f_name.split('/')[1]
        f_name = f_name.split('.')[0]
        f_name +='.txt'
        bbox = anno['bbox'] # 
        # 없으면 생성
        myfile = Path(l_dir+f_name)
        myfile.touch()
        f = open(l_dir+f_name,'a+')
        f.write(f"{anno['category_id']} {bbox[0]/img_w} {bbox[1]/img_w} {bbox[2]/img_h} {bbox[3]/img_w}\n")
        f.close()

    # val set 만들기
    os.mkdir(data_base_path+"val")
    os.mkdir(data_base_path+"val/images")
    os.mkdir(data_base_path+"val/labels")

    with open(val_json_file_path,'r') as j:
        contents=json.loads(j.read())  # open : r - 읽기모드, w-쓰기모드, a-추가모드 
    img_list = contents['images']
    data_base_path = data_base_path+""
    img_dir = data_base_path+'train/images/'
    to_img_dir = data_base_path+'val/images'
    anno_dir = data_base_path+'train/labels/'
    to_anno_dir = data_base_path+'val/labels'
    for img in img_list:
        f_name = img['file_name']
        f_name = f_name.split('/')[1]
        anno_name = f_name.split('.')[0]
        anno_name +='.txt'
        shutil.move(img_dir+f_name,to_img_dir)
        shutil.move(anno_dir+anno_name,to_anno_dir)

    # data.yaml config 만들기
    f = open(data_base_path+"data.yaml","w")
    f.write("train: "+data_base_path+"train\n")
    f.write("val: "+data_base_path+"val\n")
    f.write("\n")
    f.write("nc: 10\n")
    f.write("names: ['General trash','Paper','Paper pack','Metal','Glass','Plastic','Styrofoam','Plastic bag','Battery','Clothing']")
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='data prep data path')
    parser.add_argument('--data_from',type=str,default="/opt/ml/data_v1")
    parser.add_argument('--data_to',type=str,default="/opt/ml/level2_objectdetection-cv-05/yolov8/dataset/")
    args = parser.parse_args()
    data_prep(args.data_from,args.data_to)
    print('Done')


