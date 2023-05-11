# Yolo v8 Image Detection 사용법  

Data Prep:  
python data_prep.py --data_from "{raw_data_path}" --data_to "{yolo_working_folder_path}"  
  
Train Yolo:  
python train.py  
  
Inference: 모델은 yolo working dir의 runs/detect/{run_name}/weights 폴더에 있다.  
python inference.py --model "{model from training}"  
