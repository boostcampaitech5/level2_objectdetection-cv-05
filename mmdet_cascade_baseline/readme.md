python train.py final_cascade_rcnn.py --work-dir ./cascade_rcnn --wandb-name cascade</br>
--work-dir "이번에 돌리는 모델 저장 폴더명"</br>
--wandb-name "wandb 저장 그래프 이름"</br>
</br>
- pth 불러와서 추가 학습 시키고 싶은 경우 train.py 228, 229줄 주석 처리한 부분 수정해주면 됩니다</br>
- (+) latest.pth 는 가장 마지막 pth 를 가리키는 바로가기 이기 때문에 이거만 남기고 pth 파일 다 지워버리면... 안됨ㅠ pth 파일 지울때 epoch_? 파일 하나는 남겨야합니다</br>
- work dir 폴더명 안바꾸고 돌리면 덮어씌워지니 이점 참고!</br>
- dataset.py 에 albu_train_transforms 변수가 albu 증강입니다
