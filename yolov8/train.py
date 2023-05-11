from ultralytics import YOLO
import argparse


def train(model_size='lagrge',data='./dataset/data.yaml',imgsz=1024, epochs=100,batch=16,patience=30,optimizer='SGD',name='yoloV8_l',lr0=0.01,lrf=0.01):
    # Load a model
    # model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO(f"yolov8{model_size[0]}.pt") 
    model.train(data=data, epochs=epochs,batch=batch,imgsz=imgsz,patience=patience,pretrained=True,optimizer=optimizer,name=name,lr0=lr0,lrf=lrf)  # train the model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train yolo v8')
    parser.add_argument('--model_size',type=str,default='large',help='nano, medium, large, xlarge')
    parser.add_argument('--yaml_path',type=str,default="./dataset/data.yaml")
    parser.add_argument('--img_size',type=int,default=1024)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--batch_size',type=int,default=16)
    parser.add_argument('--patience',type=int,default=50)
    parser.add_argument('--optimizer',type=str,default="SGD",help="possible method: SGD, Adam,AdamW, RMSProp")
    parser.add_argument('--save_name',type=str,default="voloV8")
    parser.add_argument('--initial_lr',type=int,default=0.01)
    parser.add_argument('--final_lr',type=int,default=0.01)

    args = parser.parse_args()
    train(args.model_size,args.yaml_path,args.img_size,args.epochs,args.batch_size,args.patience,args.optimizer,args.save_name,args.initial_lr,args.final_lr)


