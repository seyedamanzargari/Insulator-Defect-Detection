from ultralytics import YOLO
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model parameters
    parser.add_argument('-m', '--model', default='n', choices=['n', 's', 'm', 'l', 'x'])
    parser.add_argument('-b', '--batch', type=int, default=32)  
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'mps'])
    parser.add_argument('-d', '--data', type=str, default='CPLID')
    parser.add_argument('-e', '--epochs', default=30, type=int)
    parser.add_argument('-s', '--image-size', default=640, type=int)
    parser.add_argument('-n', '--name', type=str, default=None)
    parser.add_argument('-p', '--pretrained', action='store_true', default=True)
    parser.add_argument('--optimizer', type=str, default='auto', choices=['SGD', 'Adam', 'auto'])

    args = parser.parse_args()

    if args.data == "IDD":
        args.data = 'IDD_dataset.yaml'
    elif args.data == "CPLID":
        args.data = 'CPLID_dataset.yaml'

    model = YOLO(f"yolov8{args.model}.pt")

    results = model.train(
            batch=args.batch,
            device=args.device,
            data=args.data,
            epochs=args.epochs,
            imgsz=args.image_size,
            project=args.name,
            pretrained=args.pretrained,
            optimizer=args.optimizer
        )