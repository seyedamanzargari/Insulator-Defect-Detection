from ultralytics import YOLO
import cv2
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='model path')
    parser.add_argument('-s', '--source', type=str, help='image path')

    args = parser.parse_args()

    model = YOLO(f'yolov8{args.m}.pt')

    results = model(args.s)

    annotated_frame = results[0].plot()
    cv2.imshow('Output', annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()