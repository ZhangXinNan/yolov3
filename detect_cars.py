import argparse
import time
import logging
from sys import platform
import torch
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

from detect_img import DetectorYolo


def normalize(box, w, h):
    box[0] /= float(w)
    box[2] /= float(w)
    box[1] /= float(h)
    box[3] /= float(h)
    return box


def check_car(det):
    # print('check_car : ')
    w = [label in [2,5,7] for label in det[:, -1]]
    # print(w)
    det = det[w]
    # print(det)
    return det


def get_best_car(det, w, h):
    '''biggest area
    return [x1,x2,y1,y2,conf,label]
    '''
    if len(det) <= 0:
        raise Exception

    best_idx = 0
    if len(det) > 1:
        x1, y1, x2, y2 = det[0][:4]
        best_area = (x2 - x1) * (y2 - y1)
        idx = 1
        for x1, y1, x2, y2, conf_obj, _, label in det[1:]:
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_idx = idx
                best_area = area
            idx += 1
    x1, y1, x2, y2, conf_obj, _, label = det[best_idx]
    # x1 /= float(w)
    # y1 /= float(w)
    # x2 /= float(h)
    # y2 /= float(h)
    return [x1, y1, x2, y2, conf_obj, label]
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
    # /home/zhangxin/gitlab/CarRecognition/Data/
    parser.add_argument('--source', type=str, default='all_data.txt')  # input file
    parser.add_argument('--output', type=str, default='all_data_det.txt')  # output folder
    parser.add_argument('--out_dir', default='/home/zhangxin/data_ah_det')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1) or cpu')
    # parser.add_argument('--pid', default=0, type=int)
    opt = parser.parse_args()
    return opt


def main(opt):
    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    logging.basicConfig(level=logging.INFO,
        filename=str_time + '_detect_car.log',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    save_img = True
    print(opt)
    detector = DetectorYolo(opt.device, opt.img_size, opt.cfg, opt.weights, opt.data)
    fi = open(opt.source, 'r')
    fo = open(opt.output, 'w')

    for line in fi:
        line = line.strip()
        arr = line.split('\t')
        # label = int(arr[1])
        # if opt.pid >= 0 and label % 10 != opt.pid:
        #     continue
        if len(arr) != 3:
            print(line, arr, 'len(arr) != 3')
            fo.write(line + '\n')                   # 如果列数多于3个，则表示检测过了，不需要再进行检测
            continue
        if not os.path.isfile(arr[0]):
            print(arr[0], ' is not a file')
            continue
        if save_img:
            a = arr[0].split(os.path.sep)
            # productlib_name.subdir_name.filename
            new_filename = a[-3] + '.' + a[-2] + '.' + a[-1]
            # autohome_subdir_name
            new_out_dir = os.path.join(opt.out_dir, '.'.join(arr[2][1:-1].split(',')).replace(' ', ''))
            if not os.path.isdir(new_out_dir):
                os.makedirs(new_out_dir)
            save_path = os.path.join(new_out_dir, new_filename)
            if os.path.isfile(save_path):
                # 文件已经存在了，表示已经保存过检测结果了，也忽略继续
                continue
        img = cv2.imread(arr[0])
        if img is None:
            logging.warning(arr[0] + ' imread error !!!')
            continue
        h, w = img.shape[:2]
        with torch.no_grad():
            det = detector.detect_objs(img)
            if det is None:
                logging.info(arr[0] + ' det is None !!!')
                continue
            if torch.cuda.is_available():
                det = det.cpu()
            det = det.numpy()
            print('detect_objs : ', det, type(det))
            det = check_car(det)
            print("detect_cars : ", det, type(det))

        if len(det) < 1:
            print(arr[0], ' no car !!!')
            arr.insert(2, 'NO')
            fo.write('\t'.join(arr) + '\n')         # 没有检测到车，则保存原图
            continue

        ret = get_best_car(det, w, h)
        print('get_best_car : ', ret, type(ret))
        arr.insert(2, '[' + ','.join([str(x) if i == 4 else str(int(x)) for i, x in enumerate(ret)]) + ']')
        fo.write('\t'.join(arr) + '\n')
        if save_img:
            x1, y1, x2, y2 = ret[:4]
            cv2.imwrite(save_path, img[int(y1):int(y2), int(x1):int(x2), :])
        # break

    fo.close()
    fi.close()


if __name__ == '__main__':
    # logger = logging.getLogger(__name__)
    main(get_args())
