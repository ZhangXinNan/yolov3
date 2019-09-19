import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *


class DetectorYolo:
    def __init__(self, device, img_size, cfg, weights, data):
        torch.no_grad()
        # Initialize
        self.device = torch_utils.select_device(device=device)
        # Initialize model
        self.img_size = img_size
        self.model = Darknet(cfg, img_size)
        # print(model)
        # Load weights
        if weights.endswith('.pt'):  # pytorch format
            self.model.load_state_dict(torch.load(weights, map_location=self.device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(self.model, weights)
        # Fuse Conv2d + BatchNorm2d layers
        # model.fuse()
        # Eval mode
        self.model.to(self.device).eval()
        # Get classes and colors
        self.classes = load_classes(parse_data_cfg(data)['names'])
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.classes))]

    def detect_objs(self, opt, im0s, save_img=True):
        # img_size = opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
        save_path, path = opt.output, opt.source
        # im0s = cv2.imread(path)
        img, *_ = letterbox(im0s, new_shape=self.img_size)
        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to fp16/fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        # Run inference
        t = time.time()
        # Get detections
        img = torch.from_numpy(img).to(self.device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred, _ = self.model(img)

        # for i, det in enumerate(non_max_suppression(pred, opt.conf_thres, opt.nms_thres)):  # detections per image
        dets = non_max_suppression(pred, opt.conf_thres, opt.nms_thres)
        det = dets[0]
        p, s, im0 = path, '', im0s
        s += '%gx%g ' % img.shape[2:]  # print string
        # if det is not None and len(det):
        if det is None or len(det)==0:
            raise Exception #,"det error !!!"
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        # Print results
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
            s += '%g %ss, ' % (n, self.classes[int(c)])  # add to string
        print('%sDone. (%.3fs)' % (s, time.time() - t))
        return det
        

    def plot_boxes(self, img, det):
        # Write results
        for *xyxy, conf, _, cls in det:
            label = '%s %.2f' % (self.classes[int(cls)], conf)
            plot_one_box(xyxy, img, label=label, color=self.colors[int(cls)])
            print(xyxy, conf, _, cls)
        return img  


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
    parser.add_argument('--source', type=str, default='data/samples/bus.jpg', help='source')  # input file
    parser.add_argument('--output', type=str, default='output/bus_result.jpg', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    # parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    opt = parser.parse_args()
    return opt


def main(opt):
    print(opt)
    dector = DetectorYolo(opt.device, opt.img_size, opt.cfg, opt.weights, opt.data)
    if os.path.isfile(opt.source):
        img = cv2.imread(opt.source)
        det = dector.detect_objs(opt, img)
        img_show = dector.plot_boxes(img, det)
        cv2.imwrite(opt.output, img_show)
    elif os.path.isdir(opt.source):
        if not os.path.isdir(opt.output):
            os.makedirs(opt.output)
        for filename in os.listdir(opt.source):
            print(filename)
            if os.path.splitext(filename)[-1].lower() not in img_formats:
                continue
            img_path = os.path.join(opt.source, filename)
            save_path = os.path.join(opt.output, filename)
            print(img_path)
            print(save_path)
            img = cv2.imread(img_path)
            det = dector.detect_objs(opt, img)
            img_show = dector.plot_boxes(img, det)
            cv2.imwrite(save_path, img_show)


if __name__ == '__main__':
    main(get_args())
