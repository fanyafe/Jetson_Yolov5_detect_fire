# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import platform
import sys
import time
from pathlib import Path
import socket

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()            # detect.py路径
ROOT = FILE.parents[0]                     # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))             # add ROOT to PATH 确保后续导入其他包
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative 转为相对路径

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode, time_sync

# 传感器模块
import RPi.GPIO as GPIO
import time

gas_pin = 18     # 定义烟雾传感器引脚
gas1_pin = 16    # 定义气味传感器引脚
led_pin = 12     # 控制继电器开关   

# GPIO口设置
GPIO.setmode(GPIO.BOARD)                         # 定义引脚模式
GPIO.setup(gas_pin, GPIO.IN)                     # 烟雾为输入
GPIO.setup(gas1_pin, GPIO.IN)  
GPIO.setup(led_pin, GPIO.OUT, initial=GPIO.HIGH) # 继电器为输出，初始化为高电平
GPIO.setwarnings(False)

#tcp传输
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#设置IP和端口
host = ''
port = 8000
#bind绑定该端口
sock.bind((host, port))
sock.listen(10)
print(f"waiting for connect {host}:{port}")
client_socket, client_address = sock.accept()
print(f"connect to {client_address}")


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference

):

    # 1. 处理预测路径
    source = str(source)  # 解析 python detect.py --source data/images/bus.jpg  路径转为字符串类型
    save_img = not nosave and not source.endswith('.txt')  # save inference images = 不保存and不是txt结尾
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS) # 路径是否为图片格式文件或者视频文件格式
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')) # 路径是否为网络流
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file) # 是否为数字（摄像头）or txt结尾 or 网络流不是文件
    if is_url and is_file: # 如果文件或者网络要下载                                                                                     
        source = check_file(source)  # download

    # 2. Directories 保存结果文件夹          # exp1---> exp2 依次增加
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)                  # increment run 上面有参数project
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir 《--看是否传入 save_txt参数

    # 3. Load model 加载模型
    device = select_device(device) # GPU / CPU
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half) # 选择后端框架  pytorch/tensorrt/
    stride, names, pt = model.stride, model.names, model.pt                           # 从模型读取  模型步长，检测类别名，是否pytorch(true)
    imgsz = check_img_size(imgsz, s=stride)                                           # check image size 检查图片尺寸是否为32的倍数
 
    # 4. Dataloader
    if webcam:   # 摄像头/.txt结尾/网络流
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:     # 得到  图片路径  转换后的图片  原图  null  打印数据
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size 
    vid_path, vid_writer = [None] * bs, [None] * bs  
    
    # 5. Run inference 模型推理
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup 热身，拿一张空白图片初始化
    seen, windows, dt = 0, [], [0.0, 0.0, 0.0] 

    save_img = True
    timer =0
    start_time = time.time()
    save_interval = 1
    image_count = 0 
    save_folder = '/home/fan/Desktop/yolov5-6.2/yolov5-6.2/images/'
    os.makedirs(save_folder, exist_ok=True)

    a = 0
    b = 0
    c = 0
    Roll = 0
    for path, im, im0s, vid_cap, s in dataset: # 遍历加载数据 -》 图片路径 转换后的图片 原图 null 打印信息
        # （1）图片预处理
   
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)  # torch.Size[3,640,480] 转为tensor
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0  归一化
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim torch.Size([1,3,640,480])
        t2 = time_sync()
        dt[0] += t2 - t1  # 第一部分耗时

        # （2）Inference 预测
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize) # 数据增强  保存特征图（false） torch.Size([1,18900,85])
        t3 = time_sync()         
        dt[1] += t3 - t2  # 第二部分耗时

        # （3）NMS 非极大值抑制 18900 -> 几个    0.25    0.45       类别  
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det) #[1,5,6]
        dt[2] += time_sync() - t3   # 第三部分耗时

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        

        # （4）Process predictions 画框结果
        for i, det in enumerate(pred):  # 一个batch中的每个图片  torch.Size[5,6]
            
            seen += 1  # 记数
            Roll += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
                ##保存图片，一秒一张
                if save_img:
                   current_time = time.time()
                   elapsed_time = current_time - start_time
                   if elapsed_time >= save_interval:
                       image_name = f"{image_count}.jpg"
                       image_path = os.path.join(save_folder, image_name)
                       cv2.imwrite(image_path, im0)
                       print("保存图片：", image_path)
                       image_count += 1
                       start_time = time.time()
                       
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg 图片保存目录
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string 打印出图片尺寸
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0      # for save_crop 是否裁减出目标
            annotator = Annotator(im0, line_width=line_thickness, example=str(names)) # 绘图工具
            
            status=GPIO.input(gas_pin)
            status1=GPIO.input(gas1_pin)
           
            # 如果有框det，说明有火灾，烟雾传感器不需要判断 
            if len(det) :               
                # 如果有框，说明存在火灾,控制继电器打开
                #print('fire detect')
                #GPIO.output(led_pin, GPIO.LOW)
                a += 1
                b = 1
                # Rescale boxes from img_size to im0 size 将框画到原图
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique(): # 遍历每个框
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string，几个火，几个烟
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}') # 置信度以及标签
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop: # 是否保存截取部分
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                
                    
                
            # 如果摄像头没有检测到火灾，使用传感器模块判断
            if len(det) == 0:
                if status == False or status1 == False:
                    a += 1
                    c = 1
                #if status == True and status1 == True:   # 两个传感器都没有反应 
                    #print('no gas detect')
                    #GPIO.output(led_pin, GPIO.HIGH)      # 没有检测到烟雾，灯不会亮
                    
                #else :                                   # 其中一个有反应就打开继电器
                    #print('fire detect')
                    #GPIO.output(led_pin, GPIO.LOW)
                    #a += 1
     
            # Stream results
            im0 = annotator.result()  # 返回画好的图片

 
            if view_img:  # 显示图片
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img: # 保存图片
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0) # 保存路径
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
            print(a)
            
            if Roll == 3:
                if a == 3:
                    try:
                        print('fire detect')
                        GPIO.output(led_pin, GPIO.LOW)
                        if b == 1 and c == 0:
                           client_socket.sendall(b'1')
                        if b == 0 and c == 1:
                           client_socket.sendall(b'2')
                        if b == 1 and c == 1:
                           client_socket.sendall(b'3')
             

                    except socket.error as e:
                        print(f"error send data :{str(e)}")
                        client_socket.close()
                        break
                    except:
                        print("unexpected error occurred")
                        client_socket.close()
                        break
                else:
                    print('no gas detect')
                    GPIO.output(led_pin, GPIO.HIGH)      # 没有检测到烟雾，灯不会亮
                a = 0
                Roll = 0
                b = 0
                c = 0 

            if Roll == 10:
                if a >= 8:
                    try:
                        print('fire detect')
                        GPIO.output(led_pin, GPIO.LOW)
                        if b == 1 and c == 0:
                           client_socket.sendall(b'1')
                        if b == 0 and c == 1:
                           client_socket.sendall(b'2')
                        if b == 1 and c == 1:
                           client_socket.sendall(b'3')
             

                    except socket.error as e:
                        print(f"error send data :{str(e)}")
                        client_socket.close()
                        break
                    except:
                        print("unexpected error occurred")
                        client_socket.close()
                        break
                else:
                    print('no gas detect')
                    GPIO.output(led_pin, GPIO.HIGH)      # 没有检测到烟雾，灯不会亮
                a = 0
                Roll = 0
                b = 0
                c = 0 
           #sock.close()
        # （5）Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')


    # 6. Print results 打印输出结果
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    ##LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    #if save_txt or save_img:
     #   s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
      #  LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


# 输入命令参数
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/3/best.pt', help='model path(s)') # 权重  
    parser.add_argument('--source', type=str, default= '0', help='file/dir/URL/glob, 0 for webcam')  # 检测目标
    parser.add_argument('--data', type=str, default=ROOT / 'data/fire_smoke.yaml', help='(optional) dataset.yaml path')  # 
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w') # 图片大小
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand 图片640  ---》[640,640]
    print_args(vars(opt))                         # 输出参数信息
    return opt                                    # 返回给主函数参数


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))  # 检测各种包
    run(**vars(opt)) 


if __name__ == "__main__":
    opt = parse_opt()   # 解析命令行参数
    main(opt)           # 主函数
    GPIO.cleanup()      # GPIO口清理
    #client_socket.close()
