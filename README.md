# face_mask_detection


a realtime face mask detector using opencv,yolo and ssd.



![](demo/demo_face.mp4)




1. First install cuda,cudnn for gpu acceleration.make sure they are on proper paths
2. install opencv with proper libraries

now do the following
```
git clone https://github.com/AlexeyAB/darknet
```
```
cd darknet
make OPENCV=1 GPU=1  OPENMP=1 CUDNN=1 LIBSO=1 -j $(nproc)
cd ..
```
if there is any error in makefile then its probably due to not having proper library paths for cuda,cudnn,opencv
in that case edit the makefile

```
git clone https://gitlab.com/mushrafi88/face_mask_detection.git
cp face_mask_detection/ssd_face_detector/deploy.prototxt darknet/
cp face_mask_detection/ssd_face_detector/res10_300x300_ssd_iter_140000.caffemodel darknet/
cp face_mask_detection/training_yolo_v4/backup/yolov4-obj_best.weights darknet/
cp face_mask_detection/training_yolo_v4/obj.data darknet/data/
cp face_mask_detection/training_yolo_v4/obj.names darknet/data/
cp face_mask_detection/training_yolo_v4/yolov4-obj.cfg darknet/cfg/
```

```
cp face_mask_detection/final_model_creation/face_detector_cam.py darknet/
cp face_mask_detection/final_model_creation/ssd_realtime_cam.ipynb darknet/
```
```
cd darknet
python face_detector_cam.py
```
if there is a camera error
edit the last line of the face_detector_cam.py to either 0 or 1
