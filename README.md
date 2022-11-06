# TFOD_1 CUSTOM TRAINING FOR INSTANCE SEGMENTATION

![alt text](https://github.com/sudheeshe/TFOD_1_Custom_Instance_Segmenation_Template/blob/main/readme_imgs/DV8TLgkWsAEGsEs.jpg?raw=true)

# PREPARE WORKING ENVIRONMENT FOR TRAINING 

- Download following files
  1) TFOD-1.x (v1.13.0)[click here](https://github.com/tensorflow/models/tree/v1.13.0)
  2) Choose model[click here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)
  3) 

- Extract the files and rename `models-1.13`.0 to `models` and pretrained model name to any convenient short name `eg: mask_rcnn_inception_v2`

![alt text](https://github.com/sudheeshe/TFOD_1_Custom_Instance_Segmenation_Template/blob/main/readme_imgs/0_.jpg?raw=true)

- Inside models folder we have many sub folders which we can delete unnecessary folders. Keep only research folder remaining folders we can delete

![alt text](https://github.com/sudheeshe/TFOD_1_Custom_Instance_Segmenation_Template/blob/main/readme_imgs/1_.jpg?raw=true)

![alt text](https://github.com/sudheeshe/TFOD_1_Custom_Instance_Segmenation_Template/blob/main/readme_imgs/2_.jpg?raw=true)

- Inside research folder keep only `object detection`, `sim` folders and `setup.py` file. We can delete the remaining folders and files

![alt text](https://github.com/sudheeshe/TFOD_1_Custom_Instance_Segmenation_Template/blob/main/readme_imgs/3_.jpg?raw=true)

- From slim folder copy `nets` folder and paste in research folder

![alt text](https://github.com/sudheeshe/TFOD_1_Custom_Instance_Segmenation_Template/blob/main/readme_imgs/4_.jpg?raw=true)

![alt text](https://github.com/sudheeshe/TFOD_1_Custom_Instance_Segmenation_Template/blob/main/readme_imgs/5_.jpg?raw=true)

- Now open the project in Pycharm and select python 3.6 as version
![alt text](https://github.com/sudheeshe/TFOD_1_Custom_Instance_Segmenation_Template/blob/main/readme_imgs/6_.jpg?raw=true)

- Let's install the following packages in your new environment

for CPU 
```bash
pip install pillow lxml Cython contextlib2 jupyter matplotlib pandas opencv-python tensorflow==1.15.0
```

for GPU
```bash
pip install pillow lxml Cython contextlib2 jupyter matplotlib pandas opencv-python tensorflow-gpu==1.15.0
```

- In TFOD 1.X most of the files are written in protobuf. But in our case our python compiler will not understand protobuf format.
- So here we are converting these protobuf files to python file by the help of protobuf library.
- These protos files are available at `research/object_detection/protos` location

![alt text](https://github.com/sudheeshe/TFOD_1_Custom_Instance_Segmenation_Template/blob/main/readme_imgs/7_.jpg?raw=true)

- Now lets converts these files to python file, For that we need to install the below library

```bash
conda install -c anaconda protobuf
```

- Now lets convert these protos to python files

```bash
cd models/research/

protoc object_detection/protos/*.proto --python_out=.
```
- Now we can see that every python files corresponding to every protos files got generated

![alt text](https://github.com/sudheeshe/TFOD_1_Custom_Instance_Segmenation_Template/blob/main/readme_imgs/8_.jpg?raw=true)

- Now we will be running setup.py file to install object detection library
- Run the below command in `/research` folder

```bash
python setup.py install
```

- If the run was successful you will get the below message.
![alt text](https://github.com/sudheeshe/TFOD_1_Custom_Instance_Segmenation_Template/blob/main/readme_imgs/9_.jpg?raw=true)

- To verify that everything just we have done till here is working well. Let's do an evaluation
- For that launch jupyter notebook from research and go inside `object detection` folder

![alt text](https://github.com/sudheeshe/TFOD_1_Custom_Instance_Segmenation_Template/blob/main/readme_imgs/10_.jpg?raw=true)

- From there open `object_detection_tutorial.ipynb` jupyter file

![alt text](https://github.com/sudheeshe/TFOD_1_Custom_Instance_Segmenation_Template/blob/main/readme_imgs/11_.jpg?raw=true)

- It will ask to select the kernal.

![alt text](https://github.com/sudheeshe/TFOD_1_Custom_Instance_Segmenation_Template/blob/main/readme_imgs/12_.jpg?raw=true)

- Run all the cells

![alt text](https://github.com/sudheeshe/TFOD_1_Custom_Instance_Segmenation_Template/blob/main/readme_imgs/13_.jpg?raw=true)

- The result will be 
- 
![alt text](https://github.com/sudheeshe/TFOD_1_Custom_Instance_Segmenation_Template/blob/main/readme_imgs/14_.jpg?raw=true)

- Note: 
matplotlib inline will not work by default so the image won't appear in the notebook
So run the below code in a notebook
```bash
%matplotlib inline

plt.figure(figsize=(20,20))
plt.imshow(image_np)
```


- Now we need to add some files to `/research` folder which is available in this repo

       1) create_tf_records.py
       2) read_pbtxt.py 

- Now copy these files to `/research` folder

        1) "export_inference_graph.py" file from "research/object_detection/"
        2) "string_int_label_map_pb2.py" from "research/object_detection/protos"
        3) "train.py" from "research/object_detection/legacy"

- Create a new folder named `mask_rcnn` in `/research` folder and move files inside `mask_rcnn_inception_v2` 

![alt text](https://github.com/sudheeshe/TFOD_1_Custom_Instance_Segmenation_Template/blob/main/readme_imgs/15_.jpg?raw=true)


# PREPARE DATA FOR TRAINING

### Data Folder Structure
- Create a new folder named `training_data` in `/research` folder
- Create sub folders` train_imgs, train_json,test_imgs, test_json`


- Place `training images` in `train_imgs` and `training image annotations` in `train_json` folder
- Similarly, place `validation images` in `test_imgs` folder and `validation image annotations` in `test_json` folder

![alt text](https://github.com/sudheeshe/TFOD_1_Custom_Instance_Segmenation_Template/blob/main/readme_imgs/17_.jpg?raw=true)

![alt text](https://github.com/sudheeshe/TFOD_1_Custom_Instance_Segmenation_Template/blob/main/readme_imgs/18_.jpg?raw=true)

- copy labelmap.pbtxt to `training_data` folder and change the class names.

![alt text](https://github.com/sudheeshe/TFOD_1_Custom_Instance_Segmenation_Template/blob/main/readme_imgs/19_.jpg?raw=true)

- Now do the necessary changes in `create_tf_records.py` file

![alt text](https://github.com/sudheeshe/TFOD_1_Custom_Instance_Segmenation_Template/blob/main/readme_imgs/16_.jpg?raw=true)

- Note: if absolute path is not working give full path

- With this `create_tf_records.py` file we will be creating `train.record` and `test.record`, Which is nothing but an efficient file format 
 which Object Detection API uses for training the model.

- First un-comment training section and comment test sectionand run `create_tf_records.py` which create train.record
- Then comment training section and un-comment test section and run `create_tf_records.py` which will create test.record

- On a sucessful creation of train.record we can see the below message
- 
![alt text](https://github.com/sudheeshe/TFOD_1_Custom_Instance_Segmenation_Template/blob/main/readme_imgs/20_.jpg?raw=true)

- Note: Check the size of train.record if it shows `0K` the check the specified paths
- Note: Based on labelme version the `imagePath` mentioned will be in different ways.
 If we open any annotation file and check for `imagePath`  

![alt text](https://github.com/sudheeshe/TFOD_1_Custom_Instance_Segmenation_Template/blob/main/readme_imgs/21_.jpg?raw=true)

- This `imagePath` format may change based on labelme version.

- Always check the annotation file for `imagePath` format and make any changes needed `split() function` to get output as only image name.

- So debug the `create_tf_records.py` @ line number 171 and verify it is only image name.
- 
![alt text](https://github.com/sudheeshe/TFOD_1_Custom_Instance_Segmenation_Template/blob/main/readme_imgs/22_.jpg?raw=true)

- This below image shows the wrong format
![alt text](https://github.com/sudheeshe/TFOD_1_Custom_Instance_Segmenation_Template/blob/main/readme_imgs/23_.jpg?raw=true)

- In this I have used `split('\\')` but may be in another case we need this to change `split('/')` or `split('\')` depend on labelme version.



# TRAINING CUSTOM MASK-RCNN MODEL 

- Create a new folder named `custom_training` on `/research` folder

- Now we need to config file to train the mask-rcnn model
- If we check `research/object_detection/samples/configs` path we can see multiple configuration files for all the models which are available in TFOD 1.x
- We need to copy `mask_rcnn_inception_v2_coco` file and paste in `custom_training` folder which we created previously.
- And rename to a smaller name `eg: custom_config`

![alt text](https://github.com/sudheeshe/TFOD_1_Custom_Instance_Segmenation_Template/blob/main/readme_imgs/24_.jpg?raw=true)

- Open `custom_config` file, and we need to change few lines of code

1) Change the number of classes at line number 10 as per the project 

![alt text](https://github.com/sudheeshe/TFOD_1_Custom_Instance_Segmenation_Template/blob/main/readme_imgs/25_.jpg?raw=true)

![alt text](https://github.com/sudheeshe/TFOD_1_Custom_Instance_Segmenation_Template/blob/main/readme_imgs/25.1_.jpg?raw=true)

2) Provide the path of the pre-trained model which we downloaded from Model zoo at line number 127
3) Change the number of epochs at line number 133

![alt text](https://github.com/sudheeshe/TFOD_1_Custom_Instance_Segmenation_Template/blob/main/readme_imgs/26_.jpg?raw=true)

![alt text](https://github.com/sudheeshe/TFOD_1_Custom_Instance_Segmenation_Template/blob/main/readme_imgs/26.1_.jpg?raw=true)

4) Provide path of `train.record` path at line 142, 
5) Give path of `labelmap.pbtxt` path at line 144 & 160
6) Give path of `test.record` path at line 158,

![alt text](https://github.com/sudheeshe/TFOD_1_Custom_Instance_Segmenation_Template/blob/main/readme_imgs/27_.jpg?raw=true)

![alt text](https://github.com/sudheeshe/TFOD_1_Custom_Instance_Segmenation_Template/blob/main/readme_imgs/27.1_.jpg?raw=true)

 - From `/research/slim/`  copy `deployment` folder and paste in `/research` folder

![alt text](https://github.com/sudheeshe/TFOD_1_Custom_Instance_Segmenation_Template/blob/main/readme_imgs/28_.jpg?raw=true)

![alt text](https://github.com/sudheeshe/TFOD_1_Custom_Instance_Segmenation_Template/blob/main/readme_imgs/29_.jpg?raw=true)



### Start training of the model
 
```bash
cd models/research 
```

- make changes in the code 
`python train.py --logtostderr --train_dir=<path to custom model saving folder> --pipeline_config_path=<path to config file>`

Eg:
```bash
python train.py --logtostderr --train_dir=custom_training/ --pipeline_config_path=custom_training/custom_config.config
```


- Once the Training starts you can see a message like this

![alt text](https://github.com/sudheeshe/TFOD_1_Custom_Instance_Segmenation_Template/blob/main/readme_imgs/30_.jpg?raw=true)


























### To view tensorboard logs

```bash
tensorboard --logdir runs
```

### Run Inference With Trained Weights
- 2 paths we need to provide

      1) best weight path (best.pt)
      2) test image path

- We can provide what is the confidence score (threshold)  using `--conf` argument for prediction. whatever predictions below this will not get generated. Closer to `0 - less threshold` closer to `1 - higher threshold`.


```bash
cd yolov5
python detect.py --weights runs/train/yolov5s_results/weights/best.pt --img 416 --conf 0.5 --source test/images
```

- Prediction images will be available in yolov5/runs/detect

### Export Trained Weights for Future Inference

```bash
from google.colab import drive
drive.mount('/content/gdrive')
```

```bash
cp /content/yolov5/runs/train/yolov5s_results/weights/best.pt /content/gdrive/MyDrive/Research/SignLanguageDetection
```

### Reference blogs

- https://towardsdatascience.com/how-to-train-a-custom-object-detection-model-with-yolo-v5-917e9ce13208
- https://github.com/entbappy/Sign-Language-Generation-From-Video-using-YOLOV5
  