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

- In this i have used `split('\\')` but may be in another case we need this to change `split('/')` or `split('\')` depend on labelme version.
























- Train, Val and Test folders should have sub-folder named `images` and `labels`
- labels should be in .txt format
- Test folder not need labels, since we are using test images for prediction
- Note: Ignore on `labels.cache` file it generated during previous training

![alt text](https://github.com/sudheeshe/YoloV5_Custom_training_template/blob/main/imgs/data_structure_template.png?raw=true)







### Create .zip file with for training
  1) train
  2) val
  3) test
  4) data.yaml
  5) YOLO_V5_Custom_Training.ipynb
  6) custom_yolov5s.yaml



# PREPARE PAPERSPACE VM


  ### Step - 1

  #### Create ssh on MobaXterm

  - Open a new terminal in `MobaXterm` Run the following command

  ```bash
  ssh-keygen
  ```

  ```bash
  - cd /home/sudheeshe/.ssh
  ```

  ![alt text](https://github.com/sudheeshe/YoloV5_Custom_training_template/blob/main/imgs/ssh_keygen.png?raw=true)

  - Print out your public key with
  ```bash
  cat ~/.ssh/id_rsa.pub
  ```

 ![alt text](https://github.com/sudheeshe/YoloV5_Custom_training_template/blob/main/imgs/show_ssh_key.png?raw=true)

- copy the above ssh key.

 ### Step - 2

- Select `Core virtual servers`

![alt text](https://github.com/sudheeshe/YoloV5_Custom_training_template/blob/main/imgs/1_.png?raw=true)

- Select `create a machine` from that select `ML-in-a-box-Ubuntu 20.4` version

![alt text](https://github.com/sudheeshe/YoloV5_Custom_training_template/blob/main/imgs/2_.png?raw=true)

- Select GPU as P4000

![alt text](https://github.com/sudheeshe/YoloV5_Custom_training_template/blob/main/imgs/3_.png?raw=true)

- Add SSH key if ssh was not added previously. If ssh key is already added and we need to rewrite the existing ssh key you can go with steps mentioned later on this file.

- Click create button to create the VM


### First way to access VM using MobiXterm is
### Method - 1 Adding SSH Keys

  - You can add SSH keys on the SSH Keys page in the console. Specify a name for the key and copy/paste the output from step 2 of key generation. Once you’ve added a key, you can select it during machine creation to automatically add it to new CORE machines.

#### step - i

![alt text](https://github.com/sudheeshe/YoloV5_Custom_training_template/blob/main/imgs/ssh_add_1.png?raw=true)

#### step - ii

![alt text](https://github.com/sudheeshe/YoloV5_Custom_training_template/blob/main/imgs/ssh_add_2.png?raw=true)

- Yellowline shows previously added ssh keys, with the ssh key name

#### step - iii

![alt text](https://github.com/sudheeshe/YoloV5_Custom_training_template/blob/main/imgs/ssh_add_3.png?raw=true)

- paste the copied ssh key from MobaXterm here

### Step - 3

- Start the machine we created
- Once the VM is started we can initialize the connection by using ssh, Click on connet button

![alt text](https://github.com/sudheeshe/YoloV5_Custom_training_template/blob/main/imgs/4_.png?raw=true)

- This generates the pop-up with ssh command, which we need to copy and paste it on MobaXterm to initiate the connection

![alt text](https://github.com/sudheeshe/YoloV5_Custom_training_template/blob/main/imgs/5_.png?raw=true)

- Now open a new terminal on MobaXterm

![alt text](https://github.com/sudheeshe/YoloV5_Custom_training_template/blob/main/imgs/6_.png?raw=true)

- Paste the ssh commad on this new terminal and press enter to make the connection with paperspace VM

![alt text](https://github.com/sudheeshe/YoloV5_Custom_training_template/blob/main/imgs/7_.png?raw=true)

- Connection will get established, we can now see the folders in the VM marked in red

![alt text](https://github.com/sudheeshe/YoloV5_Custom_training_template/blob/main/imgs/8_.png?raw=true)

- We will be using this terminal for code execution.

### Method - 2 By password

![alt text](https://github.com/sudheeshe/YoloV5_Custom_training_template/blob/main/imgs/ssh_password.png?raw=true)

- Password will be sent to the registered email id
- Go to created VM and click on `connect` and copy the ssh command and past on MobiXterm terminal and press enter.
- While establishing connection to VM using MobiXterm it will prompt for password.
provide the password over there.

### Choose Method - 2 over Method - 1, Since VM will ask for password even if you create connection through ssh when we try to again connect after restart of VM.


# TRAINING YOLOV5 MODEL

### Create a working directory with project name `eg: PCB_Defect_Detection` on `Paperspace or Colab`


### Change to working directory

```bash
cd <path of working folder eg: PCB_Defect_Detection>
```

### Check current working directory

```bash
pwd
```

### Install Miniconda and create environment

- Here we have used Python 3.7 64bit installer, from below link we can choose any other Linux Python installer

https://docs.conda.io/en/latest/miniconda.html#linux-installers

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh

bash Miniconda3-py37_4.12.0-Linux-x86_64.sh

source ~/.bashrc

conda create -n <env_name> python=3.7 -y

conda activate <env_name>
```


### Cloning YOLOv5 repo on working folder eg: PCB_Defect_Detection

```bash
git clone https://github.com/ultralytics/yolov5

cd yolov5

git reset --hard fbe67e465375231474a2ad80a4389efc77ecff99
```

### Install dependencies
```bash
pip install -qr requirements.txt
```

### Copy the above created zip file to the working Directory and unzip it.

- Now folder structure will look like

      yolov5/
              1) "yolo files downloaded from git repos"
              2) train
              3) val
              4) test
              5) data.yaml
              6) YOLO_V5_Custom_Training.ipynb
              7) custom_yolov5s



### Save the custom_yolov5s.yaml

#### what is `custom_yolov5s.yaml` file
~~~
- In `yolov5` folder we already have `yolov5s.yaml` which was used to train `yolov5 small version` (pretrained yolo).
- In order to train our custom yolo we are editing this `yolov5s.yaml` according to our specification.
- We are only changing the `number of classes (nc:)` in this yaml file an creating a new file called `custom_yolov5s.yaml`
~~~

- Since we have already prepared custom_yolov5s.yaml and which is available in zip file.
- So We are moving the `custom_yolov5s.yaml` to the `models/custom_yolov5s.yaml` location as shown below

![alt text](https://github.com/sudheeshe/YoloV5_Custom_training_template/blob/main/imgs/custom_yaml.png?raw=true)

### Change the data.yaml file `train and val` locations, `names` name of classes and `nc` number of classes

![alt text](https://github.com/sudheeshe/YoloV5_Custom_training_template/blob/main/imgs/data_yaml.png?raw=true)


### Train Custom YOLOv5 Detector

Here, we are able to pass a number of arguments:

```
img: define input image size
batch: determine batch size
epochs: define the number of training epochs. (Note: often, 3000+ are common here!)
data: set the path to our yaml file
cfg: specify our model configuration
weights: specify a custom path to weights. (Note: you can download weights from the Ultralytics Google Drive folder)
name: result names
nosave: only save the final checkpoint
cache: cache images for faster training
```

- We can change file location of`data.yml` and `custom_yolov5s.yaml` if needed,
- Since we have placed these files inside `yolov5` folder itself, So we can run the below command as it is. We don't need to change the path.

```bash
cd yolov5

python train.py --img 416 --batch 32 --epochs 100 --data data.yaml --cfg models/custom_yolov5s.yaml --weights '' --name yolov5s_results  --cache


```

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
  