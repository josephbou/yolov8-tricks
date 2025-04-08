<div align="">
  <p>
    <a href="https://yolovision.ultralytics.com/" target="_blank">
      </a>
  </p>


### ⭐Introduction
My master's research focuses on object detection, using YOLOv8 as the model and VisDrone2019 as the dataset, which is suitable for small object aerial image detection. Combining the <a href="https://docs.ultralytics.com/">official documentation</a> and <a href="https://www.bilibili.com/video/BV1QC4y1R74t/?spm_id_from=333.788.top_right_bar_window_custom_collection.content.click&vd_source=5fe50b1b35a25689fb0988c454fec5e0">Yan Xuechang's video tutorials</a>, I learned how to modify the model, add new modules to YOLOv8, and how to modify YAML configuration files. I've compiled this YOLOv8 training repository where you can freely add different improvement YAML configuration files for model training. If you find it helpful, please give it a star!


### ⭐Detailed Explanation


<details open>
<summary>Installation</summary>


  
First, clone this repository to your local computer using git:


```bash
git clone git@github.com:chaizwj/yolov8-tricks.git
```
Then use the pip command to install the `ultralytics` package in a [**Python>=3.8**](https://www.python.org/) environment, which also requires [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/). This will also install all necessary [dependencies](https://github.com/ultralytics/ultralytics/blob/main/requirements.txt).




```bash
pip install ultralytics
```
Next, install the required third-party libraries listed in requirements.txt:
```bash
pip install -r requirements.txt
```


</details>

<details open>
<summary>Usage</summary>


#### Model Training

Find the mytrain.py file in the root directory and run the following code:

```python
from ultralytics import YOLO


# Load YOLOv8 model from YAML configuration file. For each improvement to YOLOv8 model, just change the corresponding YAML config here
model = YOLO('ultralytics/cfg/models/v8/yolov8-biformer.yaml')

# Select pre-trained weights, default imports include yolov8s.pt and yolov8n.pt
model = YOLO('yolov8s.pt')

# Train YOLOv8 model
results = model.train(data='VisDrone.yaml')
```

If you want to use other versions of pre-trained weights, you can download them from [pre-trained weights](https://github.com/ultralytics/assets/releases).

#### Model Prediction

Find the mypredict.py file in the root directory and run the following code:

```python
from ultralytics import YOLO


# Load YOLOv8 model from YAML configuration file. For each improvement to YOLOv8 model, just change the corresponding YAML config here
model = YOLO('ultralytics/cfg/models/v8/yolov8-biformer.yaml')

# Select pre-trained weights, default imports include yolov8s.pt and yolov8n.pt
model = YOLO('yolov8s.pt')

# Train YOLOv8 model
results = model.train(data='VisDrone.yaml')
```
### ⭐Dataset

The dataset has already been downloaded in the datasets folder with a VisDrone subfolder containing training, testing, and validation sets.



### ⭐Notable Additions

#### YOLOv8 Architectural Modifications

The repository contains multiple YAML configuration files in `ultralytics/cfg/models/v8/` that implement various improvements to the standard YOLOv8 architecture, including:

1. **BiLevelRoutingAttention** - Adds bi-level routing attention mechanism to improve feature extraction
2. **ConvNext** - Integrates ConvNext blocks for better performance
3. **ParNetAttention** - Incorporates ParNet attention mechanisms 
4. **LSKblockAttention** - Implements Large Selective Kernel attention blocks
5. **EfficientNet** - Uses EfficientNet backbone for feature extraction
6. **VanillaNet** - Integrates VanillaNet architecture
7. **DCNv2** - Adds Deformable Convolutional Networks v2
8. **SPPFCSPC** - Modified Spatial Pyramid Pooling - Fast with CSPC
9. **ShuffleNetv2** - Uses ShuffleNetv2 as a backbone
10. **SwinTransformer** - Integrates Swin Transformer architecture

These modifications are designed to improve detection performance, especially for small objects in aerial images from the VisDrone dataset.

#### Heatmap Visualization

In the Hot-Pic folder, the hotPic.py code file allows you to generate heatmaps using various methods such as GradCAM, XGradCAM, EigenCAM, HiResCAM, etc. Below are examples of the generated heatmaps:

<div align="center">
  

![image](https://github.com/chaizwj/yolov8-tricks/assets/90506129/5ad97a66-cd79-4665-a295-938637bf3f61)


              
![image](https://github.com/chaizwj/yolov8-tricks/assets/90506129/f81eab4c-de25-4660-8d23-259e731dd5b6)



</div>



</details>

#### Custom Experiment Visualization

The Experiment-Pic folder contains two Python code files that allow you to generate different types of performance visualization charts. Below are examples of the generated result charts:

<div align="center">
  

![image](https://github.com/chaizwj/yolov8-tricks/assets/90506129/74d2aa1f-f8c5-4bbf-b38b-428276935a5c)
![image](https://github.com/chaizwj/yolov8-tricks/assets/90506129/641d063a-1c17-4544-8343-083f43d1e79b)




</div>


### ⭐Continuous Updates...
</details>
