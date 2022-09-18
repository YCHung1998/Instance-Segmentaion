# Instance-Segmentaion
![image](https://user-images.githubusercontent.com/93210989/190901925-ffe06add-a1c9-438e-a527-8582909ed7d3.png)

Installation
---
Please following the detectron link to insatll the environment  
I'll use the linux version.
[ detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

Folder Structure
---
```
Root/
   
├──dataset
│    │
│    ├── train (total 24 training image)
│    │    ├ TCGA-18-5592-01Z-00-DX1
│    │    │    ├ images 
│    │    │    │    └── TCGA-18-5592-01Z-00-DX1.png
│    │    │    ├ masks
│    │    │    │    ├   mask_0001.png
│    │    |    │    ├ ...
│    │    │    │    └── mask_0480.png 
│    |    ├ TCGA-21-5784-01Z-00-DX1
│    │    │    ├ images 
│    │    │    │    └── TCGA-18-5592-01Z-00-DX1.png
│    │    │    ├ masks
│    │    │    │    ├   mask_0001.png
│    │    |    │    ├ ...
│    │    │    │    └── mask_0398.png 
│    │    ├ ...
│    ├── test (total 6 testing images)
│    │    ├   TCGA-50-5931-01Z-00-DX1.png    
│    │    ├ ...
│    └──  └── TCGA-G9-6348-01Z-00-DX1.png  
├──outputs                             # DOWNLOAD       
|    └── In_Seg-12-14-2243    
├──source                              # some source code inside 
├──Data_collation.py                   # First step to set up the train.json file in the datasets.
├──Init_config.py                      # Second, initial setting the config here.
├──train.py                            # Third, train your model and all the record will save in the outputs.
├                                      # Please DOWNLOAD THE OUTPUT FOLDER in the next block before you start to the inference.py !!!         
├──Inference.py                        # inference your testing data, generate the answer
└──README.md

```
Download model weight
---
Please download the following links folder : ( folder : outputs )  
https://drive.google.com/file/d/1G7l7-YuF-idDn1s1Ucz43P4FYv1ucyI8/view?usp=sharing  
This folder will help you to implement the inference.py to denerate the answer file in the output folder.  

Train
---
If you want to train by yourself, you can open the ```Init_config``` and modify by yourself. After that, run the code in terminal or environment which had been set.

```
python train.py
```

Inference
---
Please download the ouputs folder in previous blocks and follow the folder stucture.
You will see the In_Seg-12-14-2243 folder and there have ```config.yaml``` and ```model_000XXX.pth``` which you can choose.
```
python inference.py --weight model_0001599.pth --outputs In_Seg-12-14-2243  
```


Repository
---
windows  
[DGMaxime/detectron2-windows](https://github.com/DGMaxime/detectron2-windows.git)  
linux  
[facebookresearch/detectron2](https://github.com/facebookresearch/detectron2.git)  
