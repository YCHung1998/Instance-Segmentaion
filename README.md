# Instance-Segmentaion

```
Root/
   
├──dataset
    │
    ├── train (total 24 training image)
    │    ├ TCGA-18-5592-01Z-00-DX1
    │    │    ├ images 
    │    │    │    ├   TCGA-18-5592-01Z-00-DX1.png
    │    │    ├ masks
    │    │    │    ├   mask_0001.png
    │    |    │    ├ ...
    │    │    │    └── mask_0480.png 
    |    ├ TCGA-21-5784-01Z-00-DX1
    │    │    ├ images 
    │    │    │    ├   TCGA-18-5592-01Z-00-DX1.png
    │    │    ├ masks
    │    │    │    ├   mask_0001.png
    │    |    │    ├ ...
    │    │    │    └── mask_0398.png 
    │    ├ ...
    ├── test (total 6 testing images)
    │    ├   TCGA-50-5931-01Z-00-DX1.png    
    │    ├ ...
    └──  └── TCGA-G9-6348-01Z-00-DX1.png
not yet    
├──outputs                             #        
├──source                              # some source code inside 
├──Inference.py                        # inference your testing data, generate the answer
├──train.py                            # train your model
└──README.md

```

Please download the following links folder : ( folder : outputs )  
https://drive.google.com/file/d/1G7l7-YuF-idDn1s1Ucz43P4FYv1ucyI8/view?usp=sharing  
This folder will help you to implement the inference.py to denerate the answer file in the output folder.  
python inference.py --weight model_0001599.pth --outputs In_Seg-12-14-2243  
