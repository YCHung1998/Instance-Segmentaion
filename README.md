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
├──model_saved                             # use 5 fold method saving the every best valid accuracy in each fold       
├──src                                     # some source code inside 
├──Inference.py                            # inference your testing data, generate the answer
├──main_1_train.py                         # train your model
└──README.md

```
