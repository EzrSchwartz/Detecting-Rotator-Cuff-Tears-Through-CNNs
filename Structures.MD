# Transfer Learning

***Unsupervised UNet with Muscle tears***
```
----UNet Encoder trained on Muscle Tear datasets

    ----ResNet Classification Head attached to the Encoder of the Decoder Encoder Model
```
***Clymer Et Al Proposed study***
```
----Encoder trained on ADNI
    ----Classification head
```

***Unsupervised UNet with just Acetabular Labrum***

```
----Unet encoder trained on Acetabular Labrum 
    *Creates less of a gap between the transfer data and classification data but may have limited data*
    ----ResNet Classification Head attached to the Encoder of the Decoder Encoder Model
```
# Training From Initialization

***Unsupervised ResNet Training from Initialization***
```
----ResNet-50 to binary classification
    *Possible overfitting of data*
```

***Unsupervised Clymer Random Initialization Model***
```
----Clymer Random Initialization Model
    *Possible overfitting of data*
    *Not much backing up why this architecture other than previous paper(With 34 Samples)*
```