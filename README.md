# Detecting Rotator Cuff Tears on MRI Images Through Convolutional Neural Networks
## Abstract
Rotator cuff tears are common injuries (20%general population;40%athletes (Yamamoto et al., 2010)). Undiagnosed injuries lead to chronic instability and pain. Magnetic resonance arthrograms(MRA) provide high sensitivity(≈90%) and specificity(≈95%) but are invasive, and costly. Magnetic resonance imaging(MRI) is less invasive but offers lower sensitivity(≈80%) and specificity(≈85%). This study will improve diagnostic accuracy of MRI detecting rotator cuff tears using convolutional neural networks(CNNs) and transfer learning. UNets, used for transfer learning in a musculoskeletal classification network, will demonstrate a novel methodology gaining representations of features extracted from anatomically similar data. The study will test the accuracy of transfer learning, single(grouped) vs double(successive) vs single(non-grouped) for the encoder using hip MRIs(n=>1000;NHI) and knee MRIs(n=1370;StanfordCA) to capture general structural patterns. Final models will classify healthy versus torn rotator cuffs using MRI scans confirmed by surgical operation reports, the diagnostic gold standard(n>300). All models will use PyTorch as the base library. Metrics such as sensitivity, specificity, and area under the curve(AUC) will evaluate the models’ performance. This study builds on prior work, which achieved an 69% accuracy using a 3Dimensional neural network without transfer learning (Shim et al., 2020), and hypothesizes that anatomically relevant datasets with transfer learning will yield better results. In this study we found the following improvements, our end to end training achieved an accuracy of 89% and through the usage of double transfer learning we received an accuracy of 54%.Through the use of UNets this study aims to develop a CNN-based diagnostic tool that reduces false negatives, enhances early detection, and minimizes reliance on invasive and costly procedures.
## Introduction
### Rotator Cuff Tears of the Shoulder 
Rotator Cuff Tears are common, especially among the physically active population. In the general population, about 20% will suffer a torn labrum compared to 40% of athletes who suffer the same injury. (Yamamoto et al., 2010 ) There are two common types of tears of the rotator cuff: the partial tear and the full tear. The partial tear is most common and although it is very common, the topic of how to accurately diagnose these types of tears is still controversial. Full thickness tears, typically associated with intense acute shoulder trauma, can lead to chronic instability and pain if left undiagnosed(SOURCE). Currently, there is no single diagnostic method with superior accuracy; it is generally accepted that  magnetic resonance images with an arthrogram (MRA) or a standard magnetic resonance image (MRI) read by a radiologist are the current best practices. . A meta-analysis indicated that MRA has a sensitivity of 93% and specificity of 96% for diagnosing rotator cuff tears, whereas MRI demonstrated a sensitivity of 87% and specificity of 91%.(Fanxiao et al., 2020)However, the enhanced diagnostic accuracy of MRA comes with increased patient discomfort and higher healthcare costs.
### Machine Learning Algorithms to Improve Diagnostic Accuracy
Through the use of machine learning, a convolutional neural network (CNN) could improve the diagnosis of rotator cuff tears and possibly classify them by tear type dependent solely on a MRI. These models are commonly used for medical images due to capturing simple structures in early layers but then more complex patterns in later layers. MRIs are well suited for transfer learning since the CNN trains on more commonly available datasets for the early model and beginning layers but then uses more injury-specific images for the later layers. Previously, CNNs have been utilized in musculoskeletal radiology to detect abnormalities such as knee osteoarthritis. It has been shown that a CNN model achieved approximately 97% accuracy in early detection and classification across all four Kellgren–Lawrence grades (Mahum, 2021). There are limited studies improving the diagnosis of rotator cuff tears, this is most likely due to the lack of publicly available data. In one study using a CNN for rotator cuff tears, the authors focused on the use of a 3D CNN (Shim et al., 2020). The framework utilized volumetric medical imaging data as input, allowing the model to analyze spatial features across multiple planes. By processing these 3D inputs, the CNN could capture complex patterns associated with different types and sizes of rotator cuff tears. The model's architecture using the Voxception-ResNet (VRN) enabled it to not only identify the existence of a tear but also assess its severity and precise anatomical location, which are critical factors in determining appropriate treatment strategies.

![Figure 1: Architecture of the Voxception-ResNet (VRN) used in prior study (Shim et al., 2020).](/figures/figure1.png)

Figure 1: Architecture of the Voxception-ResNet (VRN) used in prior study (Shim et al., 2020).

Through this method, the authors were able to achieve an average accuracy of 69% using LOCV (leave-one-out-cross-validation). It should be noted however that when the authors filtered their predictions to include if the result was ± 1 level of accuracy, their accuracy increased to 87.5% which is a significant improvement but was also increased with which is consistent with the diagnostic performance of the top radiologists using contrast-enhanced MR arthrography, a more costly, painful, and invasive procedure (Table 1) (Shim et al., 2020). This accuracy is a small improvement over the accuracy of radiologists when evaluating MRI images; however, the improvement is minimal and a more robust model is needed.

![Table 1: Statistics on accuracy of detecting rotator cuff tears and their severity (Shim et al., 2020).](/figures/table1.png)

Table 1: Statistics on accuracy of detecting rotator cuff tears and their severity (Shim et al., 2020).

### Transfer Learning: A solution to small datasets
Transfer learning is a machine learning technique, in which a model developed for a specific task is reused as the starting point for a new, related task, allowing the model to use prior knowledge to improve performance with less data. This method was developed and applied to medical data in an effort to counter the issue of minimal data being available for use. (Desautels, 2017) Transfer learning has been proven to increase the accuracy of medical imaging deep learning networks and has been used in radiology and dermatology, since reliable data collection is the bottleneck of developing deep learning networks in these fields. (Alzubaidi, 2021) A previous study looked at using unsupervised transfer learning and double transfer learning to increase the accuracy of a CNN in detecting diabetic foot ulcers (Alzubaidi  2021). This study focused on maximizing their F1-Score which is primarily used to determine the quality of a classification model. When trained traditionally from initialization, the CNN achieved an F1-Score of 86%, with single transfer learning the F1-Score increased to 96.25%, and with double transfer learning it they achieved an F1-Score of 99.25% (Alzubaidi, 2021)  Transfer learning has been shown to be an effective technique for using pre-trained CNN models to improve the performance of medical image analysis tasks. CNNs have demonstrated high accuracy and robustness in identifying and classifying various medical conditions from medical images. (Salehi, 2023)
 
  Given the high need to improve accuracy of rotator cuff tears, there is a need for a CNN developed using a larger data set as well as the power of using transfer learning to optimize the algorithm. This study will develop different neural networks to approach the problem of classifying and detecting rotator cuff tears. It will compare transfer learning from the ball and socket in the hip, grouped transfer learning where the encoder is trained on the hip socket as well as the muscle tear dataset, double transfer learning from the hip socket and then muscle tears in general, compared to the use of training from an initialization model (traditional learning) to classify and aid in the diagnosis process (Figure 2). Accuracy will be compared to the previous study done by Shim et al. (2020) which used a non-transfer learning 3D Voxception-ResNet. It is anticipated that using musculoskeletal structures of the hip for transfer learning will increase the accuracy and sensitivity greater than the 69% currently achieved and double transfer should increase the accuracy even more. (Figure 2) 
  
![Figure 2: Proposed hypothesis and general study design of model variations](/figures/figure2.png)

Figure 2: Proposed hypothesis and general study design of model variations

## Implications
Developing a CNN using MRI data that can more accurately detect and classify rotator cuff tears could help the 20-40% of people and athletes who suffer from a false negative diagnosis which leads to delayed interventions and severe impacts to quality of life (Hsu et al., 2015). In addition, MRA, though more sensitive, is invasive, costly, and may cause patient discomfort could be replaced by a highly accurate CNN diagnostic algorithm based on MRI. (Hsu et al., 2015) A CNN-based system could enhance the quality of life for many individuals by reducing reliance on invasive diagnostic methods and improving the accuracy of non-invasive imaging techniques. This would minimize the requirement of invasive, painful, and expensive procedures, ensure earlier detection of rotator cuff tears, and allow for more effective treatment strategies.

## Engineering Research Goal
This project aims to create a convolutional neural network (CNN) that can increase the diagnostic accuracy of rotator cuff tears using MRI images. Current methods, like regular MRIs read by radiologists, commonly miss tears or give unclear results, this leads to patients receiving high false negative rates or requires a more invasive technique (MRA). This study plans to build a CNN that performs better than these standard methods by using transfer learning from other MRI datasets (Muscle Tear MRIs and acetabular MRIs) to increase accuracy and reliability, as well as test for a double transfer learning improvement compared to traditional models. The goal of this project is to develop a tool that helps diagnose rotator cuff tears more effectively while reducing the need for invasive and costly procedures like MR arthrograms. This would mean faster, less painful, and more affordable care for patients, especially young athletes who need accurate diagnoses for timely treatment and recovery.

## Materials and Methods

### Machine Learning Types
This study proposes the usage of CNNs with transfer learning from the MRNet dataset and the hip socket in an effort to better identify rotator cuff tears of the shoulder. Different styles of machine learning will be used including transfer learning, double transfer learning, and learning from initialization. A study by Salehi et al. (2023) highlights the advantages of CNNs and transfer learning in medical imaging, noting their ability to improve accuracy and reduce the need for extensive datasets. Through the usage of transfer learning on images of the MRNet dataset, it will increase the accuracy and sensitivity of transfer learning from the hip socket MRI scans. This study also plans to evaluate four methodologies to create a diagnostic algorithm: no transfer learning (control), transfer learning labrum tears of the hip, double transfer learning of MRNet dataset to rotator cuff MRIs, and traditional training from initialization on a large data set (Figure 3). All of these methods will be performed in an unsupervised environment so that the algorithm can detect small variances on its own and have the ability to surpass the radiologist’s accuracy. 

![Fig 3: Traditional process of learning from a large dataset](/figures/figure3.png)

Fig 3: Traditional process of learning from a large dataset

This project will train and compare multiple models using the same set of MRI images, each using different machine-learning strategies to assess their effectiveness in detecting rotator cuff tears. The baseline (control) model will be the architecture proposed in Shim et al. (2020) where the algorithm using 1,924 MRI scans was trained through the VRN. Their success was a 69% accuracy when they took the entire datasets, and when they assessed their results by seeing what percentage were within 1 classification level of being correct they got an accuracy of 87%. The second model will incorporate transfer learning from hip socket MRI scans and muscle tears in general, hypothesizing that the anatomical similarities between the hip socket and the shoulder socket will yield better initial features than that found by Shim et al. (Fig. 4) A third model will utilize a single transfer learning approach using just transfer learning from the hip socket MRI data before fine-tuning on rotator cuff-specific images. This strategy aims to merge general structural insights from hip data. A final test to compare network frameworks will be using a model which was trained from initialization. This is traditional learning with a large rotator cuff dataset (Fig. 3). By comparing these models, the study will determine which transfer learning method results in the highest diagnostic accuracy, precision, recall, and F1 score. This research builds on prior work that used VRN’s trained from initialization, proposing that musculoskeletal pre-training may provide greater benefits. The findings could ultimately improve the non-invasive diagnosis of rotator cuff tears.




![Fig 4: Flow chart of proposed single transfer learning using UNet to create initial encoder through sequential double transfer learning](/figures/figure4.png)

Fig 4: Flow chart of proposed single transfer learning using UNet to create initial encoder through sequential double transfer learning

Model Architecture
In this study, all of the final models will be developed as classification to detect a healthy rotator cuff from a torn rotator cuff. All of the models will be built on the PyTorch library. They will be trained on unlabeled data and tested on gold standard scans with confirmed labral tears verified by surgical operation reports (n=y), the diagnostic gold standard. To develop an effective CNN for the diagnosis of rotator cuff tears, this study will utilize PyTorch for its flexibility when creating a CNN (Torch). The transfer learning modules will be developed through the usage of UNets (Fig. 5). These networks are commonly used for the task of image segmentation but can be used to develop an encoder. These pre-trained encoders can further be used as the transfer learning module. This practice leverages the UNet to help develop initial representations of the images before we attach a classification head to the encoder. This is an emerging method since UNets are more commonly used to segment medical images but this means that they can be used to develop representations of the images. In addition to this, the study will compare the model with a ResNet-50 architecture (traditional learning) using convolutional blocks to skip layers and change dimensionality as well as ID blocks used to skip layers and not change data dimensionality. (Fig.6) ResNet-50 has previously had great success in medical imaging, specifically in feature extraction when performing transfer learning (Ahmmed, 2023)



![Figure. 5: UNet architecture commonly used for image segmentation. (Geeks for Geeks)](/figures/figure5.png)

Figure. 5: UNet architecture commonly used for image segmentation. (Geeks for Geeks)

![Fig.6: Visual representation of ResNet50 Architecture with annotations (Mukherjee, 2022) classification](/figures/figure6.png)

Fig.6: Visual representation of ResNet50 Architecture with annotations (Mukherjee, 2022) classification

The transfer learning modules will be developed through training a UNet encoder and decoder. This process will allow the network to establish a representation of muscle tears in the encoder. This is helpful due to the classification dataset being too small which normally would cause overfitting, one potential setback to this is that the domain gap of the transfer model data to the classification dataset may be too large leading to transfer learning being ineffective. This is why training on similar anatomical locations and tears makes sense. The model will be trained, unsupervised, on (4000) muscle tear and (4000)hip socket images. This includes the Stanford ACL tear dataset (1,370 samples), the rotator cuff tear data from Kim (2,447 samples), and KneeMRI (947 samples). The hip images are obtained through the NHI database. In addition to this previous benchmark, this study will develop a single transfer learning network utilizing the  the ResNet50 architecture for the classification section and utilizing the previous approach of the UNet for the transfer model but only training on MRI scans of the hip socket for the initial layers and then transferring to the rotator cuff to test whether isolating the similar anatomical structures is superior for the transfer learning. The hip is the only other ball and socket joint which means it is the only other place we can find these patterns and structures. Using this for transfer learning should yield better results due to similar anatomical structures and features. 

For double transfer learning it is proposed that models are developed in a similar way as single transfer learning but with two encoders of unsupervised training. The initial encoder, pretrained on the muscle tears in general are designed to capture the general features of the MRI, while deeper layers will be fine-tuned on the hip MRI dataset to capture musculoskeletal-specific patterns of a ball and socket joint. The final layers will be adjusted and trained using a dataset of rotator cuff images for precise detection of tears in the rotator cuff. The first encoder will be developed through the previous strategy of using a full UNet and then removing the decoder part to be left with the UNet’s encoder; this will be trained on general muscle tears. Then transferring the encoder to train on hip MRI scans with the same approach of utilizing a UNet to receive an autoencoder. Finally transferring these two encoders to a classification head on the MRI scans of the rotator cuff. All of these will be compared to the previous architecture proposed by Shim et al.(2020) which used a VRN architecture.

In addition to this double transfer learning, the study explores the use of training from initialization (traditional training) is seen as an approach that builds all feature representations based off of the data given and does not transfer from a prior model. This method is regarded to be the most accurate method of training if you have a large enough dataset. (Khoei, 2023) To experiment with this the usage of a ResNet-50 architecture (Fig: 6) will be used and trained solely on the MRI scans of the shoulder. The one problem that is commonly encountered is overfitting the data which is very common in medical image classification due to significant lack of data. 

### Image Acquisition
The hip and shoulder datasets have come from publicly available datasets through NIH and Kaggle. Each MRI case will have two planes evaluated. Each of these planes will consist of 16 slices. These images are T2-Weighted non fat saturated. The training dataset consists of ≈8000 images without corresponding labels so we can perform unsupervised learning. The testing dataset will consist of ≈200 images and will be labeled by the operation reports from when the patient received surgery. To figure out how many test samples I would need to show an improvement in accuracy, a power analysis using a Chi-squared test was performed. Comparing the baseline accuracy of 69% from previous studies to my target accuracy of 90%. Using these numbers, calculating the difference (effect size) and setting the significance level (α) to 0.05 with a power of 0.80. The results showed that I would need about 300 samples to be confident that my model's improvement over the baseline is real and not just due to chance. This ensures my study has enough data to prove the accuracy of my model. This allows the training to take place on unlabeled images and to test on MRIs labeled through surgical findings.  

### Preprocessing
Prior to initializing the model, the data is prepared to be valid inputs and be properly augmented to increase generalization of the model. This study plans to utilize a few common techniques such as converting the MRI images from complex formats (e.g.,DICOM) to more accessible and manipulatable formats such as PNG or JPEG. Performing this conversion allows images to integrate with PyTorch seamlessly and allow for uniform processing throughout the dataset. In addition to this, standardizing the pixel values to a range between 0-1 normalizes the intensity distribution ensuring that the CNN receives consistent input across the dataset, stabilizing and accelerating training and accuracy. (Manjon, 2016)
	To further standardize inputs, images will be resized to uniform size (224 x 224 x ≈16). This step is extremely important to avoid problems with the input layer of the model, simplifying the computational requirements and allowing for batch processing. Images will also undergo center cropping to focus on the most relevant region of the shoulder joint and eliminate excess background noise that may be present in the data. (Vadmal, 2020) These approaches allow developers to guide the CNN's area of focus.
	Detecting subtle changes in the tissue is extremely difficult. To combat this, histogram equalization is used to make the subtle changes more apparent allowing for scans to be normalized across the dataset. This enables the neural network to detect the smaller changes in the tissue that is being imaged and to have comparable scans as histogram equalization allows you to bring back an image to a general light level instead of one image being at a far higher light level. (McReynolds, 2005) 
	Removing the magnetic field blurring as well as motion blur is important to allow the CNN to detect the actual structural changes rather than slight changes due to motion blur. The usage of Gaussian filters help to smooth the image by removing the random variations of pixel intensity while maintaining the important structures. (Asundi, 2002) This is important since without it there is a chance that the model will train off of the noise generated in the image and not off of the structure we are looking to classify. 


### Calculating Recall, F1-Score, and Accuracy
This study will use four main metrics to determine the overall quality of the model in the task of classifying rotator cuff tears between different models, which will be trained through different methods. These models will be evaluated on their accuracy, recall, precision, F1-Score, and the area under the curve (AUC). All of this will be tested on a testing dataset containing 5,000 unlabeled images. This test dataset will be labeled by the operation reports which is regarded in medical imaging as the gold standard of data labeling. 

Accuracy is the ratio of correct guesses to the total number of samples. This is calculated by taking the sum of TP (True Positives) and TN (True Negatives) and dividing by the total number of samples (Alzubaidi, 2021)

$`Accuracy = \frac{(TP+TN)}{(TP+TN+FP+FN)}`$

Recall is the ratio of the total number of TP to the sum of TP (True Positives) and FN (False Negatives) or the total number of positive samples in the dataset (Alzubaidi, 2021)

$`Recall = \frac{TP}{TP+FN}`$

Precision is the ratio of total true positives to the total number of positive guesses (Alzubaidi, 2021).

$`Precision = \frac{TP}{TP+FP}`$

F1-Score is the ratio between precision and recall rates
(Alzubaidi, 2021)

$`F1-Score = \frac{2Recall Precision}{Recall + Precision}`$


Receiver operating characteristic (ROC) curve is the curve of many points at different thresholds of the true positive rate (TPR) (Y-axis) and false positive rate (FPR) (X-axis). To graph this and find the curve utilizing a library such as SciKit-Learn. (Cortez, 2020)

$`TPR = \frac{TP}{(TP+FN)}`$
$`FPR = \frac{FP}{(FP+TN)}`$

After graphing the ROC curve you can calculate the area under the curve (AUC) to determine the quality of the model. The AUC is calculated by using the trapezoidal rule to determine the AUC. This can be simplified by using SciKit-learn’s built in metrics.auc() function, which will return the AUC.

$`\int_{a}^{b} f(x)\,dx`$

The comparison of the models will be based on a few factors. The accuracy of different models when it comes to classifying between a torn labrum and a healthy labrum. This will be measured by how accurately the models can sort the individual groups. What percent of testing data is in the proper groups and then what percent of each group is properly sorted. To visualize this you can utilize the area under the ROC curve (AUC) to see how the true positive rate changes in regards to the false positive rate. In general, the larger the AUC the more accurate the model is. 
Statistical Analysis

The accuracy of detecting tears for the models as a whole as well as the accuracy of detecting the type of tear from different models.
## Ethical Concerns
ADNI and Stanford AMI Shared Database
	The MRI data obtained through the ADNI database and the Stanford Artificial Intelligence in Medical Imaging (AMI) shared dataset have been brought to the highest level of de-identification while allowing them to be publicly accessible. These datasets have had all personal identification information (PII) removed from them prior to being submitted to the respective database. This means that the images are out of the scope of HIPAA rules and regulations. This ensures that they pose no risk to personal identification information being leaked and keeps the patients safe from data leakage. Nevertheless all data will be stored on secure computers and not shared with anyone who is not involved with this research study. By aligning with the terms of use outlined by ADNI and Stanford for all relevant data this study adheres to all ethical standards for secondary data usage in research. 

## Ethical Side
All research conducted will be transparent with the Yale IRB ensuring that no research conducted will follow guidelines and restrictions. The Yale IRB will be regularly updated with how the data is utilized and will provide clarification if any questions on the data usage arise. The study will take every measure possible to ensure that no personal identification information is leaked due to this study. This includes but does not limit to guaranteeing that only authorized individuals will be allowed access to the data, ensuring that any analysis of the data does not inadvertently re-identify individuals, and prioritizing that any research done on the images adheres to the strictest data protection and anonymization protocols. This study aims to provide an advancement in musculoskeletal radiology diagnosis while minimizing harm to patients by ensuring robust ethical practices are in place. The dataset will be equitable ensuring that no group is favored disproportionately without justification.


## Results (Table 2)
End to end training (Accuracy = 79%) has outperformed the predicted double transfer learning (Accuracy = 55%), previously evaluated VRN based 3D CNN (Accuracy = 69%), and Orthopedic Shoulder Specialists evaluating MRIs (Accuracy = 45.8%) (Table 2) All of thai showed that the originally thought accuracy of double transfer learning was not as successful as originally thought. This could be attributed to many factors including overfitting the transfer data or the data used for transfer learning not having enough of an anatomical similarity for accurate transfer learning. Traditionally we see increased accuracy when transfer learning is used due to the model having the ability to gain a representation before being fine tuned on data of the problem we are solving. (Desautels , 2017) In this study end to end is a possible significant improvement especially since after performing a post HOC power analysis (α = 0.01, d = 0.2, power = 0.69) indicated a minimum required sample size of n = 235.88, which was exceeded by our dataset (n = 300), suggesting results are unlikely due to chance.The performance increase (89% vs 45.8% by orthopedic specialists on a separate data set), coupled with a post hoc power analysis results, indicates a possible significant improvement


![Table 2: This studies results and accuracy compared to prior studies.](/figures/table2.png)

Table 2: This studies results and accuracy compared to prior studies.

![Graph 1: Comparing accuracy of models](/figures/graph1.png)

Graph 1: Comparing the accuracy of models

## Discussion (Graph 1)
This study revealed a lot of information into how we should approach machine learning for medical imaging. The literature says that end to end should be best when given enough data to properly train. This was supported by what was found in this study and that the increase in accuracy went from 49% in radiologist impressions to 89% in this study's end to end training. in the end to end model compared to the transfer learning models and their variants. This study found that there was an adequate quantity of data to successfully train an end to end model. Improved models could help diagnose tears in the future with greater accuracy than radiologists which would lead to reduced costs to the patients. The double transfer learning did not perform as expected regardless of manipulating the encoder’s learning rate. However, we did see that as the encoder learning rate increased, the accuracy increased as well which matches prior research and this studies hypothesis. Transfer learning data used in this study did not benefit the transfer learning process and instead it backfired and caused overfitting of the test data. Future research could look into how this model structure can be applied to other anatomical locations or other injuries at the same anatomical location such as the glenoid labrum which is a highly debated injury. This study found that for this anatomical location, the quantity of data we had was sufficient to train end to end and the Model architecture was successful to classify rotator cuffs so could theoretically be transferred to labrums. For this study the transfer data is not anatomically similar enough to get a proper representation for increasing accuracy. The increase of learning rate did in fact increase the accuracy of the models where this was a present factor. To truly understand the accuracy, this study should get the images assessed by a radiologist so there is a direct accuracy of the same dataset comparison. This will be attempted in a future study. 



# Works Cited
Alzubaidi, Laith, et al. "Novel Transfer Learning Approach for Medical Imaging with Limited Labeled Data." Cancers, vol. 13, no. 7, 30 Mar. 2021, p. 1590, https://doi.org/10.3390/cancers13071590. Accessed 8 Nov. 2024.

Asundi, Anand Krishna. "MATLAB® for Photomechanics (PMTOOLBOX)." MATLAB® for Photomechanics- a Primer, 2002, pp. 15-37, https://doi.org/10.1016/b978-008044050-7/50050-1. Accessed 10 Nov. 2024.

Bencardino, Jenny T., et al. "Superior Labrum Anterior- Posterior Lesions: Diagnosis with MR Arthrography of the Shoulder." Radiology, vol. 214, no. 1, Jan. 2000, pp. 267-71, https://doi.org/10.1148/radiology.214.1.r00ja22267. Accessed 8 Nov. 2024.

Brattain, Laura J., et al. "Machine Learning for Medical Ultrasound: Status, Methods, and Future Opportunities." Abdominal Radiology, vol. 43, no. 4, 28 Feb. 2018, pp. 786-99, https://doi.org/10.1007/s00261-018-1517-0. Accessed 8 Nov. 2024.

Bunrit, Supaporn, et al. "Evaluating on the Transfer Learning of CNN Architectures to a Construction Material Image Classification Task." International Journal of Machine Learning and Computing, vol. 9, no. 2, Apr. 2019, pp. 201-07, https://doi.org/10.18178/ijmlc.2019.9.2.787. Accessed 8 Nov. 2024.

Clymer, Daniel R., et al. "Applying Machine Learning Methods toward Classification Based on Small Datasets: Application to Shoulder Labral Tears." Journal of Engineering and Science in Medical Diagnostics and Therapy, vol. 3, no. 1, 21 Oct. 2019. The American Society of Mechanical Engineers, https://doi.org/10.1115/1.4044645. Accessed 8 Nov. 2024.

"Convert Two Grayscale Image to One 2 Channel Image in Python." Stack Overflow, Oct. 2019, stackoverflow.com/questions/58524411/convert-two-grayscale-image-to-one-2-channel-image-in-python. Accessed 8 Nov. 2024.

Desautels, Thomas, et al. "Using Transfer Learning for Improved Mortality Prediction in a Data-Scarce Hospital Setting." Biomedical Informatics Insights, vol. 9, Jan. 2017, p. 117822261771299, https://doi.org/10.1177/1178222617712994. Accessed 8 Nov. 2024.

Ho, Thao Thi, et al. "Classification of Rotator Cuff Tears in Ultrasound Images Using Deep Learning Models." Medical & Biological Engineering & Computing, vol. 60, no. 5, 18 Jan. 2022, pp. 1269-78, https://doi.org/10.1007/s11517-022-02502-6. Accessed 8 Nov. 2024.

Khan, Ali Nawaz. "Glenoid Labrum Injury Imaging." MedScape, 23 Aug. 2023, emedicine.medscape.com/article/401990-overview?form=fpf. Accessed 10 Nov. 2024.

Lev, Craig. "Definition of Convolutional Neural Network." Tech Target, 1 Jan. 2024, www.techtarget.com/searchenterpriseai/definition/convolutional-neural-network#:~:text=CNNs%20are%20especially%20useful%20for,complex%20patterns%20in%20deeper%20layers. Accessed 8 Nov. 2024.

Lin, Dana J., et al. "Deep Learning Diagnosis and Classification of Rotator Cuff Tears on Shoulder MRI." Investigative Radiology, vol. 58, no. 6, 18 Jan. 2023, pp. 405-12, https://doi.org/10.1097/rli.0000000000000951.

Lin, Weiming, et al. "Convolutional Neural Networks-Based MRI Image Analysis for the Alzheimer's Disease Prediction from Mild Cognitive Impairment." Frontiers in Neuroscience, vol. 12, 5 Nov. 2018, https://doi.org/10.3389/fnins.2018.00777. Accessed 8 Nov. 2024.

Mahum, Rabbia, et al. "A Novel Hybrid Approach Based on Deep CNN Features to Detect Knee Osteoarthritis." Sensors, vol. 21, no. 18, 15 Sept. 2021, p. 6189, https://doi.org/10.3390/s21186189. Accessed 10 Nov. 2024.

Manjón, José V. "MRI Preprocessing." Imaging Biomarkers, 3 Nov. 2016, pp. 53-63, https://doi.org/10.1007/978-3-319-43504-6_5.

Masoudi, Samira, et al. "Quick Guide on Radiology Image Pre-processing for Deep Learning Applications in Prostate Cancer Research." Journal of Medical Imaging, vol. 8, no. 01, 6 Jan. 2021, https://doi.org/10.1117/1.jmi.8.1.010901.

McREYNOLDS, Tom, and David Blythe. "Image Processing Techniques." Advanced Graphics Programming Using OpenGL, 2005, pp. 211-45, https://doi.org/10.1016/b978-155860659-3.50014-7. Accessed 10 Nov. 2024.

Mohana-Borges, Aurea V. R, et al. "Superior Labral Anteroposterior Tear: Classification and Diagnosis on MRI and MR Arthrography." American Journal of Roentgenology, vol. 181, no. 6, Dec. 2003, pp. 1449-62, https://doi.org/10.2214/ajr.181.6.1811449. Accessed 8 Nov. 2024.

Mukherejee, Suvaditya. "The Annotated ResNet-50." Towards Data Science. Medium, towardsdatascience.com/the-annotated-resnet-50-a6c536034758. Accessed 10 Nov. 2024.

Ni, Ming, et al. "A Deep Learning Approach for MRI in the Diagnosis of Labral Injuries of the Hip Joint." Journal of Magnetic Resonance Imaging, vol. 56, no. 2, 26 Jan. 2022, pp. 625-34, https://doi.org/10.1002/jmri.28069.

Salehi, Ahmad Waleed, et al. "A Study of CNN and Transfer Learning in Medical Imaging: Advantages, Challenges, Future Scope." Sustainability, vol. 15, no. 7, 29 Mar. 2023, p. 5930, https://doi.org/10.3390/su15075930. Accessed 8 Nov. 2024.

Sheridan, Kent, et al. "Accuracy of Magnetic Resonance Imaging to Diagnose Superior Labrum Anterior–posterior Tears." Knee Surgery, Sports Traumatology, Arthroscopy, vol. 23, no. 9, 2 July 2014, pp. 2645-50, https://doi.org/10.1007/s00167-014-3109-z. Accessed 10 Nov. 2024.

Shim, Eungjune, et al. "Automated Rotator Cuff Tear Classification Using 3D Convolutional Neural Network." Scientific Reports, vol. 10, no. 1, 24 Sept. 2020, https://doi.org/10.1038/s41598-020-72357-0. Accessed 8 Nov. 2024.

Smith, Toby O., et al. "A Meta-analysis of the Diagnostic Test Accuracy of MRA and MRI for the Detection of Glenoid Labral Injury." Archives of Orthopaedic and Trauma Surgery, vol. 132, no. 7, 7 Mar. 2012, pp. 905-19, https://doi.org/10.1007/s00402-012-1493-8. Accessed 10 Nov. 2024.

Solomon, Daniel J. "Editorial Commentary: Magnetic Resonance Arthrogram Is More Accurate and Precise than Conventional Magnetic Resonance Imaging for Evaluating Labral Tears after First-Time Shoulder Dislocation." Arthroscopy: The Journal of Arthroscopic & Related Surgery, vol. 40, no. 9, Sept. 2024, pp. 2370-71, https://doi.org/10.1016/j.arthro.2024.03.023. Accessed 10 Nov. 2024.

Talaei Khoei, Tala, et al. "Deep Learning: Systematic Review, Models, Challenges, and Research Directions." Neural Computing and Applications, vol. 35, no. 31, 7 Sept. 2023, pp. 23103-24, https://doi.org/10.1007/s00521-023-08957-4. Accessed 10 Nov. 2024.

Tang, Rui, et al. "Development and Clinical Application of Artificial Intelligence Assistant System for Rotator Cuff Ultrasound Scanning." Ultrasound in Medicine and Biology, vol. 50, no. 2, Feb. 2024, pp. 251-57, https://doi.org/10.1016/j.ultrasmedbio.2023.10.010.

Vadmal, Vachan, et al. "MRI Image Analysis Methods and Applications: An Algorithmic Perspective Using Brain Tumors as an Exemplar." Neuro-Oncology Advances, vol. 2, no. 1, 1 Jan. 2020, https://doi.org/10.1093/noajnl/vdaa049. Accessed 10 Nov. 2024.
Zughaib, Marc, et al. "Outcomes in Patients with Glenoid Labral Lesions: A Cohort Study." BMJ Open Sport & Exercise Medicine, vol. 2, no. 1, Feb. 2017, p. e000209, https://doi.org/10.1136/bmjsem-2016-000209. Accessed 8 Nov. 2024.
