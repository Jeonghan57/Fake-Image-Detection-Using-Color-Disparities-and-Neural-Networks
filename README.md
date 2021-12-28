# Fake Image Detection Using Color Disparities and Neural Networks

![색상 차이와 신경망을 이용한 위조 이미지 검출(수정)](https://user-images.githubusercontent.com/77098071/147436006-9b782228-0c04-4d70-9320-72f1fade6c68.png)
   
   
__IVC(Image & Vision Computing) Lab / Pukyong Nat'l Univ Electronic Engineering / Busan, Republic of Korea__   
Jeonghan Lee, Hanhoon Park(Major Professor)

* Paper(Korean) : *Attach the pdf file.*   
* Video(Korean) : https://cafe.naver.com/ictpaperconf/272


Abstract : In the field of deep learning, generative adversarial network (GAN) is in the spotlight as a representative image generation model, and it has already reached a point where it is difficult to classifiy real and fake (i.e., GAN-generated) iamges with the naked eye. However, based on the generation principle of GAN, a previous method could classify real and fake images with high accuracy and robustness by extracting useful features using hand-crafted filters from their chromatic components. Inspired by the previous method, we also attempt to classify real and fake imges in chromatic domains. However, we use convolutional neural networks(CNN) to extract features from images. To be specific, we try to use the transfer learning with pre-trained CNN models. Also, we try to train the PeleeNet, a type of deep CNNs, with or without a pre-processing high pass filter. The CNN-based methods are compared with the previous method. For experiments, we prepared four image datasets consisting of images generated with different image contexts and different GAN models. Extensive experimental results showed that CNN-based methods are not accurate for those whose generation GAN model and contexts are differnet from the training images, unlike the previous method showed high classification accuracy. However, we found that the previous method could be further improved by using luminance components together with the chromatic components.

## Settings
### Dataset
* Category : LSUN - cat, church_outdoor
* Real/Fake 각각 Train set 25K장 / Test set 2K장 임의추출하여 사용 -> 각 카테고리(cat, chruch_outdoor)당 54K장의 이미지 사용
* Image size : 256X256
* GAN : ProGAN, StyleGAN2

### Feature Extraction
* Li et al. : 4개(H, S, Cb, Cr)의 chromatic component 1개 당 75개의 feature 사용
* etc.) RGB, HSV, YCbCr, HSVYCbCr feature 값 구성
* Transfer Learning(Pytorch) : ResNeXt101_32x8d, AlexNet, SqueezeNet 1.1, VGG-19
* PeleeNet with HPF / without HPF   
 __=> 5 chromatic feature / 6 CNN feature__
 
 ### Classifier
 * Binary single linear layer
 * epoch = 50, batch_size = 64, learning rate = 0.001, optimizer = Adam

## Result
### Experiment 1
[Tab 1] When generative model and image context of the train data and test data are same
![image](https://user-images.githubusercontent.com/77098071/147437727-396b4f11-a77a-49aa-8926-06ab2c61d93c.png)

### Experiment 2
[Tab 2] When generative model of the train data and test data are differenet
![image](https://user-images.githubusercontent.com/77098071/147437772-d0e1d9e6-105c-48ac-8838-c1f55f3b133f.png)

### Experiment 3
[Tab 3] When image context of the train data and test data are different
![image](https://user-images.githubusercontent.com/77098071/147437816-fcbcfe31-2b52-40bc-9ff9-0d371a06d541.png)

### Experiment 4
[Tab 4] When generative model and image context of the train data and test data are different
![image](https://user-images.githubusercontent.com/77098071/147437858-127dd9e9-536b-4113-a10e-7f410ea142ba.png)

## Conclusion
 * [Tab 1]에서 보는 것처럼, 학습과 테스트 데이터의 영상 생성 모델 및 영상 컨택스트가 같으면 모든 방법이 높은 정확도로 위조 영상을 검출
 * 일부 AlexNet이나 VGG를 사용한 전이학습과 RGB 색공간에서 Li 방법을 사용하여 영상 특징을 추출하는 경우 상대적으로 정확도가 낮음
 * [Tab 2], [Tab 3], [Tab 4]에서 보는 것처럼, 학습과 테스트 데이터의 영상 생성 모델 또는 영상 컨택스트가 다르면, 일부 영상 생성 모델 차이에 대해서는 전처리 필터로 HPF를 사용하는 PeleeNet이 위조 영상 검출에 성공, but 대부분의 CNN을 사용하여 영상 특징을 추출하는 방법은 위조 영상 검출에 실패
 * 반면, Li 방법을 사용하여 영상 특징을 추출한 경우 사용된 색공간에 따라 정확도 차이는 있었으나 대부분 높은 정확도로 위조 영상 검출이 가능
 * __가장 의미있는 결과로,__ 기존 연구에서는 H, S, Cb, Cr 4개의 color 채널을 사용하는 것이 가장 효과적이라고 했으나, V나 Y 채널을 추가적으로 사용할 경우 정확도가 향상
-> H,S,V,Y,Cb,Cr 6개의 채널을 함께 사용하는 것이 가장 높은 정확도로 위조 영상을 검출.
  
  
__This content is supported by these departments below :__
* 이 성과는 '정부(과학기술정부통신부)'의 재원으로 '한국연구재단'의 지원을 받아 수행된 연구임(No. 2021R1F1A1045749).

<br/>

__This content is inspired by the documents below :__
1. H. Li, B. Li, S. Tan, and J. Huang, "Identification of deep network generated images using disparities in color components," Signal Processing, vol. 174, 2020.
2. I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, "Generative adversarial nets," Proc. of NeurIPS, vol. 2, pp. 2672-2680, 2014.
3. T. Karras, T. Aila, S. Laine, and J. Lehtinen, "Progressive growing of GANs for improved quality, stability, and variation," Proc. of ICLR, 2018.
4. T. Karras, S. Laine, M. Aittala, J. Hellsten, J. Lehtinen, and T. Aila, "Analyzing and improving the image quality of StyleGAN," Proc. of CVPR, pp. 8110-8119, 2020.
5. R. Wang, X. Li, and C. Ling, "Pelee: a real-time object detection system on mobile devices," Proc. of NeurIPS, pp. 1967-1976, 2018.
6. F. Yu, A. Seff, Y. Zhang, S. Song, T. Funkhouser, and J. Xiao, "LSUN: construction of a large-scale image dataset using deep learning with humans in the loop," CoRR abs/1506.03365, 2015.
