# Fake Image Detection Using Color Disparities and Neural Networks

## 이미지 자리

__IVC(Image & Vision Computing) Lab / Pukyong Nat'l Univ Electronical Engineering / Busan, Republic of Korea__   
Jeonghan Lee, Hanhoon Park(Major Professor)

Paper(Korean) :    
Video(Korean) : https://cafe.naver.com/ictpaperconf/272


Abstract : In the field of deep learning, generative adversarial network (GAN) is in the spotlight as a representative image generation model, and it has already reached a point where it is difficult to classifiy real and fake (i.e., GAN-generated) iamges with the naked eye. However, based on the generation principle of GAN, a previous method could classify real and fake images with high accuracy and robustness by extracting useful features using hand-crafted filters from their chromatic components. Inspired by the previous method, we also attempt to classify real and fake imges in chromatic domains. However, we use convolutional neural networks(CNN) to extract features from images. To be specific, we try to use the transfer learning with pre-trained CNN models. Also, we try to train the PeleeNet, a type of deep CNNs, with or without a pre-processing high pass filter. The CNN-based methods are compared with the previous method. For experiments, we prepared four image datasets consisting of images generated with different image contexts and different GAN models. Extensive experimental results showed that CNN-based methods are not accurate for those whose generation GAN model and contexts are differnet from the training images, unlike the previous method showed high classification accuracy. However, we found that the previous method could be further improved by using luminance components together with the chromatic components.


## 본문 자리






__This content is inspired by the documents below :__
1. H. Li, B. Li, S. Tan, and J. Huang, "Identification of deep network generated images using disparities in color components," Signal Processing, vol. 174, 2020.
2. I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, "Generative adversarial nets," Proc. of NeurIPS, vol. 2, pp. 2672-2680, 2014.
3. T. Karras, T. Aila, S. Laine, and J. Lehtinen, "Progressive growing of GANs for improved quality, stability, and variation," Proc. of ICLR, 2018.
4. T. Karras, S. Laine, M. Aittala, J. Hellsten, J. Lehtinen, and T. Aila, "Analyzing and improving the image quality of StyleGAN," Proc. of CVPR, pp. 8110-8119, 2020.
5. R. Wang, X. Li, and C. Ling, "Pelee: a real-time object detection system on mobile devices," Proc. of NeurIPS, pp. 1967-1976, 2018.
6. F. Yu, A. Seff, Y. Zhang, S. Song, T. Funkhouser, and J. Xiao, "LSUN: construction of a large-scale image dataset using deep learning with humans in the loop," CoRR abs/1506.03365, 2015.
