f = [];
for i = 0:24999
    file_name = strcat('E:/Deep Fake Detection/png_images/StyleGAN2/FFHQ/', int2str(i), '.png');

    % 1x300 doubl type array
    if exist(file_name)
        f = vertcat(gan_img_detection_fea_hsvycc(file_name), f);
    end
end
csvwrite('E:/GAN_image_detection-master/csv/StyleGAN2/HSVYCbCr/train_features_real_all.csv', f)

f = [];
for i = 0:24999
    file_name = strcat('E:/Deep Fake Detection/png_images/StyleGAN2/StyleGAN2/', int2str(i), '.png');

    % 1x300 doubl type array
    if exist(file_name)
        f = vertcat(gan_img_detection_fea_hsvycc(file_name), f);
    end
end
csvwrite('E:/GAN_image_detection-master/csv/StyleGAN2/HSVYCbCr/train_features_fake_all.csv', f)
%train_features => 25000 Real/Fake Images