clear;
clc;

root1 = '/home/xwhu/dataset/ISTD/train_argu/train_A/';
image_path1 = dir(fullfile(root1,'*.png'));
root3 = '/home/xwhu/dataset/ISTD/train_argu/train_C/';
image_path3 = dir(fullfile(root3,'*.png'));

save_root1 = '/home/xwhu/dataset/ISTD/train_argu/train_A_lab/';
save_root3 = '/home/xwhu/dataset/ISTD/train_argu/train_C_lab/';

colorTransform = makecform('srgb2lab');

for i=1: length(image_path1)
    i
    
    image1 = imread([root1 image_path1(i).name]);
    image3 = imread([root3 image_path3(i).name]);
    
    image1_lab = applycform(image1, colorTransform);
    image3_lab = applycform(image3, colorTransform);
    
    
    imwrite(image1_lab,[save_root1  image_path1(i).name]);
    imwrite(image3_lab,[save_root3  image_path1(i).name]);
end
