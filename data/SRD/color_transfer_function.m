clear;
clc;

root1 = '/home/xwhu/dataset/SRD/test_data/shadow/';
image_path1 = dir(fullfile(root1,'*.jpg'));
root3 = '/home/xwhu/dataset/SRD/test_data/shadow_free/';
image_path3 = dir(fullfile(root3,'*.jpg'));

save_root = '/home/xwhu/dataset/SRD/test_data/shadow_free_new/';

colorTransform = makecform('srgb2lab');

for i=1: length(image_path1)
    i
    
    name1 = image_path1(i).name;
    %name2 = image_path2(i).name;
    name3 = image_path3(i).name;
    
    image1 = imread([root1 name1]);
    %mask_temp = imread([root2 name2]);
    image3 = imread([root3 name3]);
    
    image3_new = imresize(image3,[400,400]);
    %mask_temp_new = imresize(mask_temp,[400,400]);
    image1_new = imresize(image1,[400,400]);
    
    R_gt = image3_new(:,:,1);
    G_gt = image3_new(:,:,2);
    B_gt = image3_new(:,:,3);
    
    R_in = image1_new(:,:,1);
    G_in = image1_new(:,:,2);
    B_in = image1_new(:,:,3);

    image1_lab = applycform(image1_new, colorTransform);
    image3_lab = applycform(image3_new, colorTransform);
    mask_temp_new = image3_lab(:,:,1) - image1_lab(:,:,1);

    L = graythresh(mask_temp_new);
    mask = im2bw((mask_temp_new),L);
    sample = sum(sum(1-mask));
    
    
    if sum(sum(mask))<10000
        imwrite(image3,[save_root  image_path1(i).name]);
        continue;
    end
    
    
    count = 0;
    r_gt = zeros(sample,1);
    g_gt = zeros(sample,1);
    b_gt = zeros(sample,1);
    r_in = zeros(sample,1);
    g_in = zeros(sample,1);
    b_in = zeros(sample,1);
    
    for m=1:size(R_in,1)
        for n=1:size(R_in,2)
            
            if mask(m,n)==1
                continue;
            end
            count = count+1;
            
            r_gt(count,1) = R_gt(m,n);
            g_gt(count,1) = G_gt(m,n);
            b_gt(count,1) = B_gt(m,n);
            
            r_in(count,1) = R_in(m,n);
            g_in(count,1) = G_in(m,n);
            b_in(count,1) = B_in(m,n);
            
        end
    end


   X_r = [ones(size(r_gt)) r_gt g_gt b_gt];
   f_r = regress(r_in,X_r);
   f_g = regress(g_in,X_r);
   f_b = regress(b_in,X_r);
   
   for m=1:size(image3,1)
       for n=1:size(image3,2)
           
           x_gt = double([1 image3(m,n,1) image3(m,n,2) image3(m,n,3)]);
           image3(m,n,1) = x_gt*f_r;
           image3(m,n,2) = x_gt*f_g;
           image3(m,n,3) = x_gt*f_b;
       end
   end
    
    imwrite(image3,[save_root  image_path1(i).name]);
end
