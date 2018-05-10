clear;
clc;

root1 = '/home/xwhu/dataset/ISTD/train/train_A/';
image_path1 = dir(fullfile(root1,'*.png'));
root2 = '/home/xwhu/dataset/ISTD/train/train_B/';
image_path2 = dir(fullfile(root2,'*.png'));
root3 = '/home/xwhu/dataset/ISTD/train/train_C/';
image_path3 = dir(fullfile(root3,'*.png'));

save_root = '/home/xwhu/dataset/ISTD/train/train_C_new/';

%colorTransform = makecform('srgb2lab');

for i=1: length(image_path1)
    i
    
    name1 = image_path1(i).name;
    name2 = image_path2(i).name;
    name3 = image_path3(i).name;
    
    image1 = imread([root1 name1]);
    mask = imread([root2 name2]);
    image3 = imread([root3 name3]);
    
    R_gt = image3(:,:,1);
    G_gt = image3(:,:,2);
    B_gt = image3(:,:,3);
    
    R_in = image1(:,:,1);
    G_in = image1(:,:,2);
    B_in = image1(:,:,3);

    mask = mask./255;
    sample = sum(sum(1-mask));
    
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
   
   for m=1:size(R_gt,1)
       for n=1:size(R_gt,2)
           
           x_gt = double([1 image3(m,n,1) image3(m,n,2) image3(m,n,3)]);
           image3(m,n,1) = x_gt*f_r;
           image3(m,n,2) = x_gt*f_g;
           image3(m,n,3) = x_gt*f_b;
       end
   end
    
    imwrite(image3,[save_root  image_path1(i).name]);
end
