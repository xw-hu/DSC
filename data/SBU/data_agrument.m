clear;
clc;

ori_root = './train/';
ori_images = dir(fullfile([ori_root 'ShadowImages/'],'*.jpg'));
ori_gts = dir(fullfile([ori_root 'ShadowMasks/'],'*.png'));

save_path = './train_agru/';

for i=1: length(ori_images)
    i
    ro_degree = 0;%rand(1,1)*360-180;
    
    w_e = rand(1,1)*-0.3;
    h_e = rand(1,1)*-0.3;
    
    %  b_h = rand(1,1)*90+10;
    % b_w = rand(1,1)*90+10;
    
    image = imread([ori_root 'ShadowImages/' ori_images(i).name]);
    gt = imread([ori_root 'ShadowMasks/' ori_gts(i).name]);
    
    [h,w,~]=size(image);
    
    
    new_im = imrotate(image,ro_degree);
    new_gt = imrotate(gt,ro_degree);
    [new_h,new_w,~]=size(new_im);
    
    if (ro_degree<=45 && ro_degree>=-45) || (ro_degree>=135 && ro_degree<=-135)
        f_new_im = new_im(max(new_h/2-h/2,1):min(h/2+new_h/2,size(new_im,1)),max(new_w/2-w/2,1):min(w/2+new_w/2,size(new_im,2)),:);
        f_new_gt = new_gt(max(new_h/2-h/2,1):min(h/2+new_h/2,size(new_im,1)),max(new_w/2-w/2,1):min(w/2+new_w/2,size(new_im,2)),:);
    else
        f_new_im = new_im(max(new_w/2-w/2,1):min(w/2+new_w/2,size(new_im,1)),max(new_h/2-h/2,1):min(h/2+new_h/2,size(new_im,2)),:);
        f_new_gt = new_gt(max(new_w/2-w/2,1):min(w/2+new_w/2,size(new_im,1)),max(new_h/2-h/2,1):min(h/2+new_h/2,size(new_im,2)),:);
    end
   
%     f_new_im = image;
%     f_new_gt = gt;

    f_new_im = imresize(f_new_im, [h*(1+h_e), w*(1+w_e)]);
    f_new_gt = imresize(f_new_gt, [h*(1+h_e), w*(1+w_e)]);
    
  %   figure(2),imshow(f_new_im);
% %     
  %   figure(1),imshow(f_new_gt);
    
%% 
    imwrite(f_new_im,[save_path 'ShadowImages/a_' ori_images(i).name]);
    imwrite(f_new_gt,[save_path 'ShadowMasks/a_' ori_gts(i).name]);
    
end
