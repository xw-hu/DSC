clear;
clc;

%root = '/home/xwhu/dataset/SBU-shadow/train/';
root = '/home/xwhu/dataset/SBU-shadow/test/';

res = dir(fullfile(strcat(root,'ShadowImages'),'*.jpg'));
gt = dir(fullfile(strcat(root,'ShadowMasks'),'*.png'));

fid = fopen('/home/xwhu/shadow/caffe-RADF-master/data/SBU/test.txt','w');

for i=1:length(res)
    
   
    fprintf(fid,'%s/%s', 'ShadowImages', res(i).name);
    %fprintf(fid,' %s/%s', 'ShadowMasks', gt(i).name);
    fprintf(fid,'\n');
    
end

fclose(fid);