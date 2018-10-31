clear;
clc;

root = '/home/xwhu/dataset/UCF-shadow/train_argu/';
%root = '/home/xwhu/dataset/UCF-shadow/test';

res = dir(fullfile(root,'*.jpg'));
gt = dir(fullfile(root,'*.png'));

fid = fopen('/home/xwhu/shadow/caffe-RADF-master/data/UCF/train_argu.txt','w');

for i=1:length(res)
    
   
    fprintf(fid,'%s', res(i).name);
    fprintf(fid,' %s', gt(i).name);
    fprintf(fid,'\n');
    
end

fclose(fid);