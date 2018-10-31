clear all; close all;

addpath('../matlab/');

model_root_dir = 'DSC_removal_SDR/';
definition_file = [model_root_dir 'deploy.prototxt'];
binary_file = [model_root_dir 'snapshot/' 'DSC_iter_160000_lab.caffemodel'];

assert(exist(binary_file, 'file') ~= 0);
assert(exist(definition_file, 'file') ~= 0);

caffe.reset_all();

caffe.set_mode_gpu();
caffe.set_device(0);

% Initialize a network
net = caffe.Net(definition_file, binary_file, 'test');

%root_dir = '/home/xwhu/dataset/SRD/train_argu/shadow/';
% root_dir = '/home/xwhu/dataset/ISTD/train_argu/train_A/';
%imgFiles=dir([root_dir '*.jpg']);

root_dir = '/home/xwhu/dataset/SRD/test_data/shadow/';
image_list=textread('../../data/SRD/test.txt', '%s');
%root_dir = '/home/xwhu/dataset/ISTD/test/test_A/';
%image_list=textread('../../data/ISTD/test.txt', '%s');


save_root = [model_root_dir 'result/'];

if exist(save_root, 'dir') == 0
    mkdir(save_root);
end

%nImg=length(imgFiles);
nImg=length(image_list);

imgW = 400; imgH = 400;
scale = 0.0039212686;

usedtime = 0;
show = 0;

colorTransform = makecform('srgb2lab');
colorTransform2 = makecform('lab2srgb');

for k = 1 : nImg
    
    test_image = imread([root_dir image_list{k}]);
    %test_image = imread([root_dir imgFiles(k).name]);
    
    test_image = applycform(test_image, colorTransform);
    
    if (show)
        imshow(test_image);
    end
    
    ori_size = [size(test_image,1), size(test_image,2)];
    test_image = imresize(test_image,[imgH imgW]);
    test_image = single(test_image(:,:,[3 2 1]));
    test_image = test_image.*scale;
    test_image = permute(test_image, [2 1 3]);
    
    % network forward
    tic; outputs = net.forward({test_image}); pertime=toc;
    usedtime=usedtime+pertime; avgtime=usedtime/k;
    
    res_fuse = net.blobs('upscore-fuse').get_data();
    res_global = net.blobs('res_g').get_data();
    final = (res_fuse + res_global)./2;

    
    final = permute(final, [2 1 3]);
    final = final./scale;
    final = uint8(final(:,:,[3 2 1]));
    final = imresize(final, ori_size);
    
    final = applycform(final, colorTransform2);
    
    %imwrite(final,[save_root imgFiles(k).name]);
    imwrite(final,[save_root image_list{k}]);
    
    
    if (mod(k,100)==0), fprintf('idx %i/%i, avgtime=%.4fs\n',k,nImg,avgtime); end
    
end

fprintf('idx %i/%i, avgtime=%.4fs\n',k,nImg,avgtime);
