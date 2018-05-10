clear all; close all;

addpath('../matlab/');

model_root_dir = 'DSC/';
definition_file = [model_root_dir 'deploy.prototxt'];
binary_file = [model_root_dir 'snapshot/' 'DSC_iter_12000.caffemodel'];

assert(exist(binary_file, 'file') ~= 0);
assert(exist(definition_file, 'file') ~= 0);

caffe.reset_all();

caffe.set_mode_gpu();
caffe.set_device(0);

% Initialize a network
net = caffe.Net(definition_file, binary_file, 'test');

%root_dir = '/home/xwhu/dataset/SBU-shadow/test/';
% image_list=textread('../data/SBU/test.txt', '%s');
root_dir = '/home/xwhu/dataset/UCF-shadow/test/';
image_list=textread('../data/UCF/test.txt', '%s');


save_root = './result/';

if exist(save_root, 'dir') == 0
    mkdir(save_root);
end

%nImg=length(imgFiles);
nImg=length(image_list);

imgW = 400; imgH = 400;

avgtime = 0;
usedtime = 0;
show = 0;

for k = 1 : nImg

    test_image = imread([root_dir image_list{k}]);
    
    
    if (show)
        imshow(test_image);
    end
    
    mu = ones(1,1,3); mu(:,:,1:3) = [104.00698793,116.66876762,122.67891434];
    mu = repmat(mu,[imgH,imgW,1]);
    
    
    ori_size = [size(test_image,1), size(test_image,2)];
    test_image = imresize(test_image,[imgH imgW]);
    test_image = single(test_image(:,:,[3 2 1]));
    test_image = bsxfun(@minus,test_image,mu);
    test_image = permute(test_image, [2 1 3]);
    
    % network forward
    tic; outputs = net.forward({test_image}); pertime=toc;
    usedtime=usedtime+pertime; avgtime=usedtime/k;
    
    
    glob = net.blobs('sigmoid-global').get_data();
    fuse = net.blobs('sigmoid-fuse').get_data();
    
    final = (glob + fuse)./2;
    
    final = permute(final, [2 1 3]);
    final = imresize(final, ori_size);
    
    file_name = image_list{k};
    %imwrite(final,[save_root file_name(14:end)]);
    imwrite(final,[save_root file_name]);
    
    
    if (mod(k,100)==0), fprintf('idx %i/%i, avgtime=%.4fs\n',k,nImg,avgtime); end
    
end

fprintf('idx %i/%i, avgtime=%.4fs\n',k,nImg,avgtime);
