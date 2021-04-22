I = imread('testImage1.png');
y= imresize(I, [54 54]);
YTest = classify(net, y);

imshow(y)
title(string(YTest))

