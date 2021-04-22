size = [54,54];

data = imageDatastore('D:\Dataset', 'IncludeSubfolders', true, 'labelsource', 'foldernames');

[dataTrain, dataValidation] = splitEachLabel(data, 0.7 ,'randomize');
Train = augmentedImageDatastore(size, dataTrain, 'ColorPreprocessing', 'none');
Val = augmentedImageDatastore(size, dataValidation, 'ColorPreprocessing', 'none');

layers = [
 imageInputLayer([size 3])
 
    convolution2dLayer(3,24,'Stride',1,'Padding',2)    
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
     
    convolution2dLayer(3,28,'Stride',1,'Padding',2)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Stride',1,'Padding',2)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,36,'Stride',1,'Padding',2)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2) 
    
   
    fullyConnectedLayer(11)     
    dropoutLayer(0.1)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
 'InitialLearnRate',0.001, ...
 'MiniBatchSize', 64, ...
 'MaxEpochs',10, ...
 'Shuffle','every-epoch', ...
 'ValidationData',Val, ...
 'ValidationFrequency',6, ...  
 'L2Regularization', 0.01, ...
 'Verbose',false, ...
 'Plots','training-progress');

net = trainNetwork(Train,layers,options);

save net
