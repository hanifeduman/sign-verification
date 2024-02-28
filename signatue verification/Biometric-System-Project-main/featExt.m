imageFolderTrain='dataset/train';
imdsTrain=imageDatastore(imageFolderTrain,'LabelSource','foldernames','IncludeSubfolders',true);
imdsTrain.ReadFcn=@(filename)imageProcess(filename);

imageFolderTest='dataset/test';
imdsTest=imageDatastore(imageFolderTest,'LabelSource','foldernames','IncludeSubfolders',true);
imdsTest.ReadFcn=@(filename)imageProcess(filename);

net=resnet18;
featureLayer='fc1000';

trainingFeatures=activations(net,imdsTrain,featureLayer,"OutputAs","columns");
testFeatures=activations(net,imdsTest,featureLayer,"OutputAs","columns");

trainingLabels=imdsTrain.Labels;
testLabels=imdsTest.Labels;

classifier = fitcecoc(trainingFeatures',trainingLabels);

predictions=predict(classifier,testFeatures');

accuracy=mean(predictions==testLabels);

cmResnet18=confusionchart(testLabels,predictions);