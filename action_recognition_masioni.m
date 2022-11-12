function action_recognition_masioni()
    
    I = imread('/Users/genevievemasioni/Downloads/sol lewitt B&W shapes.png');
    BW = imbinarize(I);
    figure
    imshowpair(I,BW,'montage')
    imwrite(BW, '/Users/genevievemasioni/Downloads/sol-lewitt-binary.jpeg')

%{
    % Loading the datasets
    [trainingSet, testSet] = loadDatasets("./Dataset/TrainSet/", "./Dataset/TestSet/");
    [labelNames] = displayDatasetsInfo(trainingSet, testSet);
    categories = categorical(labelNames);
     
    %displayExamples(trainingSet, testSet);
    saveas(gcf, './examples.png');

    % Selecting the best feature and cell sizes 
    img = readimage(trainingSet,1);
    processedImage = preprocessImage(img);
    displayProcessedImage(img, processedImage);
    saveas(gcf, './processed_image.png');
    compareFeatureExtractionParams(processedImage);
    

    hogFeatureVectors = {[2 2], [4 4], [8 8]};
    %hogFeatureVectors = {[8 8]};
    hogFeatureVectorsNames = {'2x2', '4x4', '8x8'};
    modelAccuracy = zeros(1,numel(hogFeatureVectorsNames));
    accuracyClasses = zeros(numel(hogFeatureVectorsNames),numel(labelNames));
    for i = 1:size(hogFeatureVectors,2)
        featureVector = hogFeatureVectors{i};
        % preprocess images and extract features
        [trainingFeatures, trainingLabels] = extractDatasetFeatures(trainingSet, featureVector);
        [testFeatures, testLabels] = extractDatasetFeatures(testSet, featureVector);
    
        % train and test the SVM model
        classifier = trainModel(trainingFeatures, trainingLabels);
        [confusionMatrix, accuracy, accuracyPerClass] = evaluateModel(classifier, testFeatures, testLabels, labelNames);
        modelAccuracy(i) = accuracy;
        accuracyClasses(i,:) = accuracyPerClass;

        % print confusion matrix as a heatmap
        figure;
        heatmap(categories,categories,confusionMatrix);
        filename = strjoin(["./confusionmatrix_" hogFeatureVectorsNames{i} ".png"]);
        title(filename)
        saveas(gcf, filename);
    end

    plotBarChart("Accuracy (%) relative to HOG feature vector size", ...
                hogFeatureVectorsNames, modelAccuracy, ...
                'HOG feature vector size', 'Accuracy (%)', false);
    saveas(gcf, './accuracyglobal.png');

    stackedAccuracy = [];
    for i = 1:size(accuracyClasses,1)
        classAccuracy = accuracyClasses(i,:);
        stackedAccuracy = [stackedAccuracy ; classAccuracy];
    end
       plotBarChart("Accuracy (%) per class", labelNames, stackedAccuracy, ...
                    'Class', 'Accuracy (%)', true);
        filename = "./accuracyperclass.png";
        saveas(gcf, filename);
end

function plotBarChart(name, xValues,yValues, xTitle, yTitle, showLegend)
    figure;
    names = categorical(xValues);
    names = reordercats(names, xValues);
    chart = bar(names, yValues);
    xtips = chart(1).XEndPoints;
    ytips = chart(1).YEndPoints;
    labels = string(chart(1).YData);
    text(xtips,ytips,labels);
    title(name);
    xlabel(xTitle); 
    ylabel(yTitle);
    ylim([0 100]);
    if(showLegend)
        set(chart, {'DisplayName'}, {'2x2', '4x4','8x8'}')
        legend() 
    end
end

function [] = resetWorkspace()
    clear;
    clc;
    close all;
end

function [trainingSet, testSet] = loadDatasets(trainingDir, testDir)
    trainingSet = imageDatastore(trainingDir,'IncludeSubfolders',true,'LabelSource','foldernames');
    testSet     = imageDatastore(testDir,'IncludeSubfolders',true,'LabelSource','foldernames');
end

function [] = displayExamples(trainingSet, testSet)
    figure;
    subplot(2,3,1);
    imshow(trainingSet.Files{1});
    title("Training (computer)");
    
    subplot(2,3,2);
    imshow(trainingSet.Files{61});
    title("Training (Photo)");
    
    subplot(2,3,3);
    imshow(trainingSet.Files{121});
    title("Training (Instrument)");
    
    subplot(2,3,4);
    imshow(testSet.Files{1});
    title("Test (computer)");
    
    subplot(2,3,5);
    imshow(testSet.Files{21});
    title("Test (Photo)")
    
    subplot(2,3,6);
    imshow(testSet.Files{41});
    title("Test (Instrument)");
end

function [labelNames] = displayDatasetsInfo(trainingSet, testSet)
    countTraining = countEachLabel(trainingSet)
    countTest = countEachLabel(testSet)
    labels = countTraining(:,1);
    labelNames = [];
    for i = 1:numel(labels)
        label = string(labels(i,1).Label);
        labelNames = [labelNames, label];
    end
end

function [processedImage] = preprocessImage(I)
% PREPROCESSIMAGE Preprocess an image.
%   IMG = PREPROCESSIMAGE(I) applies smoothing, resizing and histogram 
%   equalization on I.
    processedImage = imresize(I, 0.33);
    processedImage = histeq(processedImage);
    processedImage = imgaussfilt(processedImage,0.5);
    
end

function [] = displayProcessedImage(originalImage, processedImage)
    figure;
    subplot(1,2,1)
    imshow(originalImage)

    imageSize = size(originalImage);
    originalTitle = strjoin([ "Original image (" num2str(imageSize(1)) ...
                                "x" num2str(imageSize(2)) "x" ...
                                num2str(imageSize(3)) ")" ...
                            ]);
    title(originalTitle);
    
    subplot(1,2,2)
    imshow(processedImage)
    imageSize = size(processedImage);
    processedTitle = strjoin([ "Processed image (" num2str(imageSize(1)) ...
                                "x" num2str(imageSize(2)) "x" ...
                                num2str(imageSize(3)) ")" ...
                            ]);
    title(processedTitle);
end

function [] = compareFeatureExtractionParams(img)
% Code snippet from Matlab Online examples (Digit classification)
    [hog_2x2, vis2x2] = extractHOGFeatures(img,'CellSize',[2 2]);
    [hog_4x4, vis4x4] = extractHOGFeatures(img,'CellSize',[4 4]);
    [hog_8x8, vis8x8] = extractHOGFeatures(img,'CellSize',[8 8]);
    
    figure; 
    subplot(2,3,1:3); imshow(img);
    
    subplot(2,3,4);  
    plot(vis2x2); 
    title({'CellSize = [2 2]'; ['Length = ' num2str(length(hog_2x2))]});
    
    subplot(2,3,5);
    plot(vis4x4); 
    title({'CellSize = [4 4]'; ['Length = ' num2str(length(hog_4x4))]});
    
    subplot(2,3,6);
    plot(vis8x8); 
    title({'CellSize = [8 8]'; ['Length = ' num2str(length(hog_8x8))]});
end

function [datasetFeatures, datasetLabels] = extractDatasetFeatures(dataset, cellSize)
    numImages = numel(dataset.Files);
    featureSize = 100;
    datasetFeatures = zeros(numImages,featureSize,'single');
    datasetLabels = dataset.Labels;
    
    for i = 1:numImages
        img = preprocessImage(readimage(dataset,i));
        [features, ~] = extractHOGFeatures(img,'CellSize',cellSize);
        if(size(features,2) < featureSize)
            extendedFeatures = zeros(1,featureSize,'single');
            extendedFeatures(1:size(features,1),1:size(features,2)) = features;
            features = extendedFeatures;
        elseif(size(features,2) > featureSize)
            featureSize = size(features,2);
            extendedSetFeatures = zeros(numImages,featureSize,'single');
            extendedSetFeatures(1:size(datasetFeatures,1),1:size(datasetFeatures,2)) = datasetFeatures;
            datasetFeatures = extendedSetFeatures; 
        end
        datasetFeatures(i, :) = features;
    end
end

function [classifier] = trainModel(trainingFeatures, trainingLabels)
    classifier = fitcecoc(trainingFeatures, trainingLabels);
    %classifier = fitcknn(trainingFeatures, trainingLabels);
end

function [confMat, globalAccuracy, accuracyPerClass] = evaluateModel(classifier, testFeatures, testLabels, labelNames)
    predictedLabels = predict(classifier, testFeatures);
    confMat = confusionmat(testLabels, predictedLabels);

    % Compute accuracy
    correctlyClassified = 0;
    accuracyPerClass = zeros(1,numel(labelNames));
    totalPerClass = zeros(1,numel(labelNames));

    for i = 1:numel(testLabels)
        label = testLabels(i);
        predicted = predictedLabels(i);
        labelIndex = find(labelNames == string(label), 1);
        totalPerClass(labelIndex) = totalPerClass(labelIndex) + 1;
        
        if(label == predicted)
            correctlyClassified = correctlyClassified + 1;
            accuracyPerClass(labelIndex) = accuracyPerClass(labelIndex) + 1;
        end
    end
    globalAccuracy = (correctlyClassified / numel(testLabels)) * 100;

    for i = 1:numel(accuracyPerClass)
        accuracyPerClass(i) = (accuracyPerClass(i) / totalPerClass(i)) * 100;
    end

end
%}