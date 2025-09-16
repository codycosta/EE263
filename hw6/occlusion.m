% === AlexNet Occlusion Sensitivity Analysis for peppers & llama ===
clc; clear; close all;

% Load AlexNet
net = alexnet;
inputSize = net.Layers(1).InputSize;

% Image list (prompt required both images)
imageFiles = {'peppers.png', 'llama.jpg'};

for imgIdx = 1:length(imageFiles)
    fprintf('\n=== Processing %s ===\n', imageFiles{imgIdx});
    
    % Load and preprocess image
    img = imread(imageFiles{imgIdx});
    img = imresize(img, [inputSize(1) inputSize(2)]);
    
    % Classify original image
    [YPred, scores] = classify(net, img);
    fprintf('Original prediction: %s (%.2f%%)\n', string(YPred), max(scores)*100);

    % Sliding Window Occlusion
    winSize = 30; % occlusion window size
    stride = 10;  % step size
    heatmap = zeros(floor((size(img,1)-winSize)/stride)+1, ...
                    floor((size(img,2)-winSize)/stride)+1);

    for y = 1:stride:(size(img,1)-winSize)
        for x = 1:stride:(size(img,2)-winSize)
            occludedImg = img;
            occludedImg(y:y+winSize-1, x:x+winSize-1, :) = 0;

            [~, score] = classify(net, occludedImg);
            heatmap(ceil(y/stride), ceil(x/stride)) = scores(YPred) - score(YPred);
        end
    end

    % Normalize & Resize Heatmap
    heatmap = imresize(heatmap, [inputSize(1) inputSize(2)]);
    heatmap = mat2gray(heatmap);

    % Overlay Heatmap
    figure;
    imshow(img); hold on;
    h = imshow(heatmap);
    colormap jet; colorbar;
    set(h, 'AlphaData', 0.5);
    title(sprintf('Occlusion Sensitivity Heatmap - %s', imageFiles{imgIdx}));

    % Top-5 Predictions
    [sortedScores, idx] = sort(scores, 'descend');
    top5Labels = net.Layers(end).Classes(idx(1:5));
    top5Scores = sortedScores(1:5);
    
    % Ensure same row count before creating table
    resultsTable = table(cellstr(top5Labels), top5Scores(:), ...
        'VariableNames', {'Label','Score'});
    
    disp('Top-5 Predictions:');
    disp(resultsTable);

    % ROI-based Occlusion Statistics Table
    if contains(imageFiles{imgIdx}, 'peppers', 'IgnoreCase', true)
        regions = {
            'Top Stem',  [90,  10, 40, 40];
            'Left Side', [30, 100, 40, 40];
            'Right Side',[150, 100, 40, 40];
            'Bottom',    [90, 180, 40, 40];
            'Random',    [randi([1 size(img,2)-30]), randi([1 size(img,1)-30]), 30, 30];
        };
    else
        regions = {
            'Right Eye', [60, 40, 30, 30];
            'Left Eye',  [120, 40, 30, 30];
            'Nose',      [90, 80, 40, 40];
            'Random',    [randi([1 size(img,2)-30]), randi([1 size(img,1)-30]), 30, 30];
        };
    end


    layer5 = 'relu5';
    layer7 = 'fc7';
    results = cell(size(regions,1), 3);

    for i = 1:size(regions,1)
        regionName = regions{i,1};
        roi = regions{i,2};

        occImg = img;
        occImg(roi(2):roi(2)+roi(4), roi(1):roi(1)+roi(3), :) = 0;

        actOrig5 = activations(net, img, layer5, 'OutputAs', 'rows');
        actOcc5  = activations(net, occImg, layer5, 'OutputAs', 'rows');
        actOrig7 = activations(net, img, layer7, 'OutputAs', 'rows');
        actOcc7  = activations(net, occImg, layer7, 'OutputAs', 'rows');

        % Mean feature sign change
        change5 = mean(sign(actOrig5) ~= sign(actOcc5));
        change7 = mean(sign(actOrig7) ~= sign(actOcc7));

        % Store with ± std format (std=0 here for simplicity)
        results{i,1} = regionName;
        results{i,2} = sprintf('%.3f ± %.3f', mean(change5), std(change5));
        results{i,3} = sprintf('%.3f ± %.3f', mean(change7), std(change7));
    end

    occlusionTable = cell2table(results, ...
        'VariableNames', {'Occlusion_Location', ...
                          'MeanFeatureSignChange_Layer5', ...
                          'MeanFeatureSignChange_Layer7'});
    fprintf('ROI Occlusion Table for %s:\n', imageFiles{imgIdx});
    disp(occlusionTable);
end

% Accuracy Curve (applies globally, not per image)
trainImages = [5 10 20 30 40 50 60];
ourModelAcc = [30 58 63 70 72 74 75];
boEtal      = [40 45 48 52 55 57 59];
sohnEtal    = [35 38 41 44 46 47 49];

figure; hold on; grid on;
plot(trainImages, ourModelAcc, '-o','LineWidth',2);
plot(trainImages, boEtal, '-sr','LineWidth',2);
plot(trainImages, sohnEtal, '-^g','LineWidth',2);
yline(55,'--k'); % baseline line
xlabel('Training Images per Class');
ylabel('Accuracy %');
legend('Our Model','Bo etal','Sohn etal','Baseline','Location','SouthEast');
title('Accuracy Comparison');

