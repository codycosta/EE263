% Occlusion Sensitivity Analysis with AlexNet
% Features:
%   - Top-5 predictions per mask
%   - Heatmap of important regions
%   - Heatmap overlay on image
%   - Accuracy/probability analysis

close all; clear; clc;

% Load AlexNet
net = alexnet;
inputSize = net.Layers(1).InputSize(1:2);

% Images to test
images = {'peppers.png', 'llama.jpg'};

% Parameters
maskSize = 40;     % side length of gray square
grayValue = 128;   % mask pixel value
step = 20;         % stride for moving mask

for imgIdx = 1:numel(images)
    % --- Load and resize image ---
    I = imread(images{imgIdx});
    I = imresize(I, inputSize);
    figure, imshow(I), title(['Original - ' images{imgIdx}]);

    % --- Baseline prediction ---
    [origLabel, origScores] = classify(net, I);
    origClass = double(origLabel); % class index of baseline top-1
    fprintf('\nBaseline prediction for %s: %s\n\n', ...
        images{imgIdx}, string(origLabel));

    % --- Initialize heatmap and accuracy tracking ---
    [rows, cols, ~] = size(I);
    heatmap = zeros(floor((rows-maskSize)/step)+1, ...
                    floor((cols-maskSize)/step)+1);

    numMasks = 0;
    numCorrect = 0;
    avgProb = [];  % probability trend of baseline class

    % --- Slide mask across image ---
    rowIdx = 1;
    for r = 1:step:rows-maskSize
        colIdx = 1;
        for c = 1:step:cols-maskSize
            numMasks = numMasks + 1;

            % Apply mask
            maskedI = I;
            maskedI(r:r+maskSize-1, c:c+maskSize-1, :) = grayValue;

            % Classify masked image
            [predLabel, scores] = classify(net, maskedI);

            % --- Save for heatmap ---
            heatmap(rowIdx, colIdx) = scores(origClass);

            % --- Track accuracy ---
            if predLabel == origLabel
                numCorrect = numCorrect + 1;
            end
            avgProb(end+1) = scores(origClass);

            % --- Extra: Top-5 predictions ---
            [sortedScores, idx] = sort(scores, 'descend');
            top5Labels = net.Layers(end).ClassNames(idx(1:5));
            top5Scores = sortedScores(1:5);

            % Convert labels to cell strings for table
            top5Labels = cellstr(top5Labels(:));
            top5Scores = top5Scores(:);

            % Display table in console
            fprintf('Mask at (%d,%d):\n', r, c);
            disp(table(top5Labels, top5Scores, ...
                'VariableNames', {'Class', 'Probability'}));

            colIdx = colIdx + 1;
        end
        rowIdx = rowIdx + 1;
    end

    % --- Compute robustness accuracy ---
    maskAccuracy = numCorrect / numMasks;

    % --- Plot importance heatmap ---
    figure;
    imagesc(1 - heatmap); % invert so brighter = more important
    colormap jet;
    colorbar;
    title(['Importance map for ' images{imgIdx}]);
    xlabel('Mask column index');
    ylabel('Mask row index');

    % --- Overlay heatmap on original image ---
    heatmapResized = imresize(1 - heatmap, inputSize, 'bicubic');
    heatmapResized = mat2gray(heatmapResized);

    figure;
    imshow(I); hold on;
    h = imagesc(heatmapResized);
    colormap jet; colorbar;
    alpha(h, 0.5); % transparency
    title(['Overlay Importance Map - ' images{imgIdx}]);

    % --- Accuracy/Probability plots ---
    figure;
    subplot(1,2,1);
    plot(avgProb, '-o');
    xlabel('Mask Position Index');
    ylabel('Probability of Original Class');
    title(['Prob. Trend - ' images{imgIdx}]);

    subplot(1,2,2);
    bar(maskAccuracy*100);
    ylim([0 100]);
    ylabel('% Masks with Correct Prediction');
    title(['Mask Robustness - ' images{imgIdx}]);
end
