%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Lauren Howard - W1287305
% COEN 240 - Coding Assignment 3
% bayes.m

% This script uses the bayes decision theory technique to compute 3
% discriminants for the 3 classes of iris' using 10-fold. The accuracy is
% printed at the end.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc;
more off;
data_raw = dlmread('corrupted_2class_iris_dataset.dat');

[N W] = size(data_raw);
D = W-1;
K = 10;  % K-fold
NK = N/K; % Number of examples per fold
nu = 0.01;

% Randomly shuffle data
index = randperm(N);
data = data_raw(index,:);

% used for reporting output
accuracies = [];

for k=1:K
  fprintf('On fold: %d\n', k);

  % the start/end index of the fold
  test_start_index = (k-1)*NK + 1;
  test_end_index = (test_start_index+NK) - 1;

  % test data comes from the fold
  test_data = data(test_start_index:test_end_index,:);

  % train data is the non-fold data
  train_data = vertcat(
    data(1:test_start_index-1,:),
    data(test_end_index+1:N,:)
  );

  w = rand(W,1);
  features = train_data(:,1:D);
  classes = train_data(:,W);
  X = [repmat(1, length(features), 1) features];

  for itr = 1:1500
    Z = X * w;
    sigms = arrayfun(@(z) sigm(z), Z);
    errors = sigms - classes;
    delta = nu * (errors.' * X);
    w -= delta.';
  end

  test_features = test_data(:,1:D);
  test_X = [repmat(1, length(test_features), 1) test_features];
  test_classes = test_data(:,W);

  predictions = arrayfun(@(z) sigm(z), test_X*w);
  predictions(predictions > 0.5) = 1;
  predictions(predictions <= 0.5) = 0;

  accuracy = sum(test_classes == predictions) / NK;
  accuracies = vertcat(accuracies, accuracy);
end

fprintf('\nAccuracy per iteration =\n');
disp(accuracies);
fprintf('Average accuracy = %f\n', sum(accuracies)/K);
