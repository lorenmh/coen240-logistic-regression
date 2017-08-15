%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Lauren Howard - W1287305
% COEN 240 - Coding Assignment 3
% bayes.m

% This script uses the bayes decision theory technique to compute 3
% discriminants for the 3 classes of iris' using 10-fold. The accuracy is
% printed at the end.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc;
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
total_correct_sum = 0;
iterations_correct_vec = [];

%for k=1:K
k=1;
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

  sigms = arrayfun(@(z) sigm(z), X*w);
  class_1 = sigms > 0.5;
  class_0 = sigms <= 0.5;

  sigms(class_1) = 1;
  sigms(class_0) = 0;



  % the sum of 'correct classifications'
  current_correct_sum = 0;

  % iterate through the test data to compute accuracy
  %  % the test vector
  %  x = test_data(i,1:D);
  %  % the actual class of this example
  %  class = test_data(i,D+1);

  %  % computes the probability from the discriminants
  %  xm1 = x.' - u1;
  %  g1 = -0.5 * xm1.' * inv(cov1) * xm1;

  %  xm2 = x.' - u2;
  %  g2 = -0.5 * xm2.' * inv(cov2) * xm2;

  %  xm3 = x.' - u3;
  %  g3 = -0.5 * xm3.' * inv(cov3) * xm3;

  %  % gets the class label of the highest discriminant
  %  [_,predicted_class] = max([g1, g2, g3]);

  %  % if the class label is correct then increment the counters
  %  if predicted_class == class
  %    current_correct_sum += 1;
  %    total_correct_sum += 1;
  %  end
  %end

  % iterations_correct_vec contains the sum of correct classifications for
  % each iteration of the 10-fold
  %iterations_correct_vec = vertcat(
  %  iterations_correct_vec, 
  %  current_correct_sum
  %);

%end

%fprintf('Accuracy per iteration =\n');
%disp(iterations_correct_vec/NK);
%fprintf('Total Accuracy = %5.4f\n', total_correct_sum/N);
