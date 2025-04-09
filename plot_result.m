clc;clear all;
load ('predict/pre.mat');

figure;
subplot(311);plot(input1(:,1));
subplot(312);plot(input2(1,:));
subplot(313);pre(1,:)=pre(1,:)/max(pre(1,:));plot(pre(1,:));