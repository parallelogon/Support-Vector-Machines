%data = load('breast_cancer_wisconsin_data.mat','-ascii'); function_type = 'c';
%data = load('shuttle.dat','-ascii'); function_type = 'c';  data = data(1:700,:);
 data = load('california.dat','-ascii'); function_type = 'f';
 addpath('C:/Users/ZACHARY JONES/Documents/SVM/SVM course scripts updated/LSSVMlab')

X = data(:,1:end-1);
Y = data(:,end);
testX = [];
testY = [];

%Parameter for input space selection
%Please type >> help fsoperations; to get more information  

k = 6;
%function_type = 'c'; %'c' - classification, 'f' - regression 
kernel_type = 'lin_kernel'
%kernel_type = 'poly_kernel'
%kernel_type = 'RBF_kernel'
global_opt = 'csa'; % 'csa' or 'ds'

%Process to be performed
user_process={'FS-LSSVM'}%, 'SV_L0_norm'};
window = [15,20,25];

[e,s,t] = fslssvm(X,Y,k,function_type,kernel_type,global_opt,user_process,window,testX,testY);
%[e, s, t] = modsparseoperations(X,Y,k,function_type,kernel_type,user_process,window,testX,testY);