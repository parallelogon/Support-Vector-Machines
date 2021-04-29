
X1 = randn(50,2) + 1;
X2 = randn(51,2) -1;

y1 = ones(50,1);
y2 = -ones(51,1);

%In order to make a decision boundary we note that in the case of two
%gaussian distributions with equal covariance matrices the classifier is
%found by solving the equation -||x - mu1||^2/sigma1^2 = -||x -
%mu2||^2/sigma2^2.  Since sigma1 = sigma2 and we know the mean values ([1
%1] and [-1 -1] we can solve for a straigt line y = -x, chich performs as a
%linear classifier.  It is essentally an OLS estimate so it is optimal in
%that it finds the maximum distance from the means of the distribution.  

x = [-3:.1:3];
y1 = -x;
y2 = -x -1/4*log(50/51)
figure;
hold on;
plot(X1(:,1), X1(:,2), 'ro');
plot(X2(:,1), X2(:,2), 'bo');
plot(x,y2);
hold off;

%Polynomial Kernel Data, Trying and plotting different degrees of the
%polynomial to assess performance
load iris.mat;
t = 1;
gam = 1;
degree = 1;

performance = zeros(1,10);

for degree = 1:10;
    [alpha, b] = trainlssvm({Xtrain , Ytrain , 'classifier', gam,[t degree],'poly_kernel','preprocess'});
    Yt = simlssvm({Xtrain,Ytrain,'classifier',gam,[t degree],'poly_kernel'},{alpha,b},Xtest);

    %To evaluate performance
    performance(degree) = sum(Yt == Ytest)/length(Ytest)
end

figure;
plot(1:10,performance);
xlabel("Degree of Polynomial");
ylabel("Performance");

%We see that after degree three the performance on the test set is 100% so
%we plot the first three
for d = 1:3;
    figure;
    plotlssvm({Xtrain,Ytrain,'c',1,[1 d], 'poly_kernel'})
end

%We can do the same thing with the RBF kernel, by looping over values of
%sigma squared and assessing the performance.

sig2 = [0:0.01:14];
gam = 1;
ntrials = length(sig2);
performancerbf = zeros(1,ntrials);


%In order to check our parameters we need to first have a training and
%validation set, since there is no order to the data we are able to make
%the set randomly.

idx = randperm(size(Xtrain,1));

Xttrain = Xtrain(idx(1:80),:);
Yttrain = Ytrain(idx(1:80));
Xval = Xtrain(idx(81:100),:);
Yval = Ytrain(idx(81:100));


for sigmasq = sig2
    [alpha, b] = trainlssvm({Xttrain , Yttrain , 'classifier', gam,sigmasq,'RBF_kernel'});
    Yt = simlssvm({Xttrain,Yttrain,'classifier',gam,sigmasq,'RBF_kernel'},{alpha,b},Xval);
%To evaluate performance
    performancerbf(find(sig2 == sigmasq)) = sum(Yt == Yval)/length(Yval);
end

figure;
plot(sig2,performancerbf);
xlabel("\sigma^{2}");
ylabel("Performance");

%Seeing that sigmasq is good inbetween 0.01 and up to ~10 we set it to a
%reasonably low value 1 for convenience, and plot
figure;
plotlssvm({Xtrain,Ytrain,'classifier',gam,0.01,'RBF_kernel'})
figure;
plotlssvm({Xtrain,Ytrain,'classifier',gam,1,'RBF_kernel'})
figure;
plotlssvm({Xtrain,Ytrain,'classifier',gam,12.2,'RBF_kernel'})


%we can find a good range for gamma
sigmasq = 1;
gammalist = [0.1:0.01:100];
ntrials = length(gammalist);
performancerbf = zeros(1,ntrials);
for gam = gammalist
    [alpha, b] = trainlssvm({Xttrain , Yttrain , 'classifier', gam,sigmasq,'RBF_kernel'});
    Yt = simlssvm({Xttrain,Yttrain,'classifier',gam,sigmasq,'RBF_kernel'},{alpha,b},Xval);
%To evaluate performance
    performancerbf(find(gammalist == gam)) = sum(Yt == Yval)/length(Yval);
end
plot(gammalist,performancerbf);
xlabel("\gamma");
ylabel("Performance");

gam = 5;
sigmasq = 1;
[alpha, b] = trainlssvm({Xtrain , Ytrain , 'classifier', gam,sigmasq,'RBF_kernel'});
Yt = simlssvm({Xtrain,Ytrain,'classifier',gam,sigmasq,'RBF_kernel'},{alpha,b},Xtest);
performance = sum(Yt == Ytest)/length(Ytest)
plotlssvm({Xtrain,Ytrain,'classifier',gam,sigmasq,'RBF_kernel'},{alpha,b})


gammalist = [10^(-3):10:10^3]
sigmalist = [10^(-3):10:10^3]

lgam = length(gammalist)
lsig = length(sigmalist)

perfrand = zeros(lgam,lsig)
perfcross = zeros(lgam,lsig)
perfLO = zeros(lgam,lsig)

i = 0
j = 0
for gam = gammalist
    i = i + 1;
    j = 0;
    for sigmasq = sigmalist
        j = j + 1;
        perfrand(i,j) = rsplitvalidate({Xtrain, Ytrain, 'c', gam, sigmasq, 'RBF_kernel'}, 0.80, 'misclass');
        perfcross(i,j) = crossvalidate({Xtrain, Ytrain, 'c', gam, sigmasq, 'RBF_kernel'}, 10, 'misclass');
        perfLO(i,j) = leaveoneout({Xtrain,Ytrain,'c',gam,sigmasq,'RBF_kernel'},'misclass');
    end
end

%Selecting every 10 gamma(sigma) to represent the dataset
idgamma = int8(mod(gammalist,100)) == 0;
idsigma = int8(mod(sigmalist,100)) == 0;

figure;
h1 = heatmap(sigmalist,gammalist,perfrand,'Title','Random Split Validation Error', 'XLabel','\sigma^2','YLabel','\gamma');
h1.YDisplayLabels(~idgamma) = {''};
h1.XDisplayLabels(~idsigma) = {''};

h1;
figure;
h2 = heatmap(sigmalist,gammalist,perfcross,'Title','10 Fold Cross Validation Error', 'XLabel','\sigma^2','YLabel','\gamma');
h2.XDisplayLabels(~idsigma) = {''};
h2.YDisplayLabels(~idgamma) = {''};


figure;
h3 = heatmap(sigmalist,gammalist,perfLO,'Title','Leave One Out Error', 'XLabel','\sigma^2','YLabel','\gamma');
h3.XDisplayLabels(~idsigma) = {''};
h3.YDisplayLabels(~idgamma) = {''};

tic;
N = 1000
for i = 1:1000
    [gamNM(i),sig2NM(i),costNM(i)] = tunelssvm({Xtrain,Ytrain,'c', [], [], 'RBF_kernel'}, 'simplex','crossvalidatelssvm',{10,'misclass'});
end
tnm = toc;

timeNM = tnm/1000

medNM = median(gamNM)
mugamNM = mean(gamNM)
vargamNM = std(gamNM)^2

medsig2NM = median(sig2NM)
musig2NM = mean(sig2NM)
varsig2NM = std(sig2NM)^2

mucostNM = mean(costNM)
varcostNM = std(costNM)^2

tic;
N = 1000
for i = 1:1000
    [gamG(i),sig2G(i),costG(i)] = tunelssvm({Xtrain,Ytrain,'c', [], [], 'RBF_kernel'}, 'gridsearch','crossvalidatelssvm',{10,'misclass'});
end
tgrid = toc;

timegrid = tgrid/1000

medgamG = median(gamG)
mugamG = mean(gamG)
vargamG = std(gamG)^2

medsig2G = median(sig2G)
musig2G = mean(sig2G)
varsig2G = std(sig2G)^2

mucostG = mean(costG)
varcostG = std(costG)^2

load iris
[gam,sig2,cost] = tunelssvm({Xtrain,Ytrain,'c', [], [], 'RBF_kernel'}, 'simplex','crossvalidatelssvm',{10,'misclass'});
gam = 1537.8041
sig2 = 1.4438579
[alpha, b] = trainlssvm({Xtrain , Ytrain , 'classifier', gam,sig2,'RBF_kernel'});
[Yest,Ylatent] = simlssvm({Xtrain , Ytrain , 'classifier', gam,sig2,'RBF_kernel'},{alpha,b},Xtest);
roc(Ylatent,Ytest)


%Calculating the optimial values for a bayes classifier and plotting it
type = 'classifier';
[gam, sig2] = bay_initlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});
[model, gam_opt] = bay_optimize({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'},2);
[cost_L3,sig2_opt] = bay_optimize({Xtrain,Ytrain,type,gam_opt,sig2,'RBF_kernel'},3);

bay_modoutClass ({Xtrain , Ytrain , 'c', gam_opt , sig2_opt}, 'figure');

%We can also just choose arbitrary ones
gam = gam_opt*100000;
sig2 = sig2_opt;
bay_modoutClass ({Xtrain , Ytrain , 'c', gam , sig2}, 'figure');

%Changing the values of gamma and sigma shows clearly how the gamma
%parameter affects the 'margin' of the SVM and sig2 affects how much
%'slack' is possible.  Lower values of sigma make for less 'slack'

gam = gam_opt;
sig2 = sig2_opt*10000;
bay_modoutClass ({Xtrain , Ytrain , 'c', gam , sig2}, 'figure');