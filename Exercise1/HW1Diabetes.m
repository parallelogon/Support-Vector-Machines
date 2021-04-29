load diabetes.mat

[nobs, ncols] = size(trainset);

x1 = trainset(labels_train == 1,:)
x2 = trainset(labels_train == -1,:)


figure;
subplot(1,2,1);
boxplot(x1);
title("Positive Labeled Points");
subplot(1,2,2);
boxplot(x2);
title("Negative Labeled Points");

cmat = corr(trainset);
figure;
heatmap(cmat);


var = sum(latent);
for i = 1:length(latent)
    cumvar(i) = sum(latent(1:i))/var
end

[coefs,score,latent] = pca(trainset, 'NumComponents',3);
scatter3(score(labels_train == 1,1),score(labels_train == 1,2),score(labels_train == 1,3), 'blue','filled');
hold on;
scatter3(score(labels_train == -1,1),score(labels_train == -1,2), score(labels_train == -1,3),'red','filled');
hold off;

%We can see no real trend here

gamLin = tunelssvm({trainset, labels_train, 'c',[],[],'lin_kernel'},'gridsearch','crossvalidatelssvm',{5,'misclass'});
[alphaLin, bLin] = trainlssvm({trainset, labels_train, 'c', gamLin,[],'lin_kernel'});
[Ypred,YLatent] = simlssvm({trainset, labels_train, 'c', gamLin,[],'lin_kernel'},{alphaLin,bLin},testset);
%Test set performance
performanceLin = sum(Ypred == labels_test)/length(labels_test);
roc(YLatent,labels_test);

N = 100
gP = zeros(1,N);
gTD = zeros(N,2);

for i = 1:N
    [gP(i), gTD(i,:)] = tunelssvm({trainset, labels_train, 'c', [],[], 'poly_kernel'},'simplex','crossvalidatelssvm',{5,'misclass'});
end

gamPoly = median(gP)
degree = median(gTD(:,2))
t = median(gTD(:,1))

[alphaPoly,bPoly] = trainlssvm({trainset, labels_train, 'c', gamPoly,[t degree], 'poly_kernel'});
[YpredPoly, YlatentPoly] = simlssvm({trainset, labels_train, 'c', gamPoly,[t degree], 'poly_kernel'},{alphaPoly,bPoly},testset);
performancePoly = sum(YpredPoly == labels_test)/length(labels_test);
figure;
roc(YlatentPoly,labels_test);

gRBF = zeros(1,100);
sRBF = zeros(1,100);
for i = 1:100
    [gRBF(i),sRBF(i)] = tunelssvm({trainset, labels_train, 'c', [],[], 'RBF_kernel'},'simplex','crossvalidatelssvm',{10,'misclass'});
end

gamRBF = median(gRBF);
sig2 = median(sRBF);

[alphaRBF,bRBF] = trainlssvm({trainset, labels_train, 'c', gamRBF,sig2, 'RBF_kernel'});
[YpredRBF, YlatentRBF] = simlssvm({trainset, labels_train, 'c', gamRBF,sig2, 'RBF_kernel'},{alphaRBF,bRBF},testset);

performanceRBF = sum(YpredRBF == labels_test)/length(labels_test);
roc(YlatentRBF,labels_test);
figure;
plotlssvm({trainset, labels_train, 'c', gamRBF,sig2, 'RBF_kernel'},{alphaRBF,bRBF});

%We can try to optimize a bayesian classifier as well

type = 'classifier';

[Ppos, Pneg] = bay_modoutClass({trainset , labels_train , 'c', gamRBF , sig2},testset);
BayesLabel = 2*(Ppos > Pneg) - 1;
performanceBayes = sum(BayesLabel == labels_test)/length(labels_test);
roc(Ppos,labels_test);