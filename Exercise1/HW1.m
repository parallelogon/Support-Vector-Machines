load breast.mat;


boxplot(trainset(labels_train == 1,:));
title("Boxplot for + values");

figure;
boxplot(trainset(labels_train == -1,:));
title("Boxplot for - values");

cormap = corr(trainset);

figure;
heatmap(cormap);

[coeff, score, latent, tsquared] = pca(trainset,'NumComponents',2);

figure;
scatter(score(labels_train == 1,1),score(labels_train == 1,2),'green','filled');
xlabel("PCA 1");
ylabel("PCA 2");
hold on;
scatter(score(labels_train == -1,1), score(labels_train == -1,2),'blue','filled');
legend({"Positive Cases","Negative Cases"});
hold off;


traintest = [trainset labels_train];
grptable = grpstats(traintest, labels_train);
heatmap(grptable,'Title','Means by Variable and Classificaiton','Xlabel','Variable','Ylabel','Label','YDisplayLabels',{'No Cancer','Cancer'});

%As we can see in all of the above, the difference between the variables 4
%and 24 comprise most of the variation of the dataset and are highly
%explanatory, from the covariance matrix we see PCA is appropriate and the
%data is close to linearly seperable on two axis.  The plot of group means
%verifies this.

%For this classification task we will use different SVM kernels, linear,
%RBF, and polynomial kernels using trained parameters.

%For a Linear Classifier, degree = 1 and t = 0 or use lin_kernel

tic
gamLin = tunelssvm({trainset, labels_train, 'c',[],[],'lin_kernel'},'gridsearch','crossvalidatelssvm',{10,'misclass'});
[alphaLin, bLin] = trainlssvm({trainset, labels_train, 'c', gamLin, [],'lin_kernel'});
[Ypred,YLatent] = simlssvm({trainset, labels_train, 'c', gamLin, [],'lin_kernel'},{alphaLin,bLin},testset);
%Test set performance
tlin = toc
performanceLin = sum(Ypred == labels_test)/length(labels_test);

figure;
roc(YLatent,labels_test);

tic;
[gamPoly, tdeg] = tunelssvm({trainset, labels_train, 'c', [],[], 'poly_kernel'},'simplex','crossvalidatelssvm',{10,'misclass'});
[alphaPoly,bPoly] = trainlssvm({trainset, labels_train, 'c', gamPoly,tdeg, 'poly_kernel'});
[YpredPoly, YlatentPoly] = simlssvm({trainset, labels_train, 'c', gamPoly,tdeg, 'poly_kernel'},{alphaPoly,bPoly},testset);
tpoly = toc
performancePoly = sum(YpredPoly == labels_test)/length(labels_test);
figure;
roc(YlatentPoly,YpredPoly);

tic;
[gamRBF, sig2] = tunelssvm({trainset, labels_train, 'c', [],[], 'RBF_kernel'},'simplex','crossvalidatelssvm',{5,'misclass'});
[alphaRBF,bRBF] = trainlssvm({trainset, labels_train, 'c', gamRBF,sig2, 'RBF_kernel'});
[YpredRBF, YlatentRBF] = simlssvm({trainset, labels_train, 'c', gamRBF,sig2, 'RBF_kernel'},{alphaRBF,bRBF},testset);
trbf = toc

performanceRBF = sum(YpredRBF == labels_test)/length(labels_test);

figure;
roc(YlatentRBF,YpredRBF);

%{
[Ppos, Pneg] = bay_modoutClass({trainset , labels_train , 'c', gamRBF , sig2},testset);

BayesLabel = 2*(Ppos > Pneg) - 1
performanceBayes = sum(BayesLabel == labels_test)/length(labels_test);
%}

%Graphically showing classification on previous axis

[coefTest,scoreTest] = pca(testset, 'NumComponents',2);


bay_modoutClass({score, labels_train , 'c', gamRBF , sig2},'figure');

correctLin = scoreTest(labels_test == Ypred,:);
incorrectLin = scoreTest(labels_test ~= Ypred,:);

correctPoly = scoreTest(labels_test == YpredPoly,:);
incorrectPoly = scoreTest(labels_test ~= YpredPoly,:);

correctRBF = scoreTest(labels_test == YpredRBF,:);
incorrectRBF = scoreTest(labels_test ~= YpredRBF,:);

scatter(correctLin(:,1),correctLin(:,2), 'blue', 'filled')
hold on;
scatter(incorrectLin(:,1),incorrectLin(:,2),'red','filled')
xlabel("PCA 1");
ylabel("PCA 2");
legend({"Correctly Classified","Incorrectly Classified"});
title("Linear Kernel Results");
hold off;

scatter(correctPoly(:,1),correctPoly(:,2), 'blue','filled')
hold on;
scatter(incorrectPoly(:,1),incorrectPoly(:,2),'red','filled')
xlabel("PCA 1");
ylabel("PCA 2");
legend({"Correctly Classified","Incorrectly Classified"});
title("Polynomial Kernel Results");
hold off;

scatter(correctRBF(:,1),correctRBF(:,2), 'blue','filled')
hold on;
scatter(incorrectRBF(:,1),incorrectRBF(:,2),'red','filled')
xlabel("PCA 1");
ylabel("PCA 2");
legend({"Correctly Classified","Incorrectly Classified"});
title("RBF Kernel Results");
hold off;

heatmap([performanceLin,performancePoly,performanceRBF,performanceBayes],'Title','Performance On Test Set','Xlabel','Kernel','Ylabel','Performance','XDisplayLabels',{'Linear','Polynomical','RBF','Bayes'});