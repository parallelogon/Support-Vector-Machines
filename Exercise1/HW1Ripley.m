load ripley.mat;

group = [1 2 3 4] %[ones(1,2),2*ones(1,2)];
figure;
boxplot([Xtrain(Ytrain == 1,:),Xtrain(Ytrain == -1,:)],group,'Labels',{'Class 1 X','Class 1 Y','Class 2 X','Class 2 Y'});
title('Ripley Distribution');

figure;
scatter(Xtrain(Ytrain == 1,1),Xtrain(Ytrain == 1,2), 'blue','filled');
hold on;
scatter(Xtrain(Ytrain == -1,1),Xtrain(Ytrain == -1,2), 'red','filled');
hold off;

gamLin = tunelssvm({Xtrain, Ytrain, 'c',[],[],'lin_kernel'},'gridsearch','crossvalidatelssvm',{5,'misclass'});
[alphaLin, bLin] = trainlssvm({Xtrain, Ytrain, 'c', gamLin,[],'lin_kernel'});
[Ypred,YLatent] = simlssvm({Xtrain, Ytrain, 'c', gamLin,[],'lin_kernel'},{alphaLin,bLin},Xtest);
%Test set performance
performanceLin = sum(Ypred == Ytest)/length(Ytest);
roc(YLatent,Ytest);

[gamPoly, degree, t] = tunelssvm({Xtrain, Ytrain, 'c', [],[], 'poly_kernel'},'simplex','crossvalidatelssvm',{5,'misclass'});
[alphaPoly,bPoly] = trainlssvm({Xtrain, Ytrain, 'c', gamPoly,[degree t], 'poly_kernel'});
[YpredPoly, YlatentPoly] = simlssvm({Xtrain, Ytrain, 'c', gamPoly,[degree t], 'poly_kernel'},{alphaPoly,bPoly},Xtest);
performancePoly = sum(YpredPoly == Ytest)/length(Ytest);
roc(YlatentPoly,Ytest);
plotlssvm({Xtrain, Ytrain, 'c', gamPoly,[degree t], 'poly_kernel'});


for i = 1:1000
    [gamRBF(i), sig2(i)] = tunelssvm({Xtrain, Ytrain, 'c', [],[], 'RBF_kernel'},'simplex','crossvalidatelssvm',{10,'misclass'});
end

%We find the median results for gamma and sigma because the optimization is
%slightly unstable
gamRBF = zeros(1,1000);
sig2 = zeros(1,1000);
gRBF = median(gamRBF)
sigRBF = median(sig2)
type = 'classifier';

[alphaRBF,bRBF] = trainlssvm({Xtrain, Ytrain, 'c', gRBF,sigRBF,  'RBF_kernel'});
[YpredRBF, YlatentRBF] = simlssvm({Xtrain, Ytrain, 'c',gRBF,sigRBF, 'RBF_kernel'},{alphaRBF,bRBF},Xtest);

performanceRBF = sum(YpredRBF == Ytest)/length(Ytest);

roc(YlatentRBF,Ytest);
figure;
plotlssvm({Xtrain, Ytrain, 'c',gRBF,sigRBF, 'RBF_kernel'});



[Ppos, Pneg] = bay_modoutClass({Xtrain , Ytrain , 'c', gRBF , sigRBF, 'RBF_kernel'}, Xtest,0.5);
bay_modoutClass({Xtrain, Ytrain, 'c', gRBF,sigRBF, 'RBF_kernel'},'figure',0.5);
BayesLabel = 2*(Ppos > Pneg) - 1;

roc(Ppos, Ytest);



