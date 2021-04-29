%uiregress;

%demofun;

X = (-3:0.01:3)';
Y = sinc(X) + 0.1.*randn(length(X),1);

%Take even and odd points to make trainging and test set for model
%validation
Xtrain = X(1:2:end);
Ytrain = Y(1:2:end);

Xtest = X(2:2:end);
Ytest = Y(2:2:end);

%{
Try out a range of different gam and sig2 parameter values (e.g.,gam= 10,10^3,10^6 and sig2=  0.01,1,100)
  and  visualize  the  resulting  function  estimation  on  the  testset data points.  
Discuss the resulting function estimation.  
Report the mean squarederror for every combination (gam,sig2).
Do you think there is one optimal pair of hyperparameters?  Argument why (not).
Tune thegamandsig2parameters using thetunelssvmprocedure.
Use multiple runs:what can you say about the hyperparameters and the results?
Use both thesimplex and gridsearch algorithms and report differences.
%}


%We make a list of gamma and sig2 values to iterate through and evaluate in
%order to get a better idea of good parameter values.

gamlist = [10, 10^3, 10^6];
sig2list = [0.1, 1, 100];

Lg = length(gamlist);
Ls = length(sig2list);

type = 'function estimation';
perf = zeros(Lg,Ls);

pos = 0;
for i = 1:length(gamlist);
    for j = 1:length(sig2list);
        pos = pos + 1;
        
        [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gamlist(i),sig2list(j),'RBF_kernel','preprocess'});
        Yt = simlssvm({Xtrain,Ytrain,type,gamlist(i),sig2list(j),'RBF_kernel','preprocess'},{alpha,b},Xtest);
        perf(i,j) = immse(Yt,Ytest);
        
        %Visualize the results as a grid
        subplot(Lg,Ls,pos);
        scatter(Xtest,Ytest, 2, 'filled');
        hold on;
        plot(Xtest,Yt, 'Linewidth',2);
        hold off;
    end
end

%We can also visualize the mean squared error as a heatmap
figure;
heatmap(gamlist,sig2list,perf, 'XLabel','\gamma','YLabel','\sigma^2');

%We conduct multiple runs for the TUNELSSVM procedure in order to tune
%gamma and sig2, we want to know the median and the variance as well.
N = 100;
gamGrid = zeros(1,N);
sig2Grid = zeros(1,N);
costs = zeros(1,N);

tic
for i = 1:N
    [gamGrid(i),s2Grid(i),costs(i)] = tunelssvm({Xtrain,Ytrain,'f',[],[],'RBF_kernel'},'gridsearch','crossvalidatelssvm',{10,'mse'});
end
gridT = toc;

gamSym = zeros(1,N);
sig2Sym = zeros(1,N);
costSym = zeros(1,N);

tic
for i = 1:N
    [gamSym(i),sig2Sym(i),costSym(i)] = tunelssvm({Xtrain,Ytrain,'f',[],[],'RBF_kernel'},'simplex','crossvalidatelssvm',{10,'mse'});
end
symT = toc;

stats = [median(gamGrid) mean(gamGrid) std(gamGrid) median(s2Grid) mean(s2Grid) std(s2Grid) mean(costs) std(costs) gridT/N;...
    median(gamSym) mean(gamSym) std(gamSym) median(sig2Sym) mean(sig2Sym) std(sig2Sym) mean(costSym) std(costSym) symT/N];
stats
%For Bayesian ARD and robust regression
%We include a large number of points in order to visualize the dataset
N = 1000
X = 6.*rand(N,3) - 3;
Y = sinc(X(:,1)) + 0.1.*randn(N,1);
scatter3(X(:,1),X(:,2),Y,3,'filled');

%We decrease the number of points in order to perform the estimation
N = 100
X = 6.*rand(N,3) - 3;
Y = sinc(X(:,1)) + 0.1.*randn(N,1);


%Optimize hyperparameters on a validation set
Xtrain = X(1:2:end,:);
Ytrain = Y(1:2:end);

Xtest = X(2:2:end,:);
Ytest = Y(2:2:end);

gam = 1
sig2 = 1

[gam, sig2] = bay_initlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'});
[model, gam_opt] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'},2);
[cost_L3,sig2_opt] = bay_optimize({Xtrain,Ytrain,'f',gam_opt,sig2,'RBF_kernel'},3);


[selected , ranking] = bay_lssvmARD ({Xtrain, Ytrain, 'f', gam_opt , sig2_opt});

%Visualize the selected data
scatter(X(:,selected),Y,7,'filled');

%We can try to mimic this procedure using the crossvalidate option
[~,dim] = size(X);
c = nan(1,dim);
for i = 1:dim;
    c(i) = crossvalidate({X(:,i),Y,'f',gam_opt,sig2_opt,'RBF_kernel'},10,'mae','median');
end
[score,order] = sort(c)


%We now look at Robust Regression
X = (-6:0.2:6)'; %Remember for regression to place data in column format
Y = sinc(X) + 0.1.*rand(length(X),1);

%Adding outliers
out = [15 17 19];
Y(out) = 0.7 + 0.3*rand(length(out),1);
out = [41 44 46];
Y(out) = 1.5 + 0.2*rand(length(out),1);

plot(X,Y);


%Nonrobust model
costFun = 'crossvalidatelssvm'
model = initlssvm(X, Y, 'f', [], [], 'RBF_kernel');
model = tunelssvm(model, 'simplex', costFun, {10, 'mse'});
plotlssvm(model);


%Robust model comparing all of the weighting methods.
pos = 0;
err = zeros(1,4);
costFun = 'rcrossvalidatelssvm';

for wFun = {'whuber','whampel','wlogistic','wmyriad'}
    pos = pos + 1;
    model = initlssvm(X,Y,'f',[],[],'RBF_kernel');
    model = tunelssvm(model, 'simplex', costFun , {10, 'mae'}, char(wFun));
    model = robustlssvm(model);
    Yt = simlssvm(model,X);
    err(pos) = immse(Yt,Y);
    figure;
    plotlssvm(model); %here plots 1-4 correspond to the different weighting methods
end

disp(err);