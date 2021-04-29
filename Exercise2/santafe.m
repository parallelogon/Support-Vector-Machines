load santafe

plot(Z);
figure;
autocorr(Z,'NumLags',50);

Ztrain = Z(1:900);
Zval = Z(901:end);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%First Naive analysis - lag is 50 tunelssvm gam and sig2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lag = 50;
Xu = windowize(Z,1:lag + 1);
Xtra = Xu(1:end-lag,1:lag);
Ytra = Xu(1:end-lag, end);
Xs = Z(end - lag + 1: end, 1);
nb = length(Ztest);

%Use crossvalidate instead of rcrossvalidate and MAE as opposed to MSE
[gam sig2] = tunelssvm({Xtra, Ytra, 'f', [],[], 'RBF_kernel'},'simplex',...
    'crossvalidatelssvm',{10,'mae'});


[gam sig2] = tunelssvm({Xtra, Ytra, 'f', [],[], 'RBF_kernel'},'simplex',...
    'crossvalidatelssvm',{10,'mae'});

[alpha, b] = trainlssvm({Xtra, Ytra, 'f', gam, sig2, 'RBF_kernel'});
%Recurrent Prediction
prediction = predict({Xtra,Ytra, 'f',gam,sig2, 'RBF_kernel'},Xs,nb);
MAPE = mean(abs((prediction - Ztest)./Ztest));
figure;
plot([prediction Ztest]);
title({sprintf('Prediction for SantaFe Dataset');sprintf('\\gamma = %d \\sigma^2 = %d, lag = %d, MAPE = %d',gam,sig2,lag,MAPE)})


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Better Hyperparameter optimization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%{
%Use training set 900 observations with last 100 as validation
Xu = windowize(Ztrain,1:lag + 1);
Xtra = Xu(1:end-lag,1:lag);
Ytra = Xu(1:end-lag, end);
Xs = Ztrain(end - lag + 1: end, 1);
%}
%Matlabs fminsearch with initial values given by tunelssvm as starting
%points to optimize gam, sig2, and lag.


%Works but takes a LONG TIME
[gam sig2 lag MAPE Clist] = gridfind([8:1:80],10,Z)

tgam = gam;
tsig2 = sig2;

Xu = windowize(Z,1:lag + 1);
Xtra = Xu(1:end-lag,1:lag);
Ytra = Xu(1:end-lag, end);
Xs = Z(end - lag + 1: end, 1);
nb = length(Ztest);

[gam0 sig20 c] = tunelssvm({Xtra,Ytra,'f',[],[],'RBF_kernel'},'simplex','crossvalidatelssvm',{10,'mae'});
fun = @(b) crossvaltune(b, Z);
b_guess = [gam sig2 lag];
guesses = fminsearch(fun, b_guess);
gam = guesses(1);
sig2 = guesses(2);
lag = floor(guesses(3));
tlag = lag;

gam = 2.16E03
sig2 = 2.52E01
%We can then train on the solved values, using new lag need to rearrange
%the data
lag = 39;
Xu = windowize(Z,1:lag + 1);
Xtra = Xu(1:end-lag,1:lag);
Ytra = Xu(1:end-lag, end);
Xs = Z(end - lag + 1: end, 1);
nb = length(Ztest);
[gam sig2 c] = tunelssvm({Xtra,Ytra,'f',[],[],'RBF_kernel'},'simplex','crossvalidatelssvm',{10,'mae'});
sig2 = 9/10*sig2
[alpha, b] = trainlssvm({Xtra, Ytra, 'f', gam, sig2, 'RBF_kernel','original'});

%Recurrent Prediction

prediction = predict({Xtra,Ytra, 'f',gam,sig2, 'RBF_kernel'},Xs,nb);
MAPE = mean(abs((prediction - Ztest)./Ztest));
plot([prediction Ztest]);
title({sprintf('Prediction for SantaFe Dataset');sprintf('\\gamma = %d \\sigma^2 = %d, lag = %d, MAPE = %d',gam,sig2,lag,MAPE)})

%We can compare this to a gridsearch around the optimal lag of 42 found
%earlier

[gam sig2 lag MAPE] = gridfind([38:1:39],3,Z);

%Then we test again
Xu = windowize(Z,1:lag + 1);
Xtra = Xu(1:end-lag,1:lag);
Ytra = Xu(1:end-lag, end);
Xs = Z(end - lag + 1: end, 1);
nb = length(Ztest);


[alpha, b] = trainlssvm({Xtra, Ytra, 'f', gam, sig2, 'RBF_kernel'});

%Recurrent Prediction
prediction = predict({Xtra,Ytra, 'f',gam,sig2, 'RBF_kernel'},Xs,nb);
plot([prediction Ztest]);
title({sprintf('Prediction for SantaFe Dataset');sprintf('\\gamma = %d \\sigma^2 = %d, lag = %d, MAE = %d',gam,sig2,lag,MAPE)})