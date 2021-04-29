function [gam sig2 lag MAPE Clist] = gridfind(X,nums,Z)%takes parameters for lags, and N repeats, and data Z
laglist = X;
N = nums;


Ztrain = Z(1:900);
Zval = Z(901:end);

gammalist = zeros(1,length(laglist));
sig2list = zeros(1,length(laglist));
Clist = zeros(1,length(laglist));

gams = zeros(1,N);
sig2s = zeros(1,N);
Cs = zeros(1,N);
j = 0;

dcets = dce(Z,4,8);
[~,d] = size(dcets);

for lag = laglist
    j = j + 1;
    %Make a dataset according to the lag
    Xu = windowize(dcets,d:lag + 1);
    Xtra = Xu(1:end-lag,d:lag);
    Ytra = Xu(1:end-lag, end);
    Xs = dcets(end - lag + 1: end, 4);

    for i = 1:N
        %Tune according to the dataset
        [gams(i) sig2s(i) c] = tunelssvm({Xtra,Ytra,'f',[],[],'RBF_kernel'},'simplex',...
            'crossvalidatelssvm',{10,'mae'});
        
        %Train and predict
        [alpha, b] = trainlssvm({Xtra, Ytra, 'f', gams(i), sig2s(i), 'RBF_kernel','original'});
        
        prediction = predict({Xtra,Ytra, 'f',gams(i),sig2s(i), 'RBF_kernel'},Xs,100);
        
        
        Cs(i) = mean(abs((Zval-prediction)./Zval)); %Calculating scale invariant MAPE error
        Cs(i) = mae(prediction-Zval); %MAE error between training and prediction on valset
    end
    
    %Take median values
    gammalist(j) = median(gams);
    sig2list(j) = median(sig2s);
    Clist(j) = median(Cs);
end

[MAPE,j] = min(Clist)
lag = laglist(j);
gam = gammalist(j);
sig2 = sig2list(j);
end