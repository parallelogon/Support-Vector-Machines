function error = onesteperror(x,Z)
gam = x(1)
sig2 = x(2)
lag = floor(x(3))

%The idea in this procedure is to train the model up to some minimum time,
%t, calculate the predicted error for the next step ahead, then do the same
%thing for the point t+1, t+2, etc.  Eventually adding up all the errors in
%the model to get an estimation for model error.


time = lag + 1;
%Model uses lag + 1 point, t, t-1, t-2, ..., t-lag

i = 0;
err = zeros(1,length(time:(length(Z)-2)));
for t=time:(length(Z)-2)
    i = i + 1;
    Ztrain = Z(1:t+1); %To train we need 1 extra point
    Zval = Z(t+2); %Validation point is the one just after training
    
    Xu = windowize(Ztrain,1:lag+1);
    Xtra = Xu(:, 1: lag);
    Ytra = Xu(:, end);
    Xs = Ztrain(end-lag+1:end,1);
    
    %[gam,sig2] = tunelssvm({Xtra,Ytra,'f',[],[],'RBF_kernel'}, 'simplex',...    'crossvalidatelssvm',{10,'mae'});
    
    [alpha, b] = trainlssvm({Xtra,Ytra, 'f', gam, sig2, 'RBF_kernel'});
    
    prediction = predict({Xtra,Ytra,'f',gam,sig2,'RBF_kernel'},Xs, 1);
    err(i) = abs(Zval-prediction);
end
error = median(err)
end