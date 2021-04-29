load logmap.mat


%We first look at autocorrelation and partial autocorrelation
figure;
subplot(2,1,1);
autocorr(Z);
subplot(2,1,2);
parcorr(Z);

%Plot the data
figure;
plot(1:length(Z),Z)

%In order to find the 'best' values for gamma, sigma, and order we use
%fminsearch from matlabs optimization framework which nonparametrically
%searches for best values of the crossvalltune function, which sets up a
%rolling crossfold validation of the time series data.  Minimizes on the
%median MSE.


%Initial guesses
order = 2

X = windowize(Z, 1:( order + 1));
Y = X(:, end);
X = X(:, 1:order);

[gam0 sig20 c] = tunelssvm({X,Y,'f',[],[],'RBF_kernel'},'simplex','crossvalidatelssvm',{10,'mae'});
fun = @(b) crossvaltune(b, Z);
b_guess = [gam0 sig20 order];
guesses = fminsearch(fun, b_guess);
gam = guesses(1);
sig2 = guesses(2);
order = guesses(3); %since fminsearch gives a nonintger value we use the floor, as that is what is used by crossvaltune

order = 10;
X = windowize(Z, 1:( order + 1));
Y = X(:, end);
X = X(:, 1:order);


gam = 2.244873E01
sig2 = 8.347522
[alpha, b] = trainlssvm({X,Y,'f',gam,sig2});

%Starting values
Xstart = Z(end - order + 1:end, 1);

%Number of points to predict, size of test set
nb = 50;
prediction = predict({X,Y, 'f', gam, sig2},Xstart,nb);



figure;
hold on;
plot(Ztest, 'k');
plot(prediction, 'r');
hold off;
title(sprintf('Prediction with \\gamma = %d \\sigma^2 = %d, order = %d',gam,sig2,floor(order)))
immse(Ztest,prediction)


%We can also try a gridsearch

gamlist = [(gam-1):0.1:(gam+1)];
sig2list = [(sig2-1):0.1:(sig2+1)];
ordlist = [(order-2):1:(order+2)];

N = 10;
L = length(Z)/10;
partition = [];

for i= 1:N;
    partition = [partition Z((i - 1)*L + 1:(i*L),:)];
end

error = zeros(length(gamlist),length(sig2list),length(ordlist));
q = 0;
for gam = gamlist
    r = 0;
    q = q + 1;
    for sig2 = sig2list
        s = 0;
        r = r + 1;
        for order = ordlist
            disp(sprintf("On Combination gam: %d sig2: %d order: %d",gam,sig2,order));
            s = s + 1;
            %Make sure starting point is valid
            start = 1;
            for i = 1:10
                [l w] = size(partition(:,1:i));
                if order >= l*w
                    start = i+1;
                end
            end
            j = 0;
            err = zeros(1,length(start:9));
            for i = start:9
                j = j + 1;
                %We partition the training data according to the #of folds used,
                %increasing with each iteration.
                rollingfold = partition(:,1:i);
                rollingfold = rollingfold(:);
                X = windowize(rollingfold, 1:(order + 1));
                Y = X(:, end);
                X = X(:, 1:order);
                [alpha, b] = trainlssvm({X,Y,'f',gam,sig2});
                Xstart = rollingfold(end - order + 1:end, 1);
                nb = L;
                prediction = predict({X,Y,'f',gam,sig2},Xstart,nb);
                
                %err(j) = immse(partition(:,i+1),prediction);
                err(j) = sum(abs((partition(:,i+1)-prediction)./partition(:,i+1)));
            end
            error(q,r,s) = median(err);
        end
    end
end

minimal = min(min(min(error)))
for i= 1:length(gamlist)
    for j = 1:length(sig2list)
        for k = 1:length(ordlist)
            if error(i,j,k) == minimal
                disp(sprintf("%d %d %d",i,j,k));
                gam = gamlist(i)
                sig2 = sig2list(j)
                order = ordlist(k)
            end
        end
    end
end


X = windowize(Z, 1:( order + 1));
Y = X(:, end);
X = X(:, 1:order);


[alpha, b] = trainlssvm({X,Y,'f',gam,sig2});

%Starting values
Xstart = Z(end - order + 1:end, 1);

%Number of points to predict, size of test set
nb = 50;
prediction = predict({X,Y, 'f', gam, sig2},Xstart,nb);


figure;
hold on;
plot(Ztest, 'k');
plot(prediction, 'r');
hold off;
title(sprintf('Prediction with \\gamma = %d \\sigma^2 = %d, order = %d',gam,sig2,floor(order)))
immse(Ztest,prediction)
sum(abs(Ztest-prediction./Ztest))