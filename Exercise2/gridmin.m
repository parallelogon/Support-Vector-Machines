%A function to minimize that picks an order, performs rolling 10 fold cross
%validation, and reports the median 'mae'
function f = gridmin(gam,sig2,order,Z)


%Partition the dataset into N equal parts to concatinate on a rolling basis
%in order to perform cross validation. In order to test order you need your
%partitions to be in size order, and you can increase the number of
%partitions as you go.
N = 10;
L = length(Z)/10;
partition = []

for i= 1:N;
    partition = [partition Z((i - 1)*L + 1:(i*L),:)]
end


%We check if the order is higher than the size of the partition, if so we
%increase the size of the starting partition.
start = 1
for i = 1:10
    [l w] = size(partition(:,1:i));
    if order >= l*w
        start = i
    end
end

%Looping through we take up to 9 folds of data to predict the next fold.

err = zeros(1,length(start:9));
j = 0;
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
    Xstart = partition(:,i);
    nb = L
    prediction = predict({X,Y,'f',gam,sig2},Xstart,nb);
    
    err(j) = immse(prediction,partition(:,i+1));
end

median(err)
end