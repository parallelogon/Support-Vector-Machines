%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code Modified From digitsdn.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;


%Loading the data
load digits; clear size
[N, dim]=size(X);
Ntest=size(Xtest1,1);
minx=min(min(X));
maxx=max(max(X));



%Add Noise to the data, use 1.0 as previous.  Can experiment with differnt
%noises

noisefactor =1.0;

noise = noisefactor*maxx; % sd for Gaussian noise


Xn = X;
for i=1:N;
    randn('state', i);
    Xn(i,:) = X(i,:) + noise*randn(1, dim);
end

Xnt = Xtest1;
for i=1:size(Xtest1,1);
    randn('state', N+i);
    Xnt(i,:) = Xtest1(i,:) + noise*randn(1,dim);
end

%
% select training set and a list of possible sig2 values
%
Xtr = X(1:1:end,:);
sig2 =dim*mean(var(Xtr)); % rule of thumb
sigmafactor = 0.7;
sig2=sig2*sigmafactor; %Default values

%List for optimization
%sig2list = 10.^[-3:1:3]*sig2;
sig2list = [0.1:0.1:(sig2*2)]; %Look over a range of plausible sig2, 0 < sig2 <= 2*sig2

%Kernel PCA
[lam,U] = kpca(Xtr,'RBF_kernel',sig2,[],'eig',240);
[lam, ids]=sort(-lam); lam = -lam; U=U(:,ids);

% linear PCA - only for comparison
%[lam_lin,U_lin] = pca(Xtr);


disp(' ');
disp(' Denoise using the first PCs');

% choose the digits for test
digs=[0:9]; ndig=length(digs);
m=2; % Choose the mth data for each digit

Xdt=zeros(ndig,dim);
Xdt0 = zeros(ndig,dim);

%
% figure of all digits
%
%


% which number of eigenvalues of kpca
npcs = [16:1:64];
lpcs = length(npcs);



for j = 1:length(sig2list)
    sig2 = sig2list(j);
    
    %Retrain for different sig2
    [lam,U] = kpca(Xtr,'RBF_kernel',sig2,[],'eig',240);
    [lam, ids]=sort(-lam); lam = -lam; U=U(:,ids);
    
    for k=1:lpcs
        nb_pcs=npcs(k);
        disp(['nb_pcs = ', num2str(nb_pcs)]);
        Ud=U(:,(1:nb_pcs)); lamd=lam(1:nb_pcs);
        
        digiterr = zeros(1,ndig);
        for i=1:ndig
            dig=digs(i);
            fprintf('digit %d : ', dig)
            xt=Xnt(i,:);
            
            Xdt(i,:) = preimage_rbf(Xtr,sig2,Ud,xt,'denoise');
            digiterr(i) = sum((Xtest1(i,:) -Xdt(i,:)).^2); %error of the i'th digit
        end % for i
        err(j,k) = mean(digiterr) %Total error for all digits using the j'th sigma and the k'th number of pcs
    end % for k
end


%This also finds the minimum error much faster
minerr = err(1,1);
[l w] = size(err);
imin = 1;
jmin = 1;
for i = 1:l
    for j = 1:w
        if err(i,j) < minerr
            minerr = err(i,j);
            imin = i;
            jmin = j;
        end
    end
end


sumNc = min(err);

%Plots for minima
plot(npcs,sumNc);
hold on;
plot(npcs,min(sumNc)*ones(1,length(sumNc)), '--b');
hold off;
title("Minimum Error by Number of Components");
xlabel("Number of PCs");
ylabel("MSE");

figure;
plot(sig2list,err(:,jmin));
hold on;
plot(sig2list,min(err(:,jmin))*ones(1,length(err(:,jmin))), '--b');
hold off;
title(sprintf("Minimum Error for %d components by \\sigma^2 value",npcs(jmin)));
xlabel("\sigma^2");
ylabel("MSE");

sig2 = sig2list(imin)
nb_pcs = npcs(jmin)

sig2_0 =dim*mean(var(Xtr)); % rule of thumb
sigmafactor = 1;

sig2_0=sig2_0*sigmafactor;

% kernel PCA for optimized sigma
[lam,U] = kpca(Xtr,'RBF_kernel',sig2,[],'eig',240);
[lam, ids]=sort(-lam); lam = -lam; U=U(:,ids);


% kernel PCA for default sigma
[lam0,U0] = kpca(Xtr,'RBF_kernel',sig2_0,[],'eig',240);
[lam0,ids0] = sort(-lam0); lam0 = -lam0; U0 = U0(:,ids0);


nb_pcs=npcs(jmin);
disp(['nb_pcs = ', num2str(nb_pcs)]);

Ud=U(:,(1:nb_pcs)); lamd=lam(1:nb_pcs);
Ud0 = U0(:,(1:nb_pcs)); lamd0 = lam0(1:nb_pcs);

H = figure;
colormap('gray');
sgtitle('Denoising using KernelPCA'); tic
e1 = 0;
e2 = 0;
for i=1:ndig
    dig=digs(i);
    fprintf('digit %d : ', dig)
    xt=Xnt(i,:);
    % plot the original clean digits
    %
    subplot(2+2, ndig, i);
    pcolor(1:15,16:-1:1,reshape(Xtest1(i,:), 15, 16)'); shading interp;
    set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);
    
    if i==1, ylabel('original'), end
    
    % plot the noisy digits
    %
    subplot(2+2, ndig, i+ndig);
    pcolor(1:15,16:-1:1,reshape(xt, 15, 16)'); shading interp;
    set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);
    if i==1, ylabel('noisy'), end
    drawnow
    
    Xdt(i,:) = preimage_rbf(Xtr,sig2,Ud,xt,'denoise');
    subplot(2+2, ndig, i+(2)*ndig);
    pcolor(1:15,16:-1:1,reshape(Xdt(i,:), 15, 16)'); shading interp;
    set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);
    
    e1(i) = sum((Xdt(i,:) - Xtest1(i,:)).^2);
    if i==1, ylabel({['n=',num2str(nb_pcs)],['\sigma^2=',num2str(sig2)]}); end
    if i == 10, yyaxis right;  set(gca,'yticklabel',[]); ylabel(['MSE = ',num2str(mean(e1))]);end
    drawnow
    
    Xdt0(i,:) = preimage_rbf(Xtr,sig2_0,Ud0,xt,'denoise');
    subplot(2+2, ndig, i+(3)*ndig);
    pcolor(1:15,16:-1:1,reshape(Xdt0(i,:), 15, 16)'); shading interp;
    set(gca,'xticklabel',[]);set(gca,'yticklabel',[]);
    
    e2(i) = sum((Xdt0(i,:) - Xtest1(i,:)).^2);
    if i==1, ylabel({['n=',num2str(nb_pcs)],['\sigma^2=',num2str(sig2_0)]}); end
    if i == 10, yyaxis right; set(gca,'yticklabel',[]); ylabel(['MSE = ',num2str(mean(e2))]);end
    drawnow
end % for i

saveas(H,"KPCA_Compare",'png')
