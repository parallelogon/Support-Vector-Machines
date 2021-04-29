%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Modification of course code
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
clear;

nb = 400 %Use default 400
sig = 0.3 %Use default 0.3

nb=nb/2;


% construct data
leng = 1;
for t=1:nb, 
  yin(t,:) = [2.*sin(t/nb*pi*leng) 2.*cos(.61*t/nb*pi*leng) (t/nb*sig)]; 
  yang(t,:) = [-2.*sin(t/nb*pi*leng) .45-2.*cos(.61*t/nb*pi*leng) (t/nb*sig)]; 
  samplesyin(t,:)  = [yin(t,1)+yin(t,3).*randn   yin(t,2)+yin(t,3).*randn];
  samplesyang(t,:) = [yang(t,1)+yang(t,3).*randn   yang(t,2)+yang(t,3).*randn];
end


% plot dataset
h=figure; hold on
plot(samplesyin(:,1),samplesyin(:,2),'o');
plot(samplesyang(:,1),samplesyang(:,2),'o');
xlabel('X_1');
ylabel('X_2');
title('Structured dataset');
disp('Press any key to continue');
pause;



sig2 = 0.4 %Defaults
approx = 'eigs' %1: 'eigs' 2:'eign'

% calculate the eigenvectors in the feature space (principal components)

%[lam,U] = kpca([samplesyin;samplesyang],'RBF_kernel',sig2,[],approx,nc);


% calculate the projections on the principal components
% Xax = -3:.1:3; Yax = -3.2:.1:3.2;
% [A,B] = meshgrid(Xax,Yax);
% grid = [reshape(A,prod(size(A)),1) reshape(B,1,prod(size(B)))'];
% k = kernel_matrix([samplesyin;samplesyang],'RBF_kernel',sig2,grid)';
% projections = k*U;

% plot the projections on the first component

% plot(samplesyin(:,1),samplesyin(:,2),'o');hold on;
% plot(samplesyang(:,1),samplesyang(:,2),'o');
% contour(Xax,Yax,reshape(projections(:,1),length(Yax),length(Xax)));
% title('Kernel PCA - Projections of the input space on the first principal component');
% figure(h);
% disp('Press any key to continue');
% pause;

% Denoise the data by minimizing the reconstruction error
Rerror = [];
for nc = 1:20
    xd = denoise_kpca([samplesyin;samplesyang],'RBF_kernel',sig2,[],approx,nc);
%h2=figure;
%To calculate Reconstruction Error
    Rerror(nc) = sum(sum((xd - [samplesyin;samplesyang]).^2))
end
%Plot the reconstruction error by index
plot(1:20,Rerror);
xlabel('nc');
ylabel('Reconstruction Error');
title('Reconstruction Error by Number of Components');

%Set the number of components to be recalculated along with minimum error
%to be plotted
nc = input('\n How many score variables? [2]'); if isempty(nc) nc=1; end;
xd = denoise_kpca([samplesyin;samplesyang],'RBF_kernel',sig2,[],approx,nc);
rerror = sum(sum((xd - [samplesyin;samplesyang]).^2));

figure;
plot(samplesyin(:,1),samplesyin(:,2),'o');
hold on;
plot(samplesyang(:,1),samplesyang(:,2),'o','Color',[189/255 55/255 139/255]);
plot(xd(:,1),xd(:,2),'r+');
hold off;
title({'Kernel PCA - Denoised datapoints in red';sprintf('Reconstruction Error - %d',rerror)});