clc
clear all
close all
ds = datastore('house_prices_data_training_data.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',17999);
T = read(ds);
x=T{:,4:21};
m=length(x(1,:));
Corr_x = corr(x);
x_cov=cov(x);
K = 0;
Alpha=0.01;
lamda=0.001;
% Normalisation 
for w=1:m
    if max(abs(x(:,w)))~=0;
        x(:,w)=(x(:,w)-mean((x(:,w))))./std(x(:,w)); 
    end
end
alpha=0.5;
[eigen_vector S V]=svd(x_cov);
eigen_values=diag(S)';
for k=1:m
    alpha=1-(sum(eigen_values(1:k))/sum(eigen_values(1:m)));
    if alpha<=0.001
        break;
    end
end
U=eigen_vector;
R=U(:, 1:k)'*(x)';
app_data=U(:,1:k)*R;
error=(1/m)*(sum(app_data-x')); %DISTORTION

h=1;
Theta=zeros(m,1);
k=1;
Y=T{:,3}/mean(T{:,3});
E(k)=(1/(2*m))*sum((app_data'*Theta-Y).^2); %cost function
while h==1
    Alpha=Alpha*1;
    Theta=Theta-(Alpha/m)*app_data*(app_data'*Theta-Y);
    k=k+1;
    E(k)=(1/(2*m))*sum((app_data'*Theta-Y).^2);
    
%     Regularization
    Reg(k)=(1/(2*m))*sum((app_data'*Theta-Y).^2)+(lamda/(2*m))*sum(Theta.^2);
    %
    if E(k-1)-E(k)<0;
        break
    end
    q=(E(k-1)-E(k))./E(k-1);
    if q <.001;
        h=0;
    end
end
 X = x; %initialize dataset here
 
for K=1:10
for i=1:10
    centroids = initCentroids(X, K);
  indices = getClosestCentroids(X, centroids);
  centroids = computeCentroids(X, indices, K);
  iterations = 0;
        for ii = 1 :K
            clustering = X(find(indices == ii), :);
            J = 0;
            for z = 1 : size(clustering,1)
                J = J +(sum((clustering(z,:) - centroids(ii,:)).^2))/17999;
            end
           J_vec(1,K) = J;
            
        end
end
end
        
[ J_value k_opt ] = min(J_vec);
k_values=1:10;
plot(k_values, J_vec);

%%%%% 
X1=mean(X);
X2=std(X);
pdf=zeros(1,18);
% anomly=ones(17999,1);

for i=1:m
for j=1:17999
pdf(i,j)=normpdf(X(j,i),X1(1,i),X2(1,i));
end
end
product=prod(pdf);
for i=1:length(product)
if product(i)<0.0000000000001
    anomly(i)=0;
end
if product(i)>0.0000000000001
    anomly(i)=1;
end
end
no_of_ones=sum(anomly);