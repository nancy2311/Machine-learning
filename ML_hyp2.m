clear all
ds = datastore('house_data_complete.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',25000);
T = read(ds);
Alpha=.01;
lamda=20;
m=length(T{:,1});
training=ceil(0.6*m);
CV=ceil(0.2*m);
U0=T{1:training,2};
U=T{1:training,4:19};
U1=T{1:training,20:21};
U2=U.^3;
U3=sqrt(U);
X=[ones(training,1) U U1 U2 U3]; %U1 U.^2 U.^3
n=length(X(1,:));
%Scaling of x
for w=2:n
    if max(abs(X(:,w)))~=0
    X(:,w)=(X(:,w)-mean((X(:,w))))./std(X(:,w));
    end
end
%Scaling of y
Y=T{1:training,3}/mean(T{1:training,3});
Theta=zeros(n,1);
k=1;

E(k)=(1/(2*(training)))*sum((X*Theta-Y).^2); %regularization
% ((lamda/2*training).*(sum(Theta.^2))

R1=1;
while R1==1
Alpha=Alpha*1;
Theta=Theta-(Alpha/(training))*X'*(X*Theta-Y);
k=k+1;
E(k)=(1/(2*(training)))*sum((X*Theta-Y).^2);
% ((lamda/2*training).*(sum(Theta.^2))
if E(k-1)-E(k)<0
    break
end 
q=(E(k-1)-E(k))./E(k-1);
if q <0.001;
    R1=0;
end
end
figure(1)
plot(E)

%%%% CROSS VALIDATION
T_cv=Theta;

U0_cv=T{training+1:training+CV,2};
U_cv=T{training+1:training+CV,4:19};
U1_cv=T{training+1:training+CV,20:21};
U2_cv=U_cv.^3;
U3_cv=sqrt(U_cv);
X_cv=[ones(CV,1) U_cv U1_cv U2_cv U3_cv]; %U1 U.^2 U.^3
Y_cv=T{training+1:training+CV,3}/mean(T{training+1:training+CV,3});
nn=length(X_cv(1,:));

for w=2:nn
    if max(abs(X_cv(:,w)))~=0
    X_cv(:,w)=(X_cv(:,w)-mean((X_cv(:,w))))./std(X_cv(:,w));
    end
end

Ecv=(1/(2*CV))*sum((X_cv*T_cv-Y_cv).^2);

%%% TEST
T_test=T_cv;

U0_test=T{training+CV+1:end,2};
U_test=T{training+CV+1:end,4:19};
U1_test=T{training+CV+1:end,20:21};
U2_test=U_test.^3;
U3_test=sqrt(U_test);
X_test=[ones(length(U3_test),1) U_test U1_test U2_test U3_test]; %U1 U.^2 U.^3
Y_test=T{training+CV+1:end,3}/mean(T{training+CV+1:end,3});
nnn=length(X_test(1,:));

for w=2:nnn
    if max(abs(X_test(:,w)))~=0
    X_test(:,w)=(X_test(:,w)-mean((X_test(:,w))))./std(X_test(:,w));
    end
end

Etest=(1/(2*CV))*sum((X_test*T_test-Y_test).^2);
