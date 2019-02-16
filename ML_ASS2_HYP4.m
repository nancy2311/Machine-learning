clc
clear all
close all
ds = datastore('heart_DD.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',250);
T = read(ds);
m=length(T{:,1});
Alpha=0.0005;
lamda=1000;
training=ceil(0.6*m);
U=T{1:training,1:13};
Y=T{1:training,14};
X=[ones(training,1) U U.^2 U.^3 U.^4 U.^5];
n=length(X(1,:));
%Scaling of x
for w=2:n
    if max(abs(X(:,w)))~=0
    X(:,w)=(X(:,w)-mean((X(:,w))))./std(X(:,w));
    end
end
%Scaling of y
Y=T{1:training,14};%/mean(T{1:training,14});
%compute cost function and gradient
Theta=zeros(n,1);
z=X*Theta;
k=1;
hyp=1./(1+exp(-z));
E(k)=-(1/training)*sum(Y.*log(hyp)+(1-Y).*log(1-hyp));
% +(lamda/(2*m))*sum((Theta).^2); 
g=zeros(size(Theta,1),1);     
 for i=1:size(g)
     g(i)=(1/training)*sum((hyp-Y)'*X(:,i));
 end
R=1;
while R==1
Alpha=Alpha*1;
Theta=Theta-(Alpha/training)*X'*(hyp-Y);
z=X*Theta;
hyp=1./(1+exp(-z));
k=k+1;
E(k)=(-1/m)*sum(Y.*log(hyp)+(1-Y).*log(1-hyp));
% +(lamda/(2*m))*sum((Theta).^2);
if E(k-1)-E(k) <0 
    break
end 
q=(E(k-1)-E(k))./E(k-1);
if q <.0001
    R=0;
end
end

 %%%%% CV
CV=ceil(0.2*m);
U_CV=T{training+1:CV+training,1:13};
Y_CV=T{1+training:training+CV,14};
X_CV=[ones(CV,1) U_CV U_CV.^2 U_CV.^3 U_CV.^4 U_CV.^5];
n2=length(X_CV(1,:));
%Scaling of x
for w=2:n2
    if max(abs(X_CV(:,w)))~=0
    X_CV(:,w)=(X_CV(:,w)-mean((X_CV(:,w))))./std(X_CV(:,w));
    end
end
%Scaling of y
Y_CV=T{1+training:CV+training,14};%/mean(T{1+training:CV+training,14});
%compute cost function and gradient
Theta_CV=Theta;
z2=X_CV*Theta_CV;
k=1;
hyp2=1./(1+exp(-z2));
ECV(k)=-(1/CV)*sum(Y_CV.*log(hyp2)+(1-Y_CV).*log(1-hyp2));
% +(lamda/(2*CV))*sum((Theta_CV).^2); 
g2=zeros(size(Theta_CV,1),1);     
 for i=1:size(g2)
     g2(i)=(1/CV)*sum((hyp2-Y_CV)'*X_CV(:,i));
 end
% R=1;
% while R==1
% Alpha=Alpha*1;
% Theta_CV=Theta_CV-(Alpha/CV)*X_CV'*(hyp2-Y_CV);
% hyp2=1./(1+exp(-X_CV*Theta_CV)); 
% k=k+1;
% ECV(k)=(-1/CV)*sum(Y_CV.*log(hyp2)+(1-Y_CV).*log(1-hyp2));
% % +(lamda/(2*CV))*sum((Theta_CV).^2);
% if ECV(k-1)-ECV(k) <0 
%     break
% end 
% q=(ECV(k-1)-ECV(k))./ECV(k-1);
% if q <.0001
%     R=0;
% end
% end

  %%%%% TEST
U_TEST=T{training+1+CV:end,1:13};
Y_TEST=T{1+training+CV:end,14};
X_TEST=[ones(CV,1) U_TEST U_TEST.^2 U_TEST.^3 U_TEST.^4 U_TEST.^5];
n3=length(X_TEST(1,:));
%Scaling of x
for w=2:n3
    if max(abs(X_TEST(:,w)))~=0
    X_TEST(:,w)=(X_TEST(:,w)-mean((X_TEST(:,w))))./std(X_TEST(:,w));
    end
end
%compute cost function and gradient
Theta_TEST=Theta_CV;
z3=X_TEST*Theta_TEST;
k=1;
hyp3=1./(1+exp(-z3));
ETEST(k)=-(1/CV)*sum(Y_TEST.*log(hyp3)+(1-Y_TEST).*log(1-hyp3));
% +(lamda/(2*CV))*sum((Theta_TEST).^2); 
g3=zeros(size(Theta_TEST,1),1);     
 for i=1:size(g3)
     g(i)=(1/CV)*sum((hyp3-Y_TEST)'*X_TEST(:,i));
 end
% R=1;
% while R==1
% Alpha=Alpha*1;
% Theta_TEST=Theta_TEST-(Alpha/CV)*X_TEST'*(hyp3-Y_TEST);
% hyp3=1./(1+exp(-X_TEST*Theta_TEST)); 
% k=k+1;
% ETEST(k)=(-1/CV)*sum(Y_TEST.*log(hyp3)+(1-Y_TEST).*log(1-hyp3));
% % +(lamda/(2*CV))*sum((Theta_TEST).^2);
% if ETEST(k-1)-ETEST(k) <0 
%     break
% end 
% q=(ETEST(k-1)-ETEST(k))./ETEST(k-1);
% if q <.0001
%     R=0;
% end
% end
% plot(E,'k')
% hold on
% plot(ECV,'b')
% hold on
% plot(ETEST,'r')
% legend('Training set','Cross validation set','Test set')