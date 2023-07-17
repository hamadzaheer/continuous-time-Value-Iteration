%% Reinforcement Learning | Optimal State Feedback Control

% Implementation of model-free Value Iteration algorithm to design optimal
% state feedback controller for continuous-time LTI systems
% Referenece: bian2016_Value iteration and adaptive dynamic programming for data-driven adaptive optimal control design
% Example 6.1: Application to power system control

% System Parameters
D = 5;
H = 4;
omega_0 = 314.159;
T_d0 = 2.4232;
delta_0 = 47*pi/180;
P_m0 = 0.45;
A =[0 1 0;
    0 -D/(2*H) -omega_0/(2*H);
    0 0 -1/T_d0];
B=[0 0 1/T_d0]';
[n,m]=size(B);
% Equilibrium point
x_eq=[delta_0 0 P_m0]';
% Cost function parameters
Q=diag([0.5,0.1,0.01]);
R=1;
% Initial conditions
x0=[0 0 0.01]'+x_eq;
% Model-based optimal state-feedback controller
[P_opt,K_opt,~] = icare(A,B,Q,R);
% Initial controller (can be non-stabilizing)
K=zeros(m,n);

%-------------------------------------------------------------------------%
% Reinforcement learning - Value Iteration

simulation_time=15;% simulation time (learning+testing)
T_smpl_sim=0.0001;%simulation sampling time
stopTime=1;%simulation stop time for learning

% Exploratory signal
t_exp=0:T_smpl_sim:stopTime;
exp_signal=zeros(size(t_exp));
for i=1:100
    exp_signal=exp_signal+sin(randi([-100,100],1)*t_exp);
end
exp_signal=0.1*exp_signal./max(exp_signal);
u_exp=timeseries(exp_signal,t_exp);
u_exp.Data=permute(u_exp.Data,[3 2 1]);

mdl='system_mdl';
load_system(mdl);
cs = getActiveConfigSet(mdl);
mdl_cs = cs.copy;
set_param(mdl_cs,'StartTime','0','StopTime',num2str(stopTime),'SolverType','Fixed-Step',...
         'SolverName','FixedStepAuto','FixedStep',num2str(T_smpl_sim));

t=[];
x=[];
u=[];
err_P=[];
err_K=[];

P=0.1*eye(n);
P_0=P;
e_bar=10e-3;

% Simulate the system
out = sim(mdl,mdl_cs);
% Save data
t_sim=out.logsout.getElement('x').Values.Time;
x_sim=out.logsout.getElement('x').Values.Data;
u_sim=out.logsout.getElement('u').Values.Data;
t=t_sim;
x=x_sim;
u=u_sim;
xx=out.logsout.getElement('xx').Values.Data;
xu=out.logsout.getElement('xu').Values.Data;
simSmpl=size(x_sim,1);%number of simulation samples
% Downsampling to T=0.05 sec
x_iter=x_sim(1:500:simSmpl,:);
xx=xx(1:500:simSmpl,:);
xu=xu(1:500:simSmpl,:);

smp_itr=size(x_iter,1);
x0_bar=x_basis(x_iter(1:smp_itr-1,:));
x1_bar=x_basis(x_iter(2:smp_itr,:));
X_diff=x1_bar-x0_bar;
Ixx=xx(2:smp_itr,:)-xx(1:smp_itr-1,:);
Ixu=xu(2:smp_itr,:)-xu(1:smp_itr-1,:);
Theta=[Ixx 2*Ixu*kron(eye(n),R)];
if rank(Theta)<n*(n+1)/2+n*m
    error('Rank condition not satisfied!')
end
pinv_Theta=pinv(Theta);

% Value Iteration Loop
k=0;
q=0;
for i=1:10000
    Y=X_diff*P(:);
    p=pinv_Theta*Y;
    H=P_mat(p(1:n*n));
    K=p(n*n+1:n*n+n*m)';%need to convert from vector to matrix in case of mimo system
    epsilon=1/(1+k);
    P_tilde=P+epsilon*(H+Q-K'*R*K);
    if norm(P_tilde,2)>10*(q+1)
        P=P_0;
        q=q+1;
    elseif norm(P_tilde-P,2)/epsilon<e_bar
        disp('Threshold condition reached!')
        break
    else
        P=P_tilde;
    end
    err_P=[err_P norm(P-P_opt,2)];
    err_K=[err_K norm(K-K_opt,2)];
    k=k+1;
end

%-------------------------------------------------------------------------%
% Testing the learned controller

% Initial conditions for next iteration
x0=x(end,:)'+x_eq;
t0=t(end);
% Set exploratory signal to zero
exp_signal=zeros(size(t_exp));
u_exp=timeseries(exp_signal,t_exp);
u_exp.Data=permute(u_exp.Data,[3 2 1]);
% Simulate the system
set_param(mdl_cs,'StopTime',num2str(simulation_time-t(numel(t))))
out = sim(mdl,mdl_cs);
% Save data
t_sim=out.logsout.getElement('x').Values.Time;
x_sim=out.logsout.getElement('x').Values.Data;
u_sim=out.logsout.getElement('u').Values.Data;

t=[t;t0+t_sim(2:end,:)];
x=[x;x_sim(2:end,:)];
u=[u;u_sim(2:end,:)]; 

%-------------------------------------------------------------------------%
%Plot the results
figure
subplot(2,1,1)
plot(0:numel(err_P)-1,err_P,'LineWidth',1)
legend('$$||P^i-P_{iter}||$$','Interpreter','latex','FontSize',12)
xlabel('Iteration No.')
subplot(2,1,2)
plot(0:numel(err_P)-1,err_P,'LineWidth',1)
legend('$$||K^i-K_{iter}||$$','Interpreter','latex','FontSize',12)
xlabel('Iteration No.')
figure
subplot(2,1,1)
plot(t,x,'LineWidth',1)
legend('$$x_1$$','$$x_2$$','$$x_3$$','Interpreter','latex','FontSize',12)
xlabel('Time (sec)')
subplot(2,1,2)
plot(t,u,'LineWidth',1)
legend('$$u$$','Interpreter','latex','FontSize',12)
xlabel('Time (sec)')

function x_bar=x_basis(x)
%calculate basis functions
% x=[---states-->]
%   [||||||||||||]
%   [||||time||||]  
%   [||||||||||||]
n=size(x,2);%number of states of x
t=size(x,1);%time samples of x
x_bar=zeros(t,n*n);
for i=1:t
    x_bar(i,:)=kron(x(i,:),x(i,:));
end
end

function P=P_mat(p)
%Create matrix P from its vector form
n=sqrt(size(p,1));
P=reshape(p,[n,n]);
P=(P+P')/2;
end