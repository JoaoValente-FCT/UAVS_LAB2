clear all;
x0=zeros(6,1);
D=0.1075;
m=0.029;
g=9.81;



A1=zeros(3);
A2=eye(3);
A3=eye(3)*-D/m;

A=[A1,A2;A1,A3];


B=[zeros(3);eye(3)];

Q=eye(6)*4;

R=eye(3)*0.25;

[K,s,P]=lqr(A,B,Q,R);









%step(sys2);

% simulate nonlinear system
Dt = 0.01;
t = 0:Dt:20; 
N = length(t);
x = zeros(6,N+1); % zero initial state
u_a = zeros(3,N); 

pz_ref = zeros(1,N);
px_ref = zeros(1,N);
py_ref = zeros(1,N);
p_ref=zeros(3,N);


ue = 0; % u = T - mg => ue = 0
ve=0;



% simulation "time" loop
for k = 1:N
    % define reference and errors
    px_ref(k) = 10*(k*Dt>=2);
    py_ref(k) = 20*(k*Dt>=3);
    pz_ref(k) = 15*(k*Dt>=5);% step reference
    p_ref(:,k) = [px_ref(k);py_ref(k);pz_ref(k)];
    e_x = x(:,k)-[p_ref(:,k);ve;ve;ve]; % state error
    v_d = zeros(3,N);
    dot_v_d = zeros(3,N);

    v_d(:,k) = [0; 0; 0]; 
    dot_v_d(:,k) = [0; 0; 0];



    % define UAV dynamics
theta=0.1;
phi=0.1;
T_total=0.4;
    fp = T_total*[sin(theta)*cos(phi);
        -sin(phi);
        cos(theta)*cos(phi)];

    D=[0.1075,0,0;
        0,0.1075,0;
        0,0,0.1075];

    fa=[-D(1,1)*x(3,k);-D(2,2)*x(4,k);-D(3,3)*x(5,k)];

 

ep=x(1:3,k)-p_ref(:,k);

ev=x(4:6,k)-v_d(:,k);

    dot_error = [ep;ev];

    bar_ua=(D/m) * v_d(:,k) + dot_v_d(:,k);


kp=1;
kv=1;




    u_a(:,k)=bar_ua+D/m*v_d(:,k)+dot_v_d(:,k)-kp*ep-kv*ev-ep;

   P_dot=[x(4,k);x(5,k);x(6,k)];
    V_dot=(u_a(:,k) + fa - [0; 0; m*g]) / m;


dot_x = [P_dot; V_dot];




    % integrate dynamics
    x(:,k+1) = x(:,k) + Dt*dot_x;
end
x(:,k+1) = [];
p_ref(:,N) = [px_ref(N); py_ref(N); pz_ref(N)];



figure(341);
subplot(2,1,1);
plot(t, p_ref(1,:), 'r--', t, x(1,:), 'r', ...
    t, p_ref(2,:), 'g--', t, x(2,:), 'g', ...
    t, p_ref(3,:), 'b--', t, x(3,:), 'b', 'LineWidth', 1.5);
grid on;
ylabel('Posição [m]');
legend('x_{ref}', 'x', 'y_{ref}', 'y', 'z_{ref}', 'z', 'Location', 'best');
title('Resposta ao Degrau - Posições');


subplot(2,1,2);
plot(t, x(4,:), 'r', t, x(5,:), 'g', t, x(6,:), 'b', 'LineWidth', 1.5);
grid on;
xlabel('Tempo [s]');
ylabel('Velocidade [m/s]');
legend('v_x', 'v_y', 'v_z', 'Location', 'best');

figure(342);
plot(t, u_a(1,:), 'r', t, u_a(2,:), 'g', t, u_a(3,:), 'b', 'LineWidth', 1.5);
grid on;
xlabel('Tempo [s]');
ylabel('Força de Actuação [N]');
title('Sinais de Controlo (UAV Actuation)');
legend('U_x', 'U_y', 'U_z');