clc;
clear;

%% Initialization
% Gaussian distribution generator
gaussian_distribution_generator = @(variance) variance * randn();

% Parameters for the model
Vc = 2000;
Vp = 600;
p = 2.7;
b = 0.5;
l = 0.3;
I = 0.01;
g = 9.8;
M = p*Vc; % mass of car
m = p*Vp; % mass of pendulum

% Matrices for the state space model
p = I*(M+m)+M*m*l^2; 

% Parameters for the state space
A = [0      1              0           0;
     0 -(I+m*l^2)*b/p  (m^2*g*l^2)/p   0;
     0      0              0           1;
     0 -(m*l*b)/p       m*g*l*(M+m)/p  0];
B = [     0;
     (I+m*l^2)/p;
          0;
        m*l/p];
C = [1 0 0 0;
     0 0 1 0];
D = [0;
     0];

% Process noise covariance matrix Q
Q = [0.5, 0, 0, 0; 
    0, 0.5, 0,  0;
    0,  0,  0.5, 0
    0,  0,  0,  0.5];

% Observation noise covariance matrix R
R = [1, 0, 0, 0; 
    0, 1, 0,  0;
    0,  0,  1, 0
    0,  0,  0,  1];

% State observation matrix
H = [1, 0, 0, 0; 
    0, 1, 0,  0;
    0,  0,  1, 0
    0,  0,  0,  1];

% Initial position and velocity
X0 = [20;20;20;20];

% Initialize state estimate covariance matrix P
P = [1, 0, 0, 0; 
    0, 1, 0,  0;
    0,  0,  1, 0
    0,  0,  0,  1];

% Initialize true state
X_true = X0; 
X_posterior = X0;
P_posterior = P;

error_speed = [];
error_position = [];
error_angle = [];
error_angle_speed = [];

speed_true = [];
position_true = [];
angle_true = [];
angle_speed_true = [];

speed_measure = [];
position_measure = [];
angle_measure = [];
angle_speed_measure = [];

speed_prior_est = [];
position_prior_est = [];
angle_prior_est = [];
angle_speed_prior_est = [];

speed_posterior_est = [];
position_posterior_est = [];
angle_posterior_est = [];
angle_speed_posterior_est = [];

% design state feedback control law
% new poles
p1 = -4;
p2 = -3;
p3 = -5;
p4 = -7;
P = [p1 p2 p3 p4];

% define the state space model
sys_ss = ss(A,B,C,D);

% use pole placement to get K
K_u = place(A,B,P);

% closed-loop system
sys_cl=ss(A-B*K_u,B,C,D);

% simulation
t=0:0.05:5;
u=0*ones(size(t)); % zero input

% compute the state and output trajectories of the closed-loop system
[Y,T,X]=lsim(sys_cl,u,t,X0);
u=0;

%% Kalman Filter
for i=1:100
    % -----------------------Generate true value----------------------
    % Generate process noise
    w = [gaussian_distribution_generator(sqrt(Q(1, 1))); gaussian_distribution_generator(sqrt(Q(2, 2))); gaussian_distribution_generator(sqrt(Q(3, 3))); gaussian_distribution_generator(sqrt(Q(4, 4)))];
    % Get current state
    X_true = A * X(i,:)' +B*u + w;  


    position_true = [position_true; X_true(1)];
    speed_true = [speed_true; X_true(2)];
    angle_true = [angle_true; X_true(3)];
    angle_speed_true = [angle_speed_true; X_true(4)];
    
    % -----------------------Generate observation----------------------
    % Generate observation noise
    v = [gaussian_distribution_generator(sqrt(R(1, 1))); gaussian_distribution_generator(sqrt(R(2, 2))); gaussian_distribution_generator(sqrt(R(3, 3))); gaussian_distribution_generator(sqrt(R(4, 4)))];
    % Generate observation
    Z_measure = H * X_true + v;  

    position_measure = [position_measure; Z_measure(1)];
    speed_measure = [speed_measure; Z_measure(2)];
    angle_measure = [angle_measure; Z_measure(3)];
    angle_speed_measure = [angle_speed_measure; Z_measure(4)];

    % ----------------------Prior estimation---------------------
    X_prior = A * X_posterior;

    position_prior_est = [position_prior_est; X_prior(1)];
    speed_prior_est = [speed_prior_est; X_prior(2)];
    angle_prior_est = [angle_prior_est; X_prior(3)];
    angle_speed_prior_est = [angle_speed_prior_est; X_prior(4)];

    % Compute state estimate covariance matrix P
    P_prior = A * P_posterior * A' + Q;

    % ----------------------Calculate Kalman gain--------------------
    K = P_prior * H' / (H * P_prior * H' + R);

    % ---------------------Posterior estimation--------------------------
    X_posterior = X_prior + K * (Z_measure - H * X_prior);

    position_posterior_est = [position_posterior_est; X_posterior(1)];
    speed_posterior_est = [speed_posterior_est; X_posterior(2)];
    angle_posterior_est = [angle_posterior_est; X_posterior(3)];
    angle_speed_posterior_est = [angle_speed_posterior_est; X_posterior(4)];

    % Update state estimate covariance matrix P
    P_posterior = (eye(4) - K * H) * P_prior;

    % Update the error
    e_speed = speed_true(i,:) - speed_posterior_est(i,:);
    e_positon = position_true(i,:) - position_posterior_est(i,:);
    e_angle = angle_true(i,:) - angle_posterior_est(i,:);
    e_angle_speed = angle_speed_true(i,:) - angle_speed_posterior_est(i,:);
    error_speed = [error_speed e_speed];
    error_position = [error_position e_positon];
    error_angle = [error_angle e_angle];
    error_angle_speed = [error_angle_speed e_angle_speed];

end


%% Visualization
figure;
hold on;
plot(speed_true, '-', 'LineWidth', 1);
plot(speed_measure, '--', 'LineWidth', 1);
plot(speed_prior_est, ':', 'LineWidth', 1);
plot(speed_posterior_est, '-.', 'LineWidth', 1);
legend('True', 'Measured', 'Prior Estimate', 'Posterior Estimate');
xlabel('Time Step');
ylabel('Speed');
title('Velocity');

figure;
hold on;
plot(position_true, '-', 'LineWidth', 1);
plot(position_measure, '--', 'LineWidth', 1);
plot(position_prior_est, ':', 'LineWidth', 1);
plot(position_posterior_est, '-.', 'LineWidth', 1);
legend('True', 'Measured', 'Prior Estimate', 'Posterior Estimate');
xlabel('Time Step');
ylabel('Position');
title('Position');

figure;
hold on;
plot(angle_true, '-', 'LineWidth', 1);
plot(angle_measure, '--', 'LineWidth', 1);
plot(angle_prior_est, ':', 'LineWidth', 1);
plot(angle_posterior_est, '-.', 'LineWidth', 1);
legend('True', 'Measured', 'Prior Estimate', 'Posterior Estimate');
xlabel('Time Step');
ylabel('Angle');
title('Angle');

figure;
hold on;
plot(angle_speed_true, '-', 'LineWidth', 1);
plot(angle_speed_measure, '--', 'LineWidth', 1);
plot(angle_speed_prior_est, ':', 'LineWidth', 1);
plot(angle_speed_posterior_est, '-.', 'LineWidth', 1);
legend('True', 'Measured', 'Prior Estimate', 'Posterior Estimate');
xlabel('Time Step');
ylabel('Angle Speed');
title('Angle Speed');

figure;
hold on;
subplot(2,1,1);
plot(error_speed)
xlabel('Time')
ylabel('Error')
title('Speed')


hold on;
subplot(2,1,2);
plot(error_position)
xlabel('Time')
ylabel('Error')
title('Position')

figure;
hold on;
subplot(2,1,1);
plot(error_angle)
xlabel('Time')
ylabel('Error')
title('Angle')


hold on;
subplot(2,1,2);
plot(error_angle_speed)
xlabel('Time')
ylabel('Error')
title('Angle Speed')