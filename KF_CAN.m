clc;
clear;

%% Initialization
% Gaussian distribution generator
gaussian_distribution_generator = @(variance) variance * randn();

% State transition matrix
A = [1, 1,  0.05; 
    0, 1,   0.5;
    0,  0,  1];

% Process noise covariance matrix Q
Q = [0.5, 0, 0; 
    0, 0.5, 0;
    0,  0,  0.5];

% Observation noise covariance matrix R
R = [1, 0,  0; 
    0, 1,   0;
    0,  0,  0.5];

% State observation matrix
H = [1, 0,  0; 
    0, 1,   0;
    0,  0,  1];

% Initial position and velocity
X0 = [0; 1; 0];

% Initialize state estimate covariance matrix P
P = [1, 0,  0; 
    0, 1,   0;
    0,  0,  1];

% Initial input matrix
B=[1,0,0;
   0,1,0;
   0,0,1];
u=[0;1;1];

% Initialize true state
X_true = X0; 
X_posterior = X0;
P_posterior = P;

speed_true = [];
position_true = [];
angle_true = [];

speed_measure = [];
position_measure = [];
angle_measure = [];

speed_prior_est = [];
position_prior_est = [];
angle_prior_est = [];

speed_posterior_est = [];
position_posterior_est = [];
angle_posterior_est = [];

%% Kalman Filter
for i = 1:30
    % -----------------------Generate true value----------------------
    % Generate process noise
    w = [gaussian_distribution_generator(sqrt(Q(1, 1))); gaussian_distribution_generator(sqrt(Q(2, 2))); gaussian_distribution_generator(sqrt(Q(3, 3)))];
    % Get current state
    X_true = A * X_true +B*u + w;  
    speed_true = [speed_true; X_true(2)];
    position_true = [position_true; X_true(1)];
    angle_true = [angle_true; X_true(3)];

    % -----------------------Generate observation----------------------
    % Generate observation noise
    v = [gaussian_distribution_generator(sqrt(R(1, 1))); gaussian_distribution_generator(sqrt(R(2, 2))); gaussian_distribution_generator(sqrt(R(3, 3)))];
    % Generate observation
    Z_measure = H * X_true + v;  
    position_measure = [position_measure; Z_measure(1)];
    speed_measure = [speed_measure; Z_measure(2)];
    angle_measure = [angle_measure; Z_measure(3)];

    % ----------------------Prior estimation---------------------
    X_prior = A * X_posterior;
    position_prior_est = [position_prior_est; X_prior(1)];
    speed_prior_est = [speed_prior_est; X_prior(2)];
    angle_prior_est = [angle_prior_est; X_prior(3)];

    % Compute state estimate covariance matrix P
    P_prior = A * P_posterior * A' + Q;

    % ----------------------Calculate Kalman gain--------------------
    K = P_prior * H' / (H * P_prior * H' + R);

    % ---------------------Posterior estimation--------------------------
    X_posterior = X_prior + K * (Z_measure - H * X_prior);
    position_posterior_est = [position_posterior_est; X_posterior(1)];
    speed_posterior_est = [speed_posterior_est; X_posterior(2)];
    angle_posterior_est = [angle_posterior_est; X_posterior(3)];
    % Update state estimate covariance matrix P
    P_posterior = (eye(3) - K * H) * P_prior;
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