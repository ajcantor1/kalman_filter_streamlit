import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

def plot_multivariate_gaussian(ax, mu, sigma):
    x = np.linspace(-1,10,100)
    y = np.linspace(-1,10,100)
    X, Y = np.meshgrid(x,y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y
    rv = multivariate_normal(mean=mu, cov=sigma)
    Z = rv.pdf(pos)
    surface = ax.plot_surface(X, Y, Z, alpha=0.8, cmap=plt.cm.jet, antialiased=True)
    cset = ax.contourf(X, Y, Z, zdir='x', offset=-1, cmap=plt.cm.jet)
    cset = ax.contourf(X, Y, Z, zdir='y', offset=10, cmap=plt.cm.jet)

def predict(A, B, Q, ut, x_hat, sigma_hat):
    x_hat  =  A @ x_hat + B @ ut
    sigma_hat = A @ sigma_hat @ A.T + Q
    return (x_hat, sigma_hat)


def update(H, R, z, x_hat, sigma_hat):
    residual_x_hat = z - H @ x_hat

    residual_sigma_hat = H @ sigma_hat @ H.T + R
    kalman_gain = sigma_hat @ H.T @ np.linalg.inv(residual_sigma_hat)

    x = x_hat + kalman_gain @ residual_x_hat
    sigma =  (np.eye(2) - kalman_gain @ H) @ sigma_hat

    return x, sigma
@st.cache
def paramaters():
    # Model of our system
    A = np.identity(2)

    # System reaction to input
    B = np.identity(2)

    # process process noise covariance
    Q = np.array([[0.05, 0.00001],[0.00001, 0.05]])

    # Model of measurement
    H = np.identity(2)

    # How noise might change after each time step
    R = np.array([[0.05, 0.0001],[0.0001, 0.05]])

    return A, B, Q, H, R

@st.cache
def trajectory(A, B, Q, H, R):
    
    number_steps = 10
    ground_truth_x = np.linspace(0,10,number_steps+1)
    ground_truth_y = ground_truth_x.copy()
    ground_truth = np.stack((ground_truth_x,ground_truth_y),axis=1)

    x0 = 0.1
    y0 = 0.1

    xt = [np.array([x0, y0])] # motion state
    ut = np.array([1, 1]) # initial input
    
    xt_hats = [np.array([x0, y0])]
    sigma_hats = [np.array([[0.3, 0], [0, 0.3]])]

    #measurements
    zt = []

    for i in range(number_steps):
        motion_noise = np.random.multivariate_normal(mean=np.array([0,0]), cov=Q)
        xt_next = A @ xt[-1] + B @ ut + motion_noise
        xt.append(xt_next)

        xt_hat, sigma_hat = predict(A, B, Q, ut, xt_hats[-1], sigma_hats[-1])

        measurement_noise = np.random.multivariate_normal(mean=np.array([0,0]), cov=R)
        zt_next = H @ ground_truth[i+1] + measurement_noise
        zt.append(zt_next)

        xt_hat, sigma_hat = update(H,R, zt_next, xt_hat, sigma_hat)
        xt_hats.append(xt_hat)
        sigma_hats.append(sigma_hat)
        
    return ground_truth, xt, zt, xt_hats, sigma_hats

A, B, Q, H, R = paramaters()
ground_truth, xt, zt, xt_hats, sigma_hats = trajectory(A, B, Q, H, R)


graph_container = st.container()
slider_container = st.container()
steps = slider_container.slider('Steps', 0, 10, 0)

xt = np.array(xt)
zt = np.array(zt)
filtered = np.array(xt_hats)
plt.subplot(121)
fig = plt.figure()
ax = fig.add_subplot(1,2,1)
ax.set_xlim(-1, 10)
ax.set_ylim(-1, 10)
plt.scatter(ground_truth[:steps+1,0], ground_truth[:steps+1,1], alpha=0.5)
plt.scatter(xt[:steps+1,0], xt[:steps+1,1], alpha=0.5)
plt.scatter(zt[:steps+1,0], zt[:steps+1,1], alpha=0.5)
plt.scatter(filtered[1:steps+1,0], filtered[1:steps+1,1], alpha=0.5)
plt.xlabel('x position')
plt.ylabel('y position')
plt.legend(['ground truth', 'motion model', 'measurements', 'KF'])
plt.gca().set_aspect('equal', adjustable='box')

ax = fig.add_subplot(1,2,2, projection='3d')
ax.set_xlim(-1, 10)
ax.set_ylim(-1, 10)
#ax.set_box_aspect((1, 1, 0.25))
plot_multivariate_gaussian(ax, xt_hats[steps], sigma_hats[steps])
graph_container.write(fig)