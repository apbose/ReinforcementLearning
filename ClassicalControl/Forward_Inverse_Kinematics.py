import gym
import pybulletgym
import pybulletgym.envs
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import pinv

env = gym.make("ReacherPyBulletEnv-v0")
#env.render()
env.reset()

def getForwardModel (q0, q1, l0 = 0.1, l1 = 0.11):
    """
    Forward kinematics 
    :param q0: Central Joint angle
    :param q1: Shoulder Joint angle
    :param l0: Arm Length l0
    :param l1: Arm Length l1
    :return: x1,x2 position
    """
    H_A_B = np.array([[np.cos(q0), -1 * np.sin(q0), l0 * np.cos(q0)],
                     [ np.sin(q0),  1 * np.cos(q0), l0 * np.sin(q0)],
                     [          0,              0,                 1]]);
    V_B =  np.array([[l1 * np.cos(q1)],
                    [l1 * np.sin(q1)],
                    [1]]);
    
    V_A =  np.matmul(H_A_B, V_B)
    final = V_A[:2, :]/V_A[2,:];
    return final


def getJacobian (q0, q1, l0 = 0.1, l1 = 0.11):
    """
    Jacobian matrix 
    :param q0: Central Joint angle
    :param q1: Shoulder Joint angle
    :param l0: Arm Length l0
    :param l1: Arm Length l1
    :return: Jacobian matrix
    """
    
    #J = np.zeros(shape=(2,2))
    J =  np.array([[-1*l1*np.sin(q1+q0) - l0*np.sin(q0), -1*l1*np.sin(q1+q0)],
                   [   l1*np.cos(q1+q0) + l0*np.cos(q0),    l1*np.cos(q1+q0)]]);
    return J;

def getIK(q0, q1, delta_x, delta_y):
    """
    Inverse kinematics (using Jacobian Inverse)
    :param q0: Current Central Joint angle
    :param q1: Current Shoulder Joint angle
    :param delta_x: Error in position x
    :param delta_y: Error in position y

    :return: Joint angles
    """


    inv_jacobian = np.linalg.pinv(getJacobian(q0, q1))
    del_pos = np.array([[delta_x], [delta_y]])
    del_q = np.matmul(inv_jacobian,del_pos)

    return del_q

def getIK(q0, q1, delta_x, delta_y):
    """
    Inverse kinematics (using Jacobian Inverse)
    :param q0: Current Central Joint angle
    :param q1: Current Shoulder Joint angle
    :param delta_x: Error in position x
    :param delta_y: Error in position y

    :return: Joint angles
    """


    inv_jacobian = np.linalg.pinv(getJacobian(q0, q1))
    del_pos = np.array([[delta_x], [delta_y]])
    del_q = np.matmul(inv_jacobian,del_pos)

    return del_q

def mse (referencex , generatedx , referencey , generatedy):
    """
    MSE error
    :param referencex: xreference trajectory
    :param generatedx : xgenerated trajectory
    :param referencey: yreference trajectory
    :param generatedy : ygenerated trajectory    
    :return: MSE error
    """
    sum = 0
    for i in range(0, len(referencex)):
        sum = sum + (referencex[i] - generatedx[i]) ** 2 + (referencey[i] - generatedy[i]) ** 2
    return np.sqrt(sum/len(referencex))

def problem_1(k_pos, k_vel, steps = 5000, ini_theta = -np.pi):
    
    """
    problem_1 trajectory with input as error in end effector position
    :param k_pos: Position error gain value
    :param k_val: Velocity error gain value
    :steps: No of steps to sample
    :param ini_theta: Initial theta
    """    
    step = float((np.pi*2)/steps);
    x = []
    y = []
    xref = []
    yref = []

    xref0 = x_ref(-1* np.pi);
    yref0 = y_ref(-1* np.pi);


    env.unwrapped.robot.central_joint.reset_position(0,0);
    env.unwrapped.robot.elbow_joint.reset_position(0,0);
    
    for theta in np.arange(0,2*np.pi,step):
    
        k_pos = 25;
        k_vel = 0.9;
        k_pos = np.diag([k_pos, k_pos])
        k_vel = np.diag([k_vel, k_vel])
        q0 , q0dot = env.unwrapped.robot.central_joint.current_position();
        q1 , q1dot = env.unwrapped.robot.elbow_joint.current_position();
        #2*1
        end_effector = getForwardModel(q0,q1);
        x.append(end_effector[0][0]);
        y.append(end_effector[1][0]);
        Jacobian = getJacobian(q0,q1);
        end_effector_vel = np.matmul(Jacobian,np.array([[q0dot],[q1dot]]))
        
        e_pos = np.array([x_ref(theta)-end_effector[0][0], y_ref(theta)-end_effector[1][0]]).reshape(-1, 1)
        e_vel = np.array([-1 * end_effector_vel[0][0], -1 * end_effector_vel[0][0]]).reshape(-1,1)

        
        F = np.matmul(k_pos, e_pos) + np.matmul(k_vel, e_vel)
        xref.append(x_ref(theta))
        yref.append(y_ref(theta))
 
        Torque = np.matmul(Jacobian.T, F);
        action = [Torque[0][0], Torque[1][0]];
        env.step(action)
        
    print("The MSE error is", mse(xref,x, yref, y))
    
    plt.plot(x,y, color='b');
    plt.plot(xref, yref, color='r')        
    plt.show()


problem_1(k_pos = 15, k_vel = 0.95);

def problem_2(k_pos, k_vel, steps = 5000, ini_theta = -np.pi):    
    """
    problem_2 trajectory with input as error in joint angles
    :param k_pos: Angular Position error gain value
    :param k_val: Angular Velocity error gain value
    :steps: No of steps to sample
    :param ini_theta: Initial theta
    """    
    
    step = float((np.pi*2)/steps);
    x = []
    y = []
    xref = []
    yref = [];
    env.unwrapped.robot.central_joint.reset_position(0,0);
    env.unwrapped.robot.elbow_joint.reset_position(0,0);
    

    for theta in np.arange(0,2*np.pi,step):
        k_pos  = 15
        k_vel = 0.95

        k_pos = np.diag([k_pos, k_pos])
        k_vel = np.diag([k_vel, k_vel])
        #this gives the joint angles and the joint angle velocity
        q0 , q0dot = env.unwrapped.robot.central_joint.current_position();
        q1 , q1dot = env.unwrapped.robot.elbow_joint.current_position();
        #2*1
        end_effector = getForwardModel(q0,q1);
        Jacobian = getJacobian(q0,q1);
        x.append(end_effector[0][0]);
        y.append(end_effector[1][0]);
        xref.append(x_ref(theta))
        yref.append(y_ref(theta))
        
        #get the errors
        del_x, del_y = x_ref(theta) - end_effector[0][0], y_ref(theta) - end_effector[1][0]
        angleref_error = getIK(q0 ,q1 , del_x, del_y);
        e_angle = np.array([angleref_error[0][0], angleref_error[1][0] ]).reshape(-1, 1)
        e_vel = np.array([-q0dot, -q1dot]).reshape(-1, 1)
        #PD controller
        Torque = np.matmul(k_pos, e_angle) + np.matmul(k_vel, e_vel)  # PD Controller
        Torque = Torque.reshape(-1)
        env.step(Torque)
    
    print("The MSE error is", mse(xref,x, yref, y))
        
    plt.plot(xref, yref, color='r')
    plt.plot(x,y, color = 'g')

    plt.show()

problem_2(k_pos = 15, k_vel = 0.95)
