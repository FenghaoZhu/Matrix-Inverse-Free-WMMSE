"""
Tool functions for the project.
Author: zfh
Date: 2024.1.4
"""

# <editor-fold desc="Import Packages">
import numpy as np
import matplotlib.pyplot as plt
import copy
from copy import deepcopy
import time
import torch
import scipy.io as sio

# </editor-fold>

# <editor-fold desc="Set variables">
nr_of_users = 4
nr_of_BS_antennas = 8
nr_of_UE_antennas = 2
nr_of_data_streams = 2
Total_Power = 10  # dBm, power constraint in the weighted sum rate maximization problem
Noise_Power = 1  # dBm
nr_of_iterations = 4
scale_V_every_iteration = True  # used to normalize V at every iteration such that the power constraint is met with equality
L = nr_of_iterations  # number of layers in the deep unfolded WMMSE
J_u = 2  # number of inner iterations for the receiver precoder
J_w = 2  # number of inner iterations for the mse weighs
J_v = 4  # number of inner iterations for the transmitter precoder

# For the WMMSE
epsilon = 0.0001  # used to end the iterations of the WMMSE algorithm in Shi et al. when the number of iterations is not fixed (note that the stopping criterion has precendence over the fixed number of iterations)
power_tolerance = 0.0001  # used to end the bisection search in the WMMSE algorithm in Shi et al.
nr_of_iterations_WMMSE = nr_of_iterations  # for WMMSE algorithm in Shi et al.
learning_rate = 0.001

# For the matrix-inverse-free WMMSE
nr_of_batches_training = 10000  # used for training
nr_of_batches_test = 1000  # used for testing
nr_of_samples_per_batch = 100
batch_size = nr_of_samples_per_batch

nr_of_iterations_nn = nr_of_iterations  # for the deep unfolded WMMSE in our paper
user_weights = torch.ones((batch_size, nr_of_users))  # user weight for torch
user_weights_WMMSE = np.ones(nr_of_users, )  # user weights for numpy


# </editor-fold>


# <editor-fold desc="Define the util function">
def compute_sinr_MIMO(channel, precoder, noise_power, user_id):
    """
        Compute the SINR of the user with id user_id
        This version of SINR computation deals with numpy format, channel and precoder are a complex matrices
        :param channel: channel matrix
        :param precoder: precoder matrix
        :param noise_power: noise power
        :param user_id: user id
        :return: SINR of the user with id user_id
    """
    result = 0
    nr_of_users = np.size(channel, 0)
    nr_of_UE_antennas = np.size(channel, 1)
    inter_user_interference = np.zeros((nr_of_UE_antennas, nr_of_UE_antennas)) + 1j * np.zeros(
        (nr_of_UE_antennas, nr_of_UE_antennas))

    numerator = np.matmul(np.matmul(np.matmul(channel[user_id, :, :], precoder[user_id, :, :]),
                                    np.transpose(np.conj(precoder[user_id, :, :]))),
                          np.transpose(np.conj(channel[user_id, :, :])))

    for user_index in range(nr_of_users):
        if user_index != user_id:
            inter_user_interference = inter_user_interference + np.matmul(
                np.matmul(np.matmul(channel[user_id, :, :], precoder[user_index, :, :]),
                          np.transpose(np.conj(precoder[user_index, :, :]))),
                np.transpose(np.conj(channel[user_id, :, :])))

    denominator = noise_power * np.eye(nr_of_UE_antennas, nr_of_UE_antennas) + inter_user_interference

    result = np.matmul(numerator, np.linalg.inv(denominator))

    return result


def compute_weighted_sum_rate_MIMO(user_weights, channel, precoder, noise_power):
    """
    Compute the weighted sum rate of the system
    :param user_weights: weights of the users
    :param channel: channel matrix
    :param precoder: precoder matrix
    :param noise_power: noise power
    :return: weighted sum rate of the system
    """
    result = 0
    nr_of_users = np.size(channel, 0)
    nr_of_UE_antennas = np.size(channel, 1)

    for user_index in range(nr_of_users):
        user_sinr = compute_sinr_MIMO(channel, precoder, noise_power, user_index)
        # Attention! log2 must be used instead of log!!!
        result = result + user_weights[user_index] * np.log2(
            np.linalg.det(np.eye(nr_of_UE_antennas, nr_of_UE_antennas) + user_sinr))

    # print("result:", result)
    result = np.real(result)

    return result


def run_WMMSE_MIMO_more_streams(epsilon, channel, initial_transmitter_precoder_WMMSE, total_power, noise_power,
                                user_weights, max_nr_of_iterations, log=False):
    """
     Run the WMMSE algorithm in Shi et al. for the weighted sum rate maximization problem
    :param epsilon: criterion to stop the algorithm
    :param channel: channel matrix
    :param initial_transmitter_precoder_WMMSE: initial value of the transmitter precoder
    :param total_power: total power constraint
    :param noise_power: noise power
    :param user_weights: weights of the users
    :param max_nr_of_iterations: maximum number of iterations
    :param log: if True, then the algorithm prints the WSR
    :return: transmitter precoder, receiver precoder, MSE weights, the last WSR
    """
    # initialization
    mse_weights = np.zeros((nr_of_users, nr_of_data_streams, nr_of_data_streams)) + 1j * np.zeros(
        (nr_of_users, nr_of_data_streams, nr_of_data_streams))
    receiver_precoder = np.zeros((nr_of_users, nr_of_UE_antennas, nr_of_data_streams)) + 1j * np.zeros(
        (nr_of_users, nr_of_UE_antennas, nr_of_data_streams))
    transmitter_precoder = np.zeros((nr_of_users, nr_of_BS_antennas, nr_of_data_streams)) + 1j * np.zeros(
        (nr_of_users, nr_of_BS_antennas, nr_of_data_streams))
    new_transmitter_precoder = np.zeros((nr_of_users, nr_of_BS_antennas, nr_of_data_streams)) + 1j * np.zeros(
        (nr_of_users, nr_of_BS_antennas, nr_of_data_streams))
    new_receiver_precoder = np.zeros((nr_of_users, nr_of_UE_antennas, nr_of_data_streams)) + 1j * np.zeros(
        (nr_of_users, nr_of_UE_antennas, nr_of_data_streams))
    WSR_E = []
    WSR = []
    WSR_E.append([0])
    WSR.append(compute_weighted_sum_rate_MIMO(user_weights, channel, transmitter_precoder, noise_power))

    for user_index in range(nr_of_users):
        receiver_precoder[user_index, :, :] = np.zeros((nr_of_UE_antennas, nr_of_data_streams))
        transmitter_precoder[user_index, :, :] = np.random.normal(size=(nr_of_BS_antennas, nr_of_data_streams))

    power = (np.linalg.norm(transmitter_precoder)) ** 2
    transmitter_precoder = initial_transmitter_precoder_WMMSE
    mse_weigths_old = mse_weights

    nr_of_iteration_counter = 1  # to keep track of the number of iteration of the WMMSE
    break_condition = 2 * epsilon

    while break_condition >= epsilon and nr_of_iteration_counter <= max_nr_of_iterations:

        nr_of_iteration_counter = nr_of_iteration_counter + 1

        ###################################
        # optimize receiver precoder
        for user_index in range(nr_of_users):
            user_interference = np.zeros((nr_of_UE_antennas, nr_of_UE_antennas)) + 1j * np.zeros(
                (nr_of_UE_antennas, nr_of_UE_antennas))
            for user_index2 in range(nr_of_users):
                user_interference = user_interference + np.matmul(channel[user_index, :, :],
                                                                  np.matmul(transmitter_precoder[user_index2, :, :],
                                                                            np.matmul(np.transpose(np.conj(
                                                                                transmitter_precoder[user_index2, :,
                                                                                :])), np.transpose(
                                                                                np.conj(channel[user_index, :, :])))))

            new_receiver_precoder[user_index, :, :] = np.matmul(
                np.linalg.inv(np.eye(nr_of_UE_antennas, nr_of_UE_antennas) * noise_power + user_interference),
                np.matmul(channel[user_index, :, :], transmitter_precoder[user_index, :, :]))

        ####################################
        # optimize mse weights
        for user_index in range(nr_of_users):
            mse_weights[user_index, :, :] = np.linalg.inv(np.eye(nr_of_data_streams, nr_of_data_streams) - np.matmul(
                np.matmul(np.transpose(np.conj(new_receiver_precoder[user_index, :, :])), channel[user_index, :, :]),
                transmitter_precoder[user_index, :, :]))

        ####################################
        # optimize transmitter precoder
        A = np.zeros((nr_of_BS_antennas, nr_of_BS_antennas)) + 1j * np.zeros((nr_of_BS_antennas, nr_of_BS_antennas))

        for user_index in range(nr_of_users):
            A = A + user_weights[user_index] * np.matmul(np.matmul(np.matmul(
                np.matmul(np.transpose(np.conj(channel[user_index, :, :])), new_receiver_precoder[user_index, :, :]), \
                mse_weights[user_index, :, :]), np.transpose(np.conj(new_receiver_precoder[user_index, :, :]))),
                channel[user_index, :, :])

        Sigma_diag_elements_true, U = np.linalg.eigh(A)
        Sigma_diag_elements = copy.deepcopy(np.real(Sigma_diag_elements_true))
        Lambda = np.zeros((nr_of_BS_antennas, nr_of_BS_antennas)) + 1j * np.zeros(
            (nr_of_BS_antennas, nr_of_BS_antennas))

        for user_index in range(nr_of_users):
            Lambda = Lambda + ((user_weights[user_index]) ** 2) * np.matmul(np.matmul(np.matmul(np.matmul(np.matmul( \
                np.transpose(np.conj(channel[user_index, :, :])), \
                new_receiver_precoder[user_index, :, :]), mse_weights[user_index, :, :]),
                np.transpose(np.conj(mse_weights[user_index, :, :]))), \
                np.transpose(np.conj(new_receiver_precoder[user_index, :, :]))), channel[user_index, :, :])

        Phi = np.matmul(np.matmul(np.conj(np.transpose(U)), Lambda), U)
        Phi_diag_elements_true = np.diag(Phi)
        Phi_diag_elements = copy.deepcopy(Phi_diag_elements_true)
        Phi_diag_elements = np.real(Phi_diag_elements)

        for i in range(len(Phi_diag_elements)):
            if Phi_diag_elements[i] < np.finfo(float).eps:
                Phi_diag_elements[i] = np.finfo(float).eps
            if (Sigma_diag_elements[i]) < np.finfo(float).eps:
                Sigma_diag_elements[i] = 0

        # Check if mu = 0 is a solution (eq.s (15) and (16) of in the paper of Shi et al.)
        power = 0  # the power of transmitter precoder (i.e. sum of the squared norm)
        for user_index in range(nr_of_users):
            if np.linalg.det(A) != 0:
                temp = np.matmul(np.linalg.inv(A), np.matmul(np.matmul(np.transpose(np.conj(channel[user_index, :, :])),
                                                                       new_receiver_precoder[user_index, :, :]),
                                                             mse_weights[user_index, :, :]))
                power = power + (np.linalg.norm(temp)) ** 2

        # If mu = 0 is a solution, then mu_star = 0
        if np.linalg.det(A) != 0 and power <= total_power:
            mu_star = 0
        # If mu = 0 is not a solution then we search for the "optimal" mu by bisection
        else:
            power_distance = []  # list to store the distance from total_power in the bisection algorithm
            mu_low = np.sqrt(1 / total_power * np.sum(Phi_diag_elements))
            mu_high = 0
            low_point = compute_P(Phi_diag_elements, Sigma_diag_elements, mu_low)
            high_point = compute_P(Phi_diag_elements, Sigma_diag_elements, mu_high)

            obtained_power = total_power + 2 * power_tolerance  # initialization of the obtained power such that we enter the while

            # Bisection search
            while np.absolute(total_power - obtained_power) > power_tolerance:
                mu_new = (mu_high + mu_low) / 2
                obtained_power = compute_P(Phi_diag_elements, Sigma_diag_elements,
                                           mu_new)  # eq. (18) in the paper of Shi et al.
                power_distance.append(np.absolute(total_power - obtained_power))
                if obtained_power > total_power:
                    mu_high = mu_new
                if obtained_power < total_power:
                    mu_low = mu_new
            mu_star = mu_new

            if log == True:
                print("first value:", power_distance[0])
                plt.title("Distance from the target value in bisection (it should decrease)")
                plt.plot(power_distance)
                plt.show()

        for user_index in range(nr_of_users):
            new_transmitter_precoder[user_index, :, :] = user_weights[user_index] * np.matmul(
                np.matmul(np.matmul(np.linalg.inv(A + mu_star * np.eye(nr_of_BS_antennas, nr_of_BS_antennas)), \
                                    np.transpose(np.conj(channel[user_index, :, :]))),
                          new_receiver_precoder[user_index, :, :]), mse_weights[user_index, :, :])

        transmitter_precoder = deepcopy(new_transmitter_precoder)
        receiver_precoder = deepcopy(new_receiver_precoder)

        WSR_E.append(
            np.real(np.sum(np.multiply(np.log2(np.squeeze(np.linalg.det(np.real(mse_weights)))), user_weights))))
        mse_weights_old = mse_weights
        WSR.append(compute_weighted_sum_rate_MIMO(user_weights, channel, transmitter_precoder, noise_power))
        break_condition = np.absolute(WSR_E[-1] - WSR_E[-2])

    if log == True:
        plt.title("Change of the WSR at each iteration of the WMMSE (it should increase)")
        plt.plot(WSR, 'bo')
        plt.show()

    return transmitter_precoder, receiver_precoder, mse_weights, WSR[-1]


# def run_WMMSE_MIMO_more_streams_vectorize(epsilon, channel, initial_transmitter_precoder_WMMSE, total_power,
#                                           noise_power, user_weights, max_nr_of_iterations):
#     WMMSE_vectorize = np.vectorize(run_WMMSE_MIMO_more_streams, excluded=[0,1, 3, 4, 5, 6])
#
#     transmitter_precoder_batch, receiver_precoder_batch, mse_weights_batch, WSR_batch = WMMSE_vectorize(epsilon,
#                                                                                                         channel,
#                                                                                                         initial_transmitter_precoder_WMMSE,
#                                                                                                         total_power,
#                                                                                                         noise_power,
#                                                                                                         user_weights,
#                                                                                                         max_nr_of_iterations)
#
#     return transmitter_precoder_batch, receiver_precoder_batch, mse_weights_batch, WSR_batch


def compute_P(Phi_diag_elements, Sigma_diag_elements, mu):
    """
     Compute the power for bisection search in the optimization of the transmitter precoder
     - eq. (18) in the paper by Shi et al.
    :param Phi_diag_elements:
    :param Sigma_diag_elements:
    :param mu: the value related to power control
    :return: the sum power
    """
    nr_of_BS_antennas = Phi_diag_elements.size
    mu_array = mu * np.ones(Phi_diag_elements.size)
    result = np.divide(Phi_diag_elements, (Sigma_diag_elements + mu_array) ** 2)
    result = np.sum(result)
    return result


# Computes a channel realization and returns it in two formats, one for the WMMSE and one for the unfolded
# matrix-inverse-free WMMSE. It also returns the initialization value of the transmitter precoder and the receiver
# precoder, which are used as input in the computation graph of the unfolded matrix-inverse-free WMMSE.
def compute_channel(nr_of_BS_antennas, nr_of_users, total_power=Total_Power, noise_power=Noise_Power):
    """
      Compute a channel realization and returns it in two formats, one for the WMMSE and one for the unfolded
    :param noise_power: power of noise
    :param nr_of_BS_antennas: number of BS antennas
    :param nr_of_users: number of users
    :param total_power: total power
    :return: channel_nn, initial_transmitter_precoder, initial_receiver_precoder, channel_WMMSE, initial_transmitter_precoder_WMMSE
    """
    channel_nn = []
    initial_transmitter_precoder = []
    initial_transmitter_precoder_WMMSE = []
    initial_receiver_precoder = []
    channel_WMMSE = np.zeros((nr_of_users, nr_of_UE_antennas, nr_of_BS_antennas)) + 1j * np.zeros(
        (nr_of_users, nr_of_UE_antennas, nr_of_BS_antennas))

    transmitter_precoder_power = 0

    VVh = 0
    nr_of_Schulz_iterations = 1

    # If number of data streams is equal to one
    if nr_of_data_streams == 1:

        for i in range(nr_of_users):
            result_real = np.sqrt(0.5) * np.random.normal(size=(nr_of_UE_antennas, nr_of_BS_antennas))
            result_imag = np.sqrt(0.5) * np.random.normal(size=(nr_of_UE_antennas, nr_of_BS_antennas))

            channel_WMMSE[i, :, :] = np.reshape(result_real, (nr_of_UE_antennas, nr_of_BS_antennas)) + 1j * np.reshape(
                result_imag, (nr_of_UE_antennas, nr_of_BS_antennas))

            # print(result_real.shape)

            result_col_1 = np.vstack((result_real, result_imag))
            result_col_2 = np.vstack((-result_imag, result_real))
            result = np.hstack((result_col_1, result_col_2))

            # print(result_col_1.shape, result_col_2.shape, result.shape)
            channel_nn.append(result)

            ## transmitter precoder
            tp = np.reshape(np.sum(result_real, axis=0), (nr_of_BS_antennas, 1)) + 1j * np.reshape(
                -1 * np.sum(result_imag, axis=0), (nr_of_BS_antennas, 1))
            initial_transmitter_precoder_WMMSE.append(tp)

            real_tp = np.real(tp)
            imag_tp = np.imag(tp)

            first_row_tp = np.concatenate((real_tp, -1 * imag_tp), axis=1)
            second_row_tp = np.concatenate((imag_tp, real_tp), axis=1)

            initial_transmitter_precoder.append(np.concatenate((first_row_tp, second_row_tp), axis=0))
            transmitter_precoder_power = transmitter_precoder_power + np.linalg.norm(tp) ** 2

            ## receiver precoder
            rp = np.reshape(np.sum(result_real, axis=1), (nr_of_UE_antennas, 1)) + 1j * np.reshape(
                np.sum(result_imag, axis=1), (nr_of_UE_antennas, 1))
            real_rp = np.real(rp)
            imag_rp = np.imag(rp)

            first_row_rp = np.concatenate((real_rp, -1 * imag_rp), axis=1)
            second_row_rp = np.concatenate((imag_rp, real_rp), axis=1)

            initial_receiver_precoder.append(np.concatenate((first_row_rp, second_row_rp), axis=0))

        initial_transmitter_precoder = np.array(initial_transmitter_precoder)
        initial_transmitter_precoder_WMMSE = np.array(initial_transmitter_precoder_WMMSE)

        initial_transmitter_precoder = np.sqrt(total_power) * initial_transmitter_precoder / np.sqrt(
            transmitter_precoder_power)
        initial_transmitter_precoder_WMMSE = np.sqrt(total_power) * initial_transmitter_precoder_WMMSE / np.sqrt(
            transmitter_precoder_power)

    # If number of data streams is more than one
    else:
        ## transmitter precoder
        for i in range(nr_of_users):
            result_real = np.sqrt(0.5) * np.random.normal(size=(nr_of_UE_antennas, nr_of_BS_antennas))
            result_imag = np.sqrt(0.5) * np.random.normal(size=(nr_of_UE_antennas, nr_of_BS_antennas))

            temp = np.reshape(result_real, (nr_of_UE_antennas, nr_of_BS_antennas)) + 1j * np.reshape(result_imag, (
                nr_of_UE_antennas, nr_of_BS_antennas))

            channel_WMMSE[i, :, :] = temp

            result_col_1 = np.vstack((result_real, result_imag))
            result_col_2 = np.vstack((-result_imag, result_real))
            result = np.hstack((result_col_1, result_col_2))
            channel_nn.append(result)
            channel_norm_by_row = np.linalg.norm(temp, axis=1)
            channel_row_index = (np.argsort(channel_norm_by_row))[-nr_of_data_streams:][::-1]

            tp = np.transpose(np.conj(temp[channel_row_index,]))
            transmitter_precoder_power = transmitter_precoder_power + np.linalg.norm(tp) ** 2
            initial_transmitter_precoder_WMMSE.append(tp)

            real_tp = np.real(tp)
            imag_tp = np.imag(tp)

            first_row_tp = np.concatenate((real_tp, -1 * imag_tp), axis=1)
            second_row_tp = np.concatenate((imag_tp, real_tp), axis=1)
            initial_transmitter_precoder.append(np.concatenate((first_row_tp, second_row_tp), axis=0))

        initial_transmitter_precoder = np.array(initial_transmitter_precoder)
        initial_transmitter_precoder_WMMSE = np.array(initial_transmitter_precoder_WMMSE)

        initial_transmitter_precoder = np.sqrt(total_power) * initial_transmitter_precoder / np.sqrt(
            transmitter_precoder_power)
        initial_transmitter_precoder_WMMSE = np.sqrt(total_power) * initial_transmitter_precoder_WMMSE / np.sqrt(
            transmitter_precoder_power)

        I = np.eye(nr_of_UE_antennas)

        for i in range(nr_of_users):
            VVh = VVh + np.matmul(initial_transmitter_precoder_WMMSE[i, :, :],
                                  np.transpose(np.conj(initial_transmitter_precoder_WMMSE[i, :, :])))

        ## receiver precoder
        if nr_of_data_streams != nr_of_UE_antennas:
            # Initialize U as matched filtering
            for i in range(nr_of_users):
                rp = np.matmul(channel_WMMSE[i, :, :], initial_transmitter_precoder_WMMSE[i, :, :])
                real_rp = np.real(rp)
                imag_rp = np.imag(rp)

                first_row_rp = np.concatenate((real_rp, -1 * imag_rp), axis=1)
                second_row_rp = np.concatenate((imag_rp, real_rp), axis=1)

                initial_receiver_precoder.append(np.concatenate((first_row_rp, second_row_rp), axis=0))

        if nr_of_data_streams == nr_of_UE_antennas:
            # Initialize U as scaled identity matrix

            for i in range(nr_of_users):
                scaling = (np.trace(np.matmul(channel_WMMSE[i, :, :], initial_transmitter_precoder_WMMSE[i, :, :]))) / (
                        noise_power * nr_of_UE_antennas + np.trace(np.matmul(np.matmul(channel_WMMSE[i, :, :], VVh),
                                                                             np.transpose(
                                                                                 np.conj(channel_WMMSE[i, :, :])))))
                rp = np.eye(nr_of_UE_antennas) * np.real(scaling)
                real_rp = np.real(rp)
                imag_rp = np.imag(rp)
                first_row_rp = np.concatenate((real_rp, -1 * imag_rp), axis=1)
                second_row_rp = np.concatenate((imag_rp, real_rp), axis=1)

                initial_receiver_precoder.append(np.concatenate((first_row_rp, second_row_rp), axis=0))

    return channel_nn, initial_transmitter_precoder, initial_receiver_precoder, channel_WMMSE, initial_transmitter_precoder_WMMSE


def Schulz(W, E, I):
    """
    Schulz's iteration for the matrix inverse
    :param W:  the weight matrix in WMMSE
    :param E:  the error matrix
    :param I:  the identity matrix
    :return:   the approximate inverse of matrix W
    """
    W_temp = W @ (2 * I - E @ W)
    return (W_temp + W_temp.permute(0, 1, 3, 2)) * 0.5


def compute_WSR_neural_network(H, V, noise_power, user_weights_internal, batch_size_internal):
    """
    Compute the WSR of the system
    :param H: the channel between users and the BS
    :param V: the precoder of the BS
    :param noise_power: the power of noise
    :param batch_size_internal: the size of the batch for computing WSR
    :param user_weights_internal: the weights of the users
    :return: the WSR of the system
    """
    VVh_single_user = V @ V.permute(0, 1, 3, 2)
    VVh = torch.sum(VVh_single_user, dim=1, keepdim=True)
    VVh_other_users = VVh - VVh_single_user  # broadcast effect

    # create 2 * nr_of_UE_antennas * 2 * nr_of_UE_antennas identity matrix
    I = torch.eye(2 * nr_of_UE_antennas)
    I = I.reshape(1, 1, 2 * nr_of_UE_antennas, 2 * nr_of_UE_antennas)
    I = I.repeat(batch_size_internal, nr_of_users, 1, 1)

    HVVhHh_other_user = H @ VVh_other_users @ H.permute(0, 1, 3, 2)
    HVVhHh_single_user = H @ VVh_single_user @ H.permute(0, 1, 3, 2)

    rate_per_user = torch.multiply(user_weights_internal, 0.5 * (
        torch.log2(torch.linalg.det(HVVhHh_single_user @ torch.linalg.inv(HVVhHh_other_user + noise_power * I) + I))))

    average_rate = torch.sum(torch.sum(rate_per_user, dim=1), dim=0) / batch_size_internal

    # print("shape", batch_size_internal)
    return average_rate


# Nesterov GD iteration of the V update in the unfolded matrix-inverse-free WMMSE and computes the optimal step size
def GD_step_V_line_search_more_streams_Nesterov(init, init_momentum1, init_momentum2, mse_weights, user_weights, U, H,
                                                V, V_past, A, total_power, noise_power):
    """
    Gradient descent step for the transmitter precoder
    :param init: initial value for the coefficients of the gradient descent step
    :param init_momentum1: the first momentum of the gradient descent step
    :param init_momentum2: the second momentum of the gradient descent step
    :param init: initial value for the coefficients of the gradient descent step
    :param mse_weights: W in the WMMSE algorithm
    :param user_weights: weights of the users
    :param U: receiver precoder
    :param H: channel matrix
    :param V: transmitter precoder
    :param V_past: the past value of V
    :param A: a matrix related to mse calculation
    :param noise_power: the power of noise
    :param total_power: the total power
    :return: updated_transmitter_precoder, V, step_size_factor_temp, momentum1, momentum2
    """
    epsilon_numerical_instability = 10 ** (-9)
    step_size_factor_temp = init
    step_size_factor = 2 * torch.sigmoid(step_size_factor_temp)
    momentum1 = init_momentum1
    momentum2 = init_momentum2

    I = torch.eye(nr_of_BS_antennas)
    I = I.reshape(1, nr_of_BS_antennas, nr_of_BS_antennas)
    I = I.repeat(batch_size, 1, 1)

    # print(V.shape, V_past.shape)

    V_a = V + momentum1 * (V - V_past)
    V_b = V + momentum2 * (V - V_past)

    Uh = U.permute(0, 1, 3, 2)
    WUhU = mse_weights @ Uh @ U

    real_sum_trace_WUhU = torch.multiply(
        torch.unsqueeze(torch.unsqueeze(torch.sum(torch.einsum('abii->ab', WUhU) * 0.5, dim=-1), dim=-1), dim=-1), I)
    imag_sum_trace_WUhU = torch.multiply(torch.unsqueeze(
        torch.unsqueeze(
            torch.sum(torch.einsum('abii->ab', WUhU[:, :, -nr_of_data_streams:, :nr_of_data_streams]), dim=-1),
            dim=-1), dim=-1), I)

    real_sum_trace_WUhU_exp = torch.tile(torch.unsqueeze(real_sum_trace_WUhU, dim=1), (1, nr_of_users, 1, 1))
    imag_sum_trace_WUhU_exp = torch.tile(torch.unsqueeze(imag_sum_trace_WUhU, dim=1), (1, nr_of_users, 1, 1))

    sum_trace_WUhU_first_row = torch.concat((real_sum_trace_WUhU_exp, -1 * imag_sum_trace_WUhU_exp), dim=3)
    sum_trace_WUhU_second_row = torch.concat((imag_sum_trace_WUhU_exp, real_sum_trace_WUhU_exp), dim=3)
    sum_trace_WUhU = torch.concat((sum_trace_WUhU_first_row, sum_trace_WUhU_second_row), dim=2)

    gradient = 2 * (A @ V_b) - 2 * torch.multiply(user_weights,
                                                  H.permute(0, 1, 3, 2) @ U @ mse_weights) + 2 * noise_power * (
                       1 / total_power) * (sum_trace_WUhU @ V_b)

    ######################################
    # FIND OPTIMAL STEP SIZE##############
    ######################################

    ######################################
    ## NUMERATOR
    ######################################

    # FIRST AND SECOND TERMS NUMERATOR
    Vh = V_a.permute(0, 1, 3, 2)
    GVh = gradient @ Vh
    VGh = GVh.permute(0, 1, 3, 2)
    HG = H @ gradient

    HhU = H.permute(0, 1, 3, 2) @ U

    UhH = HhU.permute(0, 1, 3, 2)

    HhUWUhH = HhU @ mse_weights @ UhH

    HhUWUhH_sum = torch.sum(HhUWUhH, dim=1, keepdim=True)

    # first_term_numerator = torch.multiply(torch.einsum('abii->ab', HhUWUhH_sum @ VGh), user_weights[:, :, 0, 0])
    # Gh = gradient.permute(0, 1, 3, 2)
    # second_term_numerator = -1 * torch.multiply(torch.einsum('abii->ab', mse_weights @ Gh @ HhU),
    #                                             user_weights[:, :, 0, 0])
    #
    # trace_WUhU = torch.multiply(torch.einsum('abii->ab', WUhU), user_weights[:, :, 0, 0])
    # trace_WUhU_sum = torch.sum(trace_WUhU, dim=1, keepdim=True)
    #
    # third_term_numerator = noise_power * (1 / total_power) * torch.einsum('abii->ab', GVh) * trace_WUhU_sum
    #
    # numerator = first_term_numerator + second_term_numerator + third_term_numerator
    # GGh = gradient @ Gh
    # first_term_denominator = torch.multiply(torch.einsum('abii->ab', HhUWUhH_sum @ GGh), user_weights[:, :, 0, 0])
    # second_term_denominator = noise_power * (1 / total_power) * torch.einsum('abii->ab', GGh) * trace_WUhU_sum
    # denominator = first_term_denominator + second_term_denominator
    # step_size = torch.unsqueeze(torch.unsqueeze((numerator / denominator), dim=-1), dim=-1)

    first_term_numerator = torch.sum(
        torch.multiply(
            0.5 * torch.einsum('abii->ab', mse_weights @ UhH @ GVh @ HhU), user_weights[:, :, 0, 0]), dim=1)
    second_term_numerator = torch.sum(
        torch.multiply(
            0.5 * torch.einsum('abii->ab', mse_weights @ UhH @ VGh @ HhU), user_weights[:, :, 0, 0]), dim=1)
    # 这两个分子项实际上是相同的，因为在Schulz迭代中，W已经魔改变成共轭对称矩阵了

    # THIRD AND FOURTH TERMS NUMERATOR
    Gh = gradient.permute(0, 1, 3, 2)
    third_term_numerator = torch.sum(
        torch.multiply(0.5 * torch.einsum('abii->ab', mse_weights @ Gh @ HhU), user_weights[:, :, 0, 0]), dim=1)
    fourth_term_numerator = torch.sum(
        torch.multiply(0.5 * torch.einsum('abii->ab', mse_weights @ Uh @ HG), user_weights[:, :, 0, 0]), dim=1)
    # 这两个分子项实际上是相同的，因为在Schulz迭代中，W已经魔改变成共轭对称矩阵了

    # FIFTH AND SIXTH TERMS NUMERATOR
    GVh_all_users = torch.sum(GVh, dim=1)
    # print(GVh_all_users.shape)
    GVh_other_users = torch.tile(torch.unsqueeze(GVh_all_users, dim=1), (1, nr_of_users, 1, 1)) - GVh
    VGh_other_users = GVh_other_users.permute(0, 1, 3, 2)

    fifth_term_numerator = torch.sum(torch.multiply(0.5 * torch.einsum('abii->ab',
                                                                       mse_weights @ UhH @ GVh_other_users @ HhU)
                                                    , user_weights[:, :, 0, 0]), dim=1)
    sixth_term_numerator = torch.sum(
        torch.multiply(
            0.5 * torch.einsum('abii->ab', mse_weights @ UhH @ VGh_other_users @ HhU), user_weights[:, :, 0, 0]), dim=1)
    # 这两个分子项实际上是相同的，因为在Schulz迭代中，W已经魔改变成共轭对称矩阵了

    # SEVENTH TERM NUMERATOR
    trace_WUhU = 0.5 * torch.einsum('abii->ab', WUhU)

    seventh_term_numerator = torch.sum(torch.multiply(torch.multiply(
        torch.tile(
            torch.unsqueeze(-1 * noise_power * (1 / total_power) * torch.einsum('aii->a', GVh_all_users), dim=-1),
            (1, nr_of_users)), trace_WUhU), user_weights[:, :, 0, 0]), dim=1)  # B

    numerator = (
            first_term_numerator + second_term_numerator - third_term_numerator - fourth_term_numerator +
            fifth_term_numerator + sixth_term_numerator - seventh_term_numerator)

    ############################################################################
    # DENOMINATOR
    ############################################################################

    # FIRST TERM DENOMINATOR
    GGh = gradient @ Gh
    first_term_denominator = torch.sum(
        torch.multiply(
            0.5 * torch.einsum('abii->ab', mse_weights @ UhH @ GGh @ HhU),
            user_weights[:, :, 0, 0]), dim=-1)

    # SECOND TERM DENOMINATOR
    GGh_all_users = torch.sum(GGh, dim=1)
    GGh_other_users = torch.tile(torch.unsqueeze(GGh_all_users, dim=1), (1, nr_of_users, 1, 1)) - GGh

    second_term_denominator = torch.sum(torch.multiply(0.5 * torch.einsum('abii->ab', mse_weights @ UhH @
                                                                          GGh_other_users @ HhU),
                                                       user_weights[:, :, 0, 0]), dim=-1)
    # 这两个分母项是不同的

    # THIRD TERM DENOMINATOR
    GGh_trace = torch.tile(torch.unsqueeze(torch.sum(0.5 * torch.einsum('abii->ab', GGh), dim=1), dim=-1),
                           (1, nr_of_users))
    third_term_denominator = noise_power * (1 / total_power) * torch.sum(
        torch.multiply(torch.multiply(GGh_trace, trace_WUhU), user_weights[:, :, 0, 0]), dim=1)

    denominator = 2 * (first_term_denominator + second_term_denominator + third_term_denominator)

    step_size = torch.unsqueeze(torch.unsqueeze(torch.tile(torch.unsqueeze((numerator /
                                                                            (
                                                                                    denominator + epsilon_numerical_instability)),
                                                                           dim=-1), (1, nr_of_users)), dim=-1), dim=-1)

    # step_size = torch.ones((batch_size, nr_of_users, 1, 1))
    # step_size_factor = step_size_factor.reshape(1, nr_of_users, 1, 1).repeat(batch_size, 1, 1, 1)
    # print(step_size_factor, step_size.shape)
    # exit()
    updated_transmitter_precoder = V_a - step_size * step_size_factor * gradient
    return updated_transmitter_precoder, V, step_size_factor_temp, momentum1, momentum2


# Nesterov GD iteration of the U update in the unfolded matrix-inverse-free WMMSE and computes the optimal step size
def GD_step_U_line_search_more_streams_Nesterov(init, init_momentum1, init_momentum2, mse_weights, user_weights,
                                                U, U_past, H, V, VVh, total_power, noise_power):
    """
    Gradient descent step for the receiver precoder
    :param init: initial value for the coefficients of the gradient descent step
    :param init_momentum1: the first momentum of the gradient descent step
    :param init_momentum2: the second momentum of the gradient descent step
    :param mse_weights: W in the WMMSE algorithm
    :param user_weights: weights of the users
    :param U: receiver precoder
    :param U_past: the past value of U
    :param H: channel matrix
    :param V: transmitter precoder
    :param VVh: V * Vh
    :param total_power: the total power
    :param noise_power: the power of noise
    :return: updated_receiver_precoder, U, step_size_factor_temp, momentum1, momentum2
    """
    epsilon_numerical_instability = 10 ** (-9)
    step_size_factor_temp = init
    step_size_factor = 2 * torch.sigmoid(step_size_factor_temp)
    momentum1 = init_momentum1
    momentum2 = init_momentum2

    U_a = U + momentum1 * (U - U_past)
    U_b = U + momentum2 * (U - U_past)

    first_term = -1 * H @ V
    second_term = H @ VVh @ H.permute(0, 1, 3, 2) @ U_b

    power_V = torch.tile(torch.unsqueeze(
        torch.tile(torch.unsqueeze(torch.tile(torch.einsum('abii->ab', VVh) * 0.5, (1, nr_of_users)), dim=-1),
                   (1, 1, 2 * nr_of_UE_antennas)), dim=-1), (1, 1, 1, 2 * nr_of_data_streams))
    third_term = noise_power * (1 / total_power) * torch.multiply(U_b, power_V)

    gradient = 2 * torch.multiply(user_weights, (first_term + second_term + third_term) @ mse_weights)

    ######################################
    # FIND OPTIMAL STEP SIZE##############
    ######################################

    #############################
    # NUMERATOR
    #############################
    power_V = torch.tile(torch.unsqueeze(
        torch.tile(torch.unsqueeze(torch.tile(torch.einsum('abii->ab', VVh) * 0.5, (1, nr_of_users)), dim=-1),
                   (1, 1, 2 * nr_of_data_streams)), dim=-1), (1, 1, 1, 2 * nr_of_data_streams))

    first_term_num = 0.5 * torch.einsum('abii->ab', mse_weights @ U_a.permute(0, 1, 3, 2) @ H @ VVh @
                                        H.permute(0, 1, 3, 2) @ gradient)
    second_term_num = 0.5 * torch.einsum('abii->ab',
                                         mse_weights @ gradient.permute(0, 1, 3, 2) @ H @ VVh
                                         @ H.permute(0, 1, 3, 2) @ U_a)
    # 这两个分子项实际上是相同的，因为在Schulz迭代中，W已经魔改变成共轭对称矩阵了

    third_term_num = 0.5 * torch.einsum('abii->ab', mse_weights @ (
            noise_power * (1 / total_power) * torch.multiply(U_a.permute(0, 1, 3, 2) @ gradient, power_V)))
    fourth_term_num = 0.5 * torch.einsum('abii->ab', mse_weights @ (
            noise_power * (1 / total_power) * torch.multiply(gradient.permute(0, 1, 3, 2) @ U_a, power_V)))
    # 这两个分子项实际上是相同的，因为在Schulz迭代中，W已经魔改变成共轭对称矩阵了

    fifth_term_num = -1 * 0.5 * torch.einsum('abii->ab',
                                             mse_weights @ V.permute(0, 1, 3, 2) @ H.permute(0, 1, 3, 2) @ gradient)
    sixth_term_num = -1 * 0.5 * torch.einsum('abii->ab', mse_weights @ gradient.permute(0, 1, 3, 2) @ H @ V)
    # 这两个分子项实际上是相同的，因为在Schulz迭代中，W已经魔改变成共轭对称矩阵了

    numerator = first_term_num + second_term_num + third_term_num + fourth_term_num + fifth_term_num + sixth_term_num

    #############################
    # DENOMINATOR
    ##############################

    first_term_den = torch.einsum('abii->ab', mse_weights @ (
            noise_power * (1 / total_power) * torch.multiply(gradient.permute(0, 1, 3, 2) @ gradient, power_V)))
    second_term_den = torch.einsum('abii->ab', mse_weights @ gradient.permute(0, 1, 3, 2) @ H @ VVh
                                   @ H.permute(0, 1, 3, 2) @ gradient)
    denominator = first_term_den + second_term_den + epsilon_numerical_instability

    step_size = torch.unsqueeze(torch.unsqueeze((numerator / denominator), dim=-1), dim=-1)

    # print(step_size)
    # exit()
    # step_size = torch.ones((batch_size, nr_of_users, 1, 1))
    # step_size_factor = step_size_factor.reshape(1, nr_of_users, 1, 1).repeat(batch_size, 1, 1, 1)

    updated_receiver_precoder = U_a - step_size * step_size_factor * gradient

    return updated_receiver_precoder, U, step_size_factor_temp, momentum1, momentum2


# </editor-fold>

# <editor-fold desc="Test the unit">
if __name__ == '__main__':
    # Only for testing the functions
    channel_nn, initial_transmitter_precoder, initial_receiver_precoder, channel_WMMSE, initial_transmitter_precoder_WMMSE = compute_channel(
        nr_of_BS_antennas, nr_of_users, Total_Power, Noise_Power)
    # channel_nn[i] indicates the channel matrix of the i-th user in the format of numpy array
    channel_nn_torch = torch.zeros((1, nr_of_users, 2 * nr_of_UE_antennas, 2 * nr_of_BS_antennas))
    initial_transmitter_torch = torch.zeros((1, nr_of_users, 2 * nr_of_BS_antennas, 2 * nr_of_data_streams))
    initial_receiver_torch = torch.zeros((1, nr_of_users, 2 * nr_of_UE_antennas, 2 * nr_of_data_streams))
    for i in range(nr_of_users):
        channel_nn_torch[0, i, :, :] = torch.tensor(channel_nn[i])
        initial_transmitter_torch[0, i, :, :] = torch.tensor(initial_transmitter_precoder[i])
        initial_receiver_torch[0, i, :, :] = torch.tensor(initial_receiver_precoder[i])
    # print(channel_nn[0].shape, initial_transmitter_precoder[0].shape, initial_receiver_precoder[0].shape,
    #       channel_WMMSE.shape, initial_transmitter_precoder_WMMSE.shape)
    # print(channel_nn[0])
    # print(channel_WMMSE[0])
    # save channel
    sio.savemat('H_WMMSE.mat', {'H_WMMSE': channel_WMMSE.transpose(1, 2, 0)})

    begin_time = time.time()
    last_V, last_U, last_W, last_WSR = run_WMMSE_MIMO_more_streams(epsilon, channel_WMMSE,
                                                                   initial_transmitter_precoder_WMMSE,
                                                                   Total_Power, Noise_Power,
                                                                   user_weights_WMMSE, nr_of_iterations, log=False)
    time1 = time.time()
    average_rate1 = compute_weighted_sum_rate_MIMO(user_weights_WMMSE, channel_WMMSE, initial_transmitter_precoder_WMMSE
                                                   , Noise_Power)

    print("Average_initial_Rate: ", average_rate1, "\n" "elapsed time: ", time1 - begin_time)

    time2 = time.time()
    average_rate2 = compute_WSR_neural_network(channel_nn_torch,
                                               initial_transmitter_torch, Noise_Power,
                                               torch.tensor(user_weights_WMMSE), 1)
    print("Average_Unfolding_Rate: ", average_rate2, "\n" "elapsed time: ", time2 - time1)

    end_time = time.time()
    print("last_WSR", last_WSR, "\n", "time_consumption:", end_time - time2)

    # channel_WMMSE_batch = channel_WMMSE.reshape(1, nr_of_users, nr_of_UE_antennas, nr_of_BS_antennas)
    # channel_WMMSE_batch = np.tile(channel_WMMSE_batch, (batch_size, 1, 1, 1))
    #
    # initial_transmitter_precoder_WMMSE_batch = initial_transmitter_precoder_WMMSE.reshape(1, nr_of_users,
    #                                                                                       nr_of_BS_antennas,
    #                                                                                       nr_of_data_streams)
    # initial_transmitter_precoder_WMMSE_batch = np.tile(initial_transmitter_precoder_WMMSE_batch, (batch_size, 1, 1, 1))
    #
    # V_batch, U_batch, W_batch, WSR_batch = run_WMMSE_MIMO_more_streams_vectorize(epsilon, channel_WMMSE_batch,
    #                                                                              initial_transmitter_precoder_WMMSE_batch,
    #                                                                              Total_Power, Noise_Power,
    #                                                                              user_weights_WMMSE, nr_of_iterations)

# </editor-fold>
