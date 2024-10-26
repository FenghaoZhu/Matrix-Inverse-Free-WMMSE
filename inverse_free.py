"""
Inverse-Free Algorithm for WMMSE.
Author: zfh
Date: 2024.1.4
"""

# <editor-fold desc="Import Packages">
from util import *
import torch
import torch.nn as nn

# </editor-fold>


# <editor-fold desc="Define the parameters">
# Define the parameters
WSR_WMMSE = []  # to store the WSR attained by the WMMSE
WSR_nn = []  # to store the WSR attained by the deep unfolded WMMSE
training_loss = []  # to store the training loss
# Define the initial value for the Nesterov acceleration scheme
initial_transmitter_precoder_batch_past = torch.zeros(batch_size, nr_of_users, nr_of_BS_antennas * 2,
                                                      2 * nr_of_data_streams)
initial_receiver_precoder_batch_past = torch.zeros(batch_size, nr_of_users, nr_of_UE_antennas * 2,
                                                   2 * nr_of_data_streams)

# Arrays that contain the initialization values of the step sizes for receiver precoder
step_size_init_U = torch.zeros(J_u, nr_of_iterations, requires_grad=True)
momentum1_init_U = torch.zeros(J_u, nr_of_iterations, requires_grad=True)
momentum2_init_U = torch.zeros(J_u, nr_of_iterations, requires_grad=True)

# Arrays that contain the initialization values of the step sizes for transmitter precoder
step_size_init_V = torch.zeros(J_v, nr_of_iterations, requires_grad=True)
momentum1_init_V = torch.zeros(J_v, nr_of_iterations, requires_grad=True)
momentum2_init_V = torch.zeros(J_v, nr_of_iterations, requires_grad=True)

parameters = [
    step_size_init_U,
    momentum1_init_U,
    momentum2_init_U,
    step_size_init_V,
    momentum1_init_V,
    momentum2_init_V,
]

optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=1e-4)
np.random.seed(5678)  # For reproducibility
# </editor-fold>


# <editor-fold desc="Run the inverse-free WMMSE algorithm">
for i in range(nr_of_batches_training):
    batch_for_training = torch.zeros(batch_size, nr_of_users, 2 * nr_of_data_streams, nr_of_BS_antennas * 2)
    initial_transmitter_precoder_batch = torch.zeros(batch_size, nr_of_users, nr_of_BS_antennas * 2,
                                                     2 * nr_of_data_streams, )
    initial_receiver_precoder_batch = torch.zeros(batch_size, nr_of_users, 2 * nr_of_data_streams,
                                                  2 * nr_of_data_streams)

    WSR_WMMSE_batch = 0.0
    # Building a batch for training
    for ii in range(batch_size):
        (channel_realization_nn, init_transmitter_precoder, init_receiver_precoder,
         channel_WMMSE, initial_transmitter_precoder_WMMSE) = compute_channel(
            nr_of_BS_antennas, nr_of_users, Total_Power)
        # print("channel_realization_nn.shape: ", channel_realization_nn.shape,
        #       "init_transmitter_precoder.shape: ", init_transmitter_precoder.shape,
        #       "init_receiver_precoder.shape: ", init_receiver_precoder.shape)

        # Run the WMMSE algorithm
        _, _, _, WSR_WMMSE_one_sample = run_WMMSE_MIMO_more_streams(epsilon, channel_WMMSE,
                                                                    initial_transmitter_precoder_WMMSE, Total_Power,
                                                                    Noise_Power, user_weights_WMMSE,
                                                                     nr_of_iterations_WMMSE, log=False)
        # Accumulate the achieved WSR by WMMSE
        WSR_WMMSE_batch = WSR_WMMSE_batch + WSR_WMMSE_one_sample

        # Fill the batch for training
        for user_index in range(nr_of_users):
            batch_for_training[ii, user_index, :, :] = torch.tensor(channel_realization_nn[user_index])
            initial_transmitter_precoder_batch[ii, user_index, :, :] = torch.tensor(
                init_transmitter_precoder[user_index])
            initial_receiver_precoder_batch[ii, user_index, :, :] = torch.tensor(init_receiver_precoder[user_index])

    print("Training: The WSR achieved with the WMMSE algorithm is: ", WSR_WMMSE_batch / batch_size)

    # load the sample in each batch
    channel_input = batch_for_training
    initial_tp = initial_transmitter_precoder_batch
    initial_tp_past = initial_transmitter_precoder_batch_past
    initial_rp = initial_receiver_precoder_batch
    initial_rp_past = initial_receiver_precoder_batch_past

    initial_transmitter_precoder = initial_tp
    initial_transmitter_precoder_past = initial_tp_past
    initial_receiver_precoder = initial_rp
    initial_receiver_precoder_past = initial_rp_past

    profit = []  # stores the WSR obtained at each iteration
    profit_alternative = []  # stores the WSR (computed through the mse weights) at each iteration for the training

    user_weights_U_expanded = user_weights.reshape(batch_size, nr_of_users, 1, 1)
    user_weights_U_expanded = user_weights_U_expanded.repeat(1, 1, 2 * nr_of_UE_antennas, 2 * nr_of_data_streams)

    user_weights_A_expanded = user_weights.reshape(batch_size, nr_of_users, 1, 1)
    user_weights_A_expanded = user_weights_A_expanded.repeat(1, 1, 2 * nr_of_BS_antennas, 2 * nr_of_BS_antennas)

    user_weights_V_expanded = user_weights.reshape(batch_size, nr_of_users, 1, 1)
    user_weights_V_expanded = user_weights_V_expanded.repeat(1, 1, 2 * nr_of_BS_antennas, 2 * nr_of_data_streams)

    ####################################
    # UPDATE OF MSE WEIGHTS
    ####################################
    I = torch.eye(2 * nr_of_data_streams, 2 * nr_of_data_streams)
    I = I.reshape((1, 1, 2 * nr_of_data_streams, 2 * nr_of_data_streams))
    I = I.repeat(batch_size, nr_of_users, 1, 1)

    # print(initial_receiver_precoder.shape, channel_input.shape, initial_transmitter_precoder.shape)
    # FIRST TERM OF E
    I_UhHV = I - initial_receiver_precoder.permute(0, 1, 3, 2) @ channel_input @ initial_transmitter_precoder
    first_term = I_UhHV @ I_UhHV.permute(0, 1, 3, 2)

    # SECOND TERM OF E
    VVh_single_user = initial_transmitter_precoder @ initial_transmitter_precoder.permute(0, 1, 3, 2)
    VVh = torch.unsqueeze(torch.sum(VVh_single_user, dim=1), dim=1)
    VVh_other_users = VVh - VVh_single_user
    UhHVVhHhU = (initial_receiver_precoder.permute(0, 1, 3, 2) @ channel_input
                 @ VVh_other_users @ channel_input.permute(0, 1, 3, 2) @ initial_receiver_precoder)
    second_term = UhHVVhHhU

    # THIRD TERM OF E
    power_V = torch.tile(torch.unsqueeze(
        torch.tile(torch.unsqueeze(torch.tile(torch.einsum('abii->ab', VVh) * 0.5, (1, nr_of_users)), dim=-1),
                   (1, 1, 2 * nr_of_data_streams)), dim=-1), (1, 1, 1, 2 * nr_of_data_streams))
    third_term = Noise_Power * (1 / Total_Power) * torch.multiply(
        initial_receiver_precoder.permute(0, 1, 3, 2) @ initial_receiver_precoder, power_V)

    E = first_term + second_term + third_term

    ###########################################
    # SCHULZ WITH SPECTRAL RADIUS NORMALIZATION
    ###########################################

    mse_weights_init = torch.multiply(I, torch.tile(torch.unsqueeze(
        torch.tile(torch.unsqueeze(torch.reciprocal(0.5 * torch.einsum('abii->ab', E)), dim=-1),
                   (1, 1, 2 * nr_of_data_streams)),
        dim=-1), (1, 1, 1, 2 * nr_of_data_streams)))

    D = E @ mse_weights_init
    abs_D = torch.sqrt(
        (D ** 2)[:, :, :nr_of_data_streams, :nr_of_data_streams] + (D ** 2)[:, :, :nr_of_data_streams,
                                                                   -nr_of_data_streams:])
    sum_abs_D = torch.unsqueeze(torch.sum(abs_D, dim=2), dim=-1)

    g = torch.amax(abs_D @ sum_abs_D, dim=[-2, -1])
    scaling = torch.sqrt(
        torch.tile(torch.unsqueeze(torch.tile(torch.unsqueeze(g, dim=-1), (1, 1, 2 * nr_of_data_streams)), dim=-1),
                   (1, 1, 1, 2 * nr_of_data_streams)))

    mse_weights = torch.divide(mse_weights_init, scaling)

    # Re-normalize W in each algorithm iteration before updating W
    for j in range(J_w):
        mse_weights = Schulz(mse_weights, E, I)

    for loop in range(0, L):

        # To update VVh for receiver precoder
        VVh = torch.unsqueeze(torch.sum(
            initial_transmitter_precoder @ initial_transmitter_precoder.permute(0, 1, 3, 2),
            dim=1), dim=1)

        #######################################
        # UPDATE OF RECEIVER PRECODER
        #######################################
        temp_receiver_precoder = initial_receiver_precoder
        temp_receiver_precoder_past = initial_receiver_precoder_past

        # the gradient descent steps of the receiver precoder
        for i_u in range(J_u):
            temp_receiver_precoder, temp_receiver_precoder_past, step_size1_U, momentum1_1_U, momentum2_1_U \
                = GD_step_U_line_search_more_streams_Nesterov(step_size_init_U[i_u, loop],
                                                              momentum1_init_U[i_u, loop], momentum2_init_U[i_u, loop],
                                                              mse_weights,
                                                              user_weights_U_expanded, temp_receiver_precoder,
                                                              temp_receiver_precoder_past, channel_input,
                                                              initial_transmitter_precoder, VVh, Total_Power,
                                                              Noise_Power)
        receiver_precoder_final = temp_receiver_precoder
        ####################################
        # UPDATE OF MSE WEIGHTS
        ####################################

        # FIRST TERM OF E
        I_UhHV = I - receiver_precoder_final.permute(0, 1, 3, 2) @ channel_input @ initial_transmitter_precoder
        first_term = I_UhHV @ I_UhHV.permute(0, 1, 3, 2)

        # SECOND TERM OF E
        VVh_single_user = initial_transmitter_precoder @ initial_transmitter_precoder.permute(0, 1, 3, 2)
        VVh = torch.unsqueeze(torch.sum(VVh_single_user, dim=1), dim=1)
        VVh_other_users = VVh - VVh_single_user
        UhHVVhHhU = (receiver_precoder_final.permute(0, 1, 3, 2) @ channel_input @ VVh_other_users
                     @ channel_input.permute(0, 1, 3, 2) @ receiver_precoder_final)
        second_term = UhHVVhHhU

        # THIRD TERM OF E
        power_V = torch.tile(torch.unsqueeze(
            torch.tile(torch.unsqueeze(torch.tile(torch.einsum('abii->ab', VVh) * 0.5, (1, nr_of_users)), dim=-1),
                       (1, 1, 2 * nr_of_data_streams)), dim=-1), (1, 1, 1, 2 * nr_of_data_streams))
        third_term = Noise_Power * (1 / Total_Power) * torch.multiply(
            receiver_precoder_final.permute(0, 1, 3, 2) @ receiver_precoder_final, power_V)

        E = first_term + second_term + third_term

        ###########################################
        # SCHULZ WITH SPECTRAL RADIUS NORMALIZATION
        ###########################################

        D = E @ mse_weights
        abs_D = torch.sqrt(
            (D ** 2)[:, :, :nr_of_data_streams, :nr_of_data_streams] + (D ** 2)[:, :, :nr_of_data_streams,
                                                                       -nr_of_data_streams:])
        sum_abs_D = torch.unsqueeze(torch.sum(abs_D, dim=2), dim=-1)

        g = torch.amax(abs_D @ sum_abs_D, dim=[-2, -1])
        scaling = torch.sqrt(
            torch.tile(
                torch.unsqueeze(torch.tile(torch.unsqueeze(g, dim=-1), (1, 1, 2 * nr_of_data_streams)), dim=-1),
                (1, 1, 1, 2 * nr_of_data_streams)))

        mse_weights = torch.divide(mse_weights, scaling)

        # the Schulz iteration
        for j in range(J_w):
            mse_weights = Schulz(mse_weights, E, I)

        ##########################################
        # UPDATE OF TRANSMITTER PRECODER
        ##########################################

        A = torch.unsqueeze(torch.sum(torch.multiply(user_weights_A_expanded, channel_input.permute(0, 1, 3, 2)
                                                     @ receiver_precoder_final @ mse_weights
                                                     @ receiver_precoder_final.permute(0, 1, 3, 2) @ channel_input),
                                      dim=1), dim=1)
        temp_transmitter_precoder = initial_transmitter_precoder
        temp_transmitter_precoder_past = initial_transmitter_precoder_past

        # print(initial_transmitter_precoder.shape, initial_transmitter_precoder_past.shape)

        # the gradient descent steps of the transmitter precoder
        for i_v in range(J_v):
            temp_transmitter_precoder, temp_transmitter_precoder_past, step_size1_V, momentum1_1_V, momentum2_1_V = (
                GD_step_V_line_search_more_streams_Nesterov(step_size_init_V[i_v, loop],
                                                            momentum1_init_V[i_v, loop], momentum2_init_V[i_v, loop],
                                                            mse_weights, user_weights_V_expanded,
                                                            receiver_precoder_final, channel_input,
                                                            temp_transmitter_precoder, temp_transmitter_precoder_past,
                                                            A, Total_Power, Noise_Power))
        transmitter_precoder_final = temp_transmitter_precoder
        ##############################################################################
        # For the next loop
        initial_transmitter_precoder = transmitter_precoder_final
        initial_receiver_precoder = receiver_precoder_final
        ##############################################################################

        ##############################################################################
        if scale_V_every_iteration == True:
            transmitter_precoder_power = torch.unsqueeze(
                torch.sum((0.5 * (torch.norm(transmitter_precoder_final, dim=[-2, -1])) ** 2), dim=1),
                dim=-1)
            power_scaling_ref = torch.divide(torch.tensor(1), torch.sqrt(transmitter_precoder_power)) * torch.sqrt(
                torch.tensor(Total_Power))
            power_scaling_expanded = torch.tile(torch.unsqueeze(
                torch.tile(torch.unsqueeze(torch.tile(power_scaling_ref, (1, nr_of_users)), dim=-1),
                           (1, 1, 2 * nr_of_BS_antennas)), dim=-1), (1, 1, 1, 2 * nr_of_data_streams))
            initial_transmitter_precoder = torch.multiply(transmitter_precoder_final, power_scaling_expanded)
        ##############################################################################

        # LOSS FUNCTION
        # scale the transmit precoder at the last iteration
        if loop == (nr_of_iterations_nn - 1.0) and scale_V_every_iteration == False:
            transmitter_precoder_power = torch.unsqueeze(
                torch.sum((0.5 * (torch.norm(transmitter_precoder_final, dim=[-2, -1])) ** 2), dim=1), dim=-1)
            power_scaling_ref = torch.divide(torch.tensor(1), torch.sqrt(transmitter_precoder_power)) * torch.sqrt(
                torch.tensor(Total_Power))
            power_scaling_expanded = torch.tile(torch.unsqueeze(
                torch.tile(torch.unsqueeze(torch.tile(power_scaling_ref, (1, nr_of_users)), dim=-1),
                           (1, 1, 2 * nr_of_BS_antennas)), dim=-1), (1, 1, 1, 2 * nr_of_data_streams))
            initial_transmitter_precoder = torch.multiply(transmitter_precoder_final, power_scaling_expanded)

        profit.append(
            compute_WSR_neural_network(channel_input, initial_transmitter_precoder, Noise_Power, user_weights,
                                       batch_size))

        if loop == (nr_of_iterations_nn - 2.0):

            # we scale the final precoder to meet the power constraint
            if scale_V_every_iteration == False:
                transmitter_precoder_power = torch.unsqueeze(
                    torch.sum((0.5 * (torch.norm(transmitter_precoder_final, dim=[-2, -1])) ** 2), dim=1),
                    dim=-1)
                power_scaling_ref = (torch.divide(torch.tensor(1), torch.sqrt(transmitter_precoder_power))
                                     * torch.sqrt(torch.tensor(Total_Power)))
                power_scaling_expanded = torch.tile(torch.unsqueeze(
                    torch.tile(torch.unsqueeze(torch.tile(power_scaling_ref, (1, nr_of_users)), dim=-1),
                               (1, 1, 2 * nr_of_BS_antennas)), dim=-1), (1, 1, 1, 2 * nr_of_data_streams))
                initial_transmitter_precoder = torch.multiply(transmitter_precoder_final, power_scaling_expanded)

            # compute the WSR given by transmitter_precoder_to_use
            WSR_from_V_previous_iteration = compute_WSR_neural_network(channel_input, initial_transmitter_precoder,
                                                                       Noise_Power, user_weights, batch_size)

    WSR = sum(profit)
    WSR_final = profit[-1]
    print(f"Training batch {i}: The WSR achieved with the Unfolding algorithm is: {WSR_final}")
    Loss = -WSR
    # print(step_size_factor1_init_U.grad)
    # print(step_size_factor1_init_U)
    optimizer.zero_grad()  # clear gradients for this training step
    Loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients optimization

# print("The WSR achieved with unfolded matrix-inverse-free WMMSE is: ", np.mean(WSR_nn))
# print("The WSR achieved with the WMMSE algorithm is: ", WSR_WMMSE_batch/(nr_of_samples_per_batch*nr_of_batches_test))
# </editor-fold>
