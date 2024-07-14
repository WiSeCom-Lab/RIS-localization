import pywarraychannels
import channels
import numpy as np
import numpy.matlib as npmatlib
import scipy
import hardware
import MOMP
import localization
from time import time

# Params
method = "MOMP"         # Channel estimation method (MOMP or OMP)
K_res = 128             # Method's dictionary resolution
K_res_lr = 4            # Method's dictionary low resolution
samples = 100           # Number of samples from the dataset to evaluate
scenario = 0            # Scenarios 0: No RIS, 1: No BS-UE, 2: Both LOS
loc_method = 0          # Localization method 0: DoA, 1: DoD
# Power
p_t_dBm_set = [20]             # dBm
# Noise related
T = 15                  # C
k_B = 1.38064852e-23    # Boltzmanz's constant
# Speed of light
c = 3e8                 # m/s
# Antennas
N_UE = 4                # Number of UE antennas in each dimension
N_AP = 8                # Number of AP antennas in each dimension
N_RIS = 16              # Number of RIS antennas in each dimension
N_RF_UE = 4             # Number of UE RF-chains in total
N_RF_AP = 8             # Number of AP RF-chains in total
N_M_UE = 4              # Number of UE measurements in each dimension
N_M_AP = 8              # Number of AP measurements in each dimension
N_M_RIS = 16            # Number of RIS measurements in total
# orientations_AP = [pywarraychannels.uncertainties.Static(roll=np.pi/2)]*6 + [pywarraychannels.uncertainties.Static(roll=-np.pi/2)]*6
orientation_RIS = pywarraychannels.uncertainties.Static(roll=-np.pi/2)
# Carriers
f_c = 60                # GHz
B = 1                   # GHz
K = 64                  # Number of delay taps
Q = 64                  # Length of the training pilot
# Estimation
N_est = 5               # Number of estimated paths


# Define pulse shape filter
filter = pywarraychannels.filters.RCFilter(early_samples=8, late_samples=8)

# Pilot signals
Pilot = np.concatenate([scipy.linalg.hadamard(Q)[:N_RF_AP], np.zeros((N_RF_AP, K//2))], axis=1)
# Pilot = np.concatenate([np.ones((N_RF_AP,Q)), np.zeros((N_RF_AP, K//2))], axis=1)
P_len = Pilot.shape[1]
D = K+filter.early_samples+filter.late_samples
Pilot = np.concatenate([np.zeros((Pilot.shape[0], D)), Pilot], axis=1)          # Zero-padding

# Define antennas
antenna_UE = pywarraychannels.antennas.RectangularAntenna((N_UE, N_UE))
antenna_AP = pywarraychannels.antennas.RectangularAntenna((N_AP, N_AP))
antenna_RIS = pywarraychannels.antennas.RectangularAntenna((N_RIS, N_RIS))
antenna_RIS.uncertainty = orientation_RIS

# Transform params to natural units
f_c *= 1e9
B *= 1e9
T += 273.1

# Load data
with open("data/AP_pos.txt") as f:
    AP_pos = [[float(el) for el in line.split()] for line in f.read().split("\n")[1:-1]]
with open("data/UE_pos.txt") as f:
    UE_pos = [[float(el) for el in line.split()] for line in f.read().split("\n")[1:-1]]
with open("data/RIS_pos.txt") as f:
    RIS_pos = [[float(el) for el in line.split()] for line in f.read().split("\n")[1:-1]]
with open("data/Info_BM.txt") as f:
    Rays_BM = [pywarraychannels.em.Geometric([[float(p) for p in line.split()] for line in ue_block.split("\n")], bool_flip_RXTX=False) for ue_block in f.read()[:-1].split("\n<ue>\n")]
with open("data/Info_RM.txt") as f:
    Rays_RM = [pywarraychannels.em.Geometric([[float(p) for p in line.split()] for line in ue_block.split("\n")], bool_flip_RXTX=False) for ue_block in f.read()[:-1].split("\n<ue>\n")]    
with open("data/Info_BR.txt") as f:
    Rays_BR = [pywarraychannels.em.Geometric([[float(p) for p in line.split()] for line in ue_block.split("\n")], bool_flip_RXTX=False) for ue_block in f.read()[:-1].split("\n<ue>\n")]

# Crop data
UE_pos, Rays_BM, Rays_RM = [X[:samples] for X in [UE_pos, Rays_BM, Rays_RM]]

# Define channel Geomtric-MIMO-AWGN
channel_Geometric = channels.Geometric(
    antenna_UE, antenna_AP, antenna_RIS, f_c=f_c,
    B=B, K=K, filter=filter, bool_sync=True)
channel_MIMO = channels.MIMO(channel_Geometric, pilot=Pilot)

# Define channel Geomtric-MIMO-AWGN for RM channel
channel_Geometric_RM = channels.Geometric(
    antenna_UE, antenna_RIS, antenna_RIS, f_c=f_c,
    B=B, K=K, filter=filter, bool_sync=True)
channel_MIMO_RM = channels.MIMO(channel_Geometric_RM, pilot=Pilot)

# Hardware impairments at the BS
M_B = hardware.MutualCoupling(N_AP*N_AP, ideal=True)
gamma_B = hardware.PhaseGainError(N_AP*N_AP, ideal=True)
MgammaB = np.dot(M_B, gamma_B)

# Hardware impairments at the RIS
M_R = hardware.MutualCoupling(N_RIS*N_RIS, ideal=True)
gamma_R = hardware.PhaseGainError(N_RIS*N_RIS, ideal=True)
MgammaR = np.dot(M_R, gamma_R)

# Hardware impairments at the MS
M_M = hardware.MutualCoupling(N_UE*N_UE, ideal=False)
gamma_M = hardware.PhaseGainError(N_UE*N_UE, ideal=False)
MgammaM = np.dot(M_M, gamma_M)

# Known LoS BS-RIS channel
RIS_DoA_az = np.radians(Rays_BR[0].ray_info[0,3])
RIS_DoA_el = np.radians(Rays_BR[0].ray_info[0,4])
RIS_DoD_az = np.radians(Rays_RM[0].ray_info[0,5])
RIS_DoD_el = np.radians(Rays_RM[0].ray_info[0,6])
BS_DoD_az = np.radians(Rays_BR[0].ray_info[0,5])
BS_DoD_el = np.radians(Rays_BR[0].ray_info[0,6])

RIS_DoA = np.array([np.cos(RIS_DoA_el)*np.cos(RIS_DoA_az), np.cos(RIS_DoA_el)*np.sin(RIS_DoA_az), np.sin(RIS_DoA_el)])
RIS_DoD = np.array([np.cos(RIS_DoD_el)*np.cos(RIS_DoD_az), np.cos(RIS_DoD_el)*np.sin(RIS_DoD_az), np.sin(RIS_DoD_el)])
BS_DoD = np.array([np.cos(BS_DoD_el)*np.cos(BS_DoD_az), np.cos(BS_DoD_el)*np.sin(BS_DoD_az), np.sin(BS_DoD_el)]) 

BR_pow = Rays_BR[0].ray_info[0,2]
BR_phase = np.radians(Rays_BR[0].ray_info[0,0])
BR_gain = np.power(10, (BR_pow-30)/20)*np.exp(1j*BR_phase)

scalar_RIS_DoA = antenna_RIS.scalar_dir(RIS_DoA)
scalar_RIS_DoD = antenna_RIS.scalar_dir(RIS_DoD)
scalar_BS_DoD = antenna_AP.scalar_dir(BS_DoD)

H_BR = BR_gain*np.exp(1j*np.pi*(scalar_RIS_DoA[:, np.newaxis]-scalar_BS_DoD[np.newaxis, :]))

# Sparse decomposition components
angles_AP_x = np.linspace(-np.pi, np.pi, int(N_AP*K_res))
angles_AP_y = np.linspace(-np.pi, np.pi, int(N_AP*K_res))
angles_UE_x = np.linspace(-np.pi, np.pi, int(N_UE*K_res))
angles_UE_y = np.linspace(-np.pi, np.pi, int(N_UE*K_res))
angles_RIS_x = np.linspace(-np.pi, np.pi, int(N_RIS*K_res))
angles_RIS_y = np.linspace(-np.pi, np.pi, int(N_RIS*K_res))
A_AP_x = np.exp(1j*np.arange(N_AP)[:, np.newaxis]*angles_AP_x[np.newaxis, :])
A_AP_y = np.exp(1j*np.arange(N_AP)[:, np.newaxis]*angles_AP_y[np.newaxis, :])
A_UE_x = np.exp(1j*np.arange(N_UE)[:, np.newaxis]*angles_UE_x[np.newaxis, :])
A_UE_y = np.exp(1j*np.arange(N_UE)[:, np.newaxis]*angles_UE_y[np.newaxis, :])
A_RIS_x = np.exp(1j*np.arange(N_RIS)[:, np.newaxis]*angles_RIS_x[np.newaxis, :])
A_RIS_y = np.exp(1j*np.arange(N_RIS)[:, np.newaxis]*angles_RIS_y[np.newaxis, :])
delays = np.linspace(0, K, int(K*K_res))
A_time = filter.response(K, delays)

A_UE = np.kron(A_UE_x,A_UE_y)

# Reduced dimensional sparse decomposition components
angles_AP_lr_x = np.linspace(-np.pi, np.pi, int(N_AP*K_res_lr))
angles_AP_lr_y = np.linspace(-np.pi, np.pi, int(N_AP*K_res_lr))
angles_UE_lr_x = np.linspace(-np.pi, np.pi, int(N_UE*K_res_lr))
angles_UE_lr_y = np.linspace(-np.pi, np.pi, int(N_UE*K_res_lr))
angles_RIS_lr_x = np.linspace(-np.pi, np.pi, int(N_RIS*K_res_lr))
angles_RIS_lr_y = np.linspace(-np.pi, np.pi, int(N_RIS*K_res_lr))
A_AP_lr_x = np.exp(1j*np.arange(N_AP)[:, np.newaxis]*angles_AP_lr_x[np.newaxis, :])
A_AP_lr_y = np.exp(1j*np.arange(N_AP)[:, np.newaxis]*angles_AP_lr_y[np.newaxis, :])
A_UE_lr_x = np.exp(1j*np.arange(N_UE)[:, np.newaxis]*angles_UE_lr_x[np.newaxis, :])
A_UE_lr_y = np.exp(1j*np.arange(N_UE)[:, np.newaxis]*angles_UE_lr_y[np.newaxis, :])
A_RIS_lr_x = np.exp(1j*np.arange(N_RIS)[:, np.newaxis]*angles_RIS_lr_x[np.newaxis, :])
A_RIS_lr_y = np.exp(1j*np.arange(N_RIS)[:, np.newaxis]*angles_RIS_lr_y[np.newaxis, :])
delays_lr = np.linspace(0, K, int(K*K_res_lr))
A_time_lr = filter.response(K, delays_lr)

# Dictionaries
if scenario == 0:
    X = [
        np.conj(A_AP_x),
        np.conj(A_AP_y),
        A_time
    ]
    X_lr = [
        np.conj(A_AP_lr_x),
        np.conj(A_AP_lr_y),
        A_time_lr
    ]
elif scenario == 1:
    X = [
        np.conj(A_RIS_x),
        np.conj(A_RIS_y),
        A_time
    ]
    X_lr = [
        np.conj(A_RIS_lr_x),
        np.conj(A_RIS_lr_y),
        A_time_lr
    ]
elif scenario == 2:
    X_AP = [
        np.conj(A_AP_x),
        np.conj(A_AP_y),
        A_time
    ]
    X_AP_lr = [
        np.conj(A_AP_lr_x),
        np.conj(A_AP_lr_y),
        A_time_lr
    ]
    X_RIS = [
        np.conj(A_RIS_x),
        np.conj(A_RIS_y),
        A_time
    ]
    X_RIS_lr = [
        np.conj(A_RIS_lr_x),
        np.conj(A_RIS_lr_y),
        A_time_lr
    ]        

for p_t_dBm in p_t_dBm_set:
            
    # Transmit power
    p_t = np.power(10, (p_t_dBm-30)/10)

    # Compute noise level
    p_n = k_B*T*B
    print("Noise level: {:.2f}dBm".format(10*np.log10(p_n)+30))
    
    # Create channels
    channel = channels.AWGN(channel_MIMO, power=p_t, noise=p_n)
    channel_RM = channels.AWGN(channel_MIMO_RM, power=p_t, noise=p_n)
    print("Channel")
    
    # Build channels and decompose them
    estimation = []
    UE_pos_est = []
    for rays_BM, rays_RM, ue_pos, ii_ue in zip(Rays_BM, Rays_RM, UE_pos, range(len(UE_pos))):
        
        # This is to check the quality of the localization
        # If the z component of the estimate is beyond some threshold,
        # the estimate does not make sense.
        # Then, the estimation is repeated.
        z_val = 100
        if scenario == 0:
            z_th = RIS_pos[0][2]
        elif scenario == 1:
            z_th = RIS_pos[0][2]
        else:
            z_th = RIS_pos[0][2]
        count = 0
        count_th = 3
        while z_val < 0 or z_val > z_th:
            print("Power: {}, User: {}/{}".format(p_t_dBm,ii_ue, samples))
            tic = time()

            # Define codebooks
            antenna_UE.set_reduced_codebook((N_M_UE, N_M_UE))
            if scenario == 0:
                antenna_AP.set_codebook(np.exp(1j*(2*np.pi*np.random.rand(N_AP*N_AP,N_M_AP*N_M_AP)-np.pi)))
            elif scenario == 1:
                antenna_AP.set_codebook(npmatlib.repmat(np.exp(1j*np.pi*scalar_BS_DoD), N_M_AP*N_M_AP, 1).T)
            elif scenario == 2:
                antenna_AP.set_codebook(np.concatenate((np.exp(1j*(2*np.pi*np.random.rand(N_AP*N_AP,int(N_M_AP*N_M_AP/2))-np.pi)),npmatlib.repmat(np.exp(1j*np.pi*scalar_BS_DoD), int(N_M_AP*N_M_AP/2), 1).T),axis=1))

            # Split codebooks according to number of RF-chains
            cdbks_UE = np.transpose(np.reshape(antenna_UE.codebook, [N_UE**2, -1, N_RF_UE]), [1, 0, 2])
            cdbks_AP = np.transpose(np.reshape(antenna_AP.codebook, [N_AP**2, -1, N_RF_AP]), [1, 0, 2])

            cdbks_RIS_phases = (2*np.pi*np.random.rand(N_RIS*N_RIS,int(N_M_AP*N_M_AP/N_RF_AP))-np.pi)
            cdbks_RIS_coeff = hardware.RISmodel(cdbks_RIS_phases, ideal=True)
            cdbks_RIS = np.zeros((int(N_M_AP*N_M_AP/N_RF_AP), N_RIS*N_RIS, N_RIS*N_RIS), dtype="complex128")
            for idx, cdbk_RIS in enumerate(cdbks_RIS_coeff.T):
                cdbks_RIS[idx] = np.dot(np.conj(MgammaR.T), np.dot(np.diag(cdbk_RIS), MgammaR))

            # Whitening matrices
            LLinv = [np.linalg.inv(np.linalg.cholesky(np.dot(np.conj(cdbk.T), cdbk))) for cdbk in cdbks_UE]

            # Measurement matrix
            if scenario == 2:
                FE_AP_conv = []
                FE_RIS_conv = []
                for idx, cdbk_AP in enumerate(cdbks_AP):
                    FE_AP = np.dot(np.dot(np.conj(MgammaB.T), cdbk_AP), Pilot)
                    FE_AP_conv.append(np.zeros((N_AP**2, D, P_len), dtype="complex128"))
                    FE_RIS = np.dot(np.dot(np.dot(cdbks_RIS[idx], H_BR), np.dot(np.conj(MgammaB.T), cdbk_AP)), Pilot)
                    # FE_RIS = np.dot(np.dot(np.dot(np.diag(cdbks_RIS[:,idx]),H_BR),cdbk_AP)/ \
                    #                             np.linalg.norm(np.dot(np.dot(np.diag(cdbks_RIS[:,idx]),H_BR),cdbk_AP),ord=2,axis=0), Pilot)                
                    FE_RIS_conv.append(np.zeros((N_RIS**2, D, P_len), dtype="complex128"))              
                    for k in range(D):
                        FE_AP_conv[-1][:, k, :] = FE_AP[:, D-k:P_len+D-k]
                        FE_RIS_conv[-1][:, k, :] = FE_RIS[:, D-k:P_len+D-k]
                FE_AP_conv = np.concatenate(FE_AP_conv, axis=2)
                FE_AP_conv = FE_AP_conv.transpose([2, 0, 1])      # (P_len*N_M_TX/N_RF_TX)  x  N_TX x D
                FE_AP_conv_U = np.eye(FE_AP_conv.shape[0])
                FE_AP_conv_x_U = np.tensordot(FE_AP_conv_U.conj(), FE_AP_conv, axes=(0, 0))
                FE_RIS_conv = np.concatenate(FE_RIS_conv, axis=2)
                FE_RIS_conv = FE_RIS_conv.transpose([2, 0, 1])      # (P_len*N_M_TX/N_RF_TX)  x  N_TX x D
                FE_RIS_conv_U = np.eye(FE_RIS_conv.shape[0])
                FE_RIS_conv_x_U = np.tensordot(FE_RIS_conv_U.conj(), FE_RIS_conv, axes=(0, 0))
                A_AP = FE_AP_conv_x_U.reshape((-1, N_AP, N_AP, D))
                A_RIS = FE_RIS_conv_x_U.reshape((-1, N_RIS, N_RIS, D))
            else:
                FE_conv = []
                for idx, cdbk_AP in enumerate(cdbks_AP):
                    if scenario == 0:
                        # FE = np.dot(cdbk_AP, Pilot)
                        FE = np.dot(np.dot(np.conj(MgammaB.T), cdbk_AP), Pilot)
                        FE_conv.append(np.zeros((N_AP**2, D, P_len), dtype="complex128"))
                    elif scenario == 1:
                        FE = np.dot(np.dot(np.dot(cdbks_RIS[idx], H_BR), np.dot(np.conj(MgammaB.T), cdbk_AP)), Pilot)
                        FE_conv.append(np.zeros((N_RIS**2, D, P_len), dtype="complex128"))            
                    for k in range(D):
                        FE_conv[-1][:, k, :] = FE[:, D-k:P_len+D-k]
                FE_conv = np.concatenate(FE_conv, axis=2)
                FE_conv = FE_conv.transpose([2, 0, 1])      # (P_len*N_M_TX/N_RF_TX)  x  N_TX x D
                FE_conv_U = np.eye(FE_conv.shape[0])
                FE_conv_x_U = np.tensordot(FE_conv_U.conj(), FE_conv, axes=(0, 0))
                if scenario == 0:
                    A = FE_conv_x_U.reshape((-1, N_AP, N_AP, D))
                elif scenario == 1:
                    A = FE_conv_x_U.reshape((-1, N_RIS, N_RIS, D))

            # Define decomposition algorithm        
            if scenario == 2:
                stop = MOMP.stop.General(maxIter=N_est)     # Stop when reached the desired number of estimated paths
                proj_AP_init = MOMP.proj.MOMP_greedy_proj(A_AP[:int(A_AP.shape[0]/2),:,:,:], X_AP, X_AP_lr, normallized=False)
                proj_AP = MOMP.proj.MOMP_proj(A_AP[:int(A_AP.shape[0]/2),:,:,:], X_AP, initial=proj_AP_init, normallized=False)     
                proj_RIS_init = MOMP.proj.MOMP_greedy_proj(A_RIS[:int(A_AP.shape[0]/2),:,:,:], X_RIS, X_RIS_lr, normallized=False)
                proj_RIS = MOMP.proj.MOMP_proj(A_RIS[:int(A_AP.shape[0]/2),:,:,:], X_RIS, initial=proj_RIS_init, normallized=False)
                proj_AP_init2 = MOMP.proj.MOMP_greedy_proj(A_AP[int(A_AP.shape[0]/2):,:,:,:], X_AP, X_AP_lr, normallized=False)
                proj_AP2 = MOMP.proj.MOMP_proj(A_AP[int(A_AP.shape[0]/2):,:,:,:], X_AP, initial=proj_AP_init2, normallized=False)     
                proj_RIS_init2 = MOMP.proj.MOMP_greedy_proj(A_RIS[int(A_AP.shape[0]/2):,:,:,:], X_RIS, X_RIS_lr, normallized=False)
                proj_RIS2 = MOMP.proj.MOMP_proj(A_RIS[int(A_RIS.shape[0]/2):,:,:,:], X_RIS, initial=proj_RIS_init2, normallized=False)        
                alg = MOMP.mp.OMPmultiproj([proj_AP, proj_RIS], stop)
                alg2 = MOMP.mp.OMPmultiproj([proj_AP2, proj_RIS2], stop)
            else:
                stop = MOMP.stop.General(maxIter=N_est)     # Stop when reached the desired number of estimated paths
                if method == "OMP":
                    X_kron = A.copy()
                    for x in X:
                        X_kron = np.tensordot(X_kron, x, axes = (1, 0))
                    X_kron = np.reshape(X_kron, [X_kron.shape[0], -1])
                    proj = MOMP.proj.OMP_proj(X_kron)
                else:
                    proj_init = MOMP.proj.MOMP_greedy_proj(A, X, X_lr, normallized=False)
                    proj = MOMP.proj.MOMP_proj(A, X, initial=proj_init, normallized=False)
                alg = MOMP.mp.OMP(proj, stop)
            
            # Overall received signal
            MM = []
            if scenario == 0:
                channel.build(rays_BM)
            else:
                channel.build_RIS_2(rays_BM, rays_RM, Rays_BR[0], scenario)                
            for cdbk_UE, Linv in zip(cdbks_UE, LLinv):
                MMM = []
                antenna_UE.set_codebook(cdbk_UE)
                channel.set_corr(np.dot(np.conj(antenna_UE.codebook.T), antenna_UE.codebook))
                for idx, cdbk_AP in enumerate(cdbks_AP):
                    antenna_AP.set_codebook(cdbk_AP)
                    if scenario == 0:
                        MMM.append(np.dot(Linv, channel.measure(MgammaB, MgammaM)))
                    else:
                        MMM.append(np.dot(Linv, channel.measure_RIS_2(MgammaB, MgammaM, rays_RM , Rays_BR[0], cdbks_RIS[idx], scenario)))
                MM.append(MMM)
            M = np.concatenate([np.concatenate(MMM, axis=1) for MMM in MM], axis=0)
            
            # Channel estimation, I contains the indices in dictionaries
            if scenario == 2:
                M_U_U = np.tensordot(M, FE_AP_conv_U.conj(), axes=(1, 0))
                I_p, I, alpha = alg(M_U_U[:,:int(M_U_U.shape[1]/2)].T)
                I_p2, I2, alpha2 = alg2(M_U_U[:,int(M_U_U.shape[1]/2):].T)
            else:
                M_U_U = np.tensordot(M, FE_conv_U.conj(), axes=(1, 0))
                I, alpha = alg(M_U_U.T)
            toc = time()-tic
            print(toc)
            if method == "OMP":
                I = [list(np.unravel_index(ii, [x.shape[1] for x in X])) for ii in I]
                
            if scenario == 2:
                Alpha = []
                Power = []
                DoA = []
                DoD = []
                ToF = []
               
                AP_rays = np.where(I_p==0)
                RIS_rays = np.where(I_p2==1)
                # Alpha and power information
                Alpha.append(alpha[AP_rays[0][0]])
                Alpha.append(alpha2[RIS_rays[0][0]])
                Power.append(20*np.log10(np.linalg.norm(alpha[AP_rays[0][0]])))
                Power.append(20*np.log10(np.linalg.norm(alpha2[RIS_rays[0][0]])))
                Alpha = np.array(Alpha)
                Power = np.array(Power)
                # Delay information
                ToF = np.array((delays[I[AP_rays[0][0]][2]],delays[I2[RIS_rays[0][0]][2]]))/B
                ToF[1] = ToF[1] - Rays_BR[0].ray_info[0,1]
                TDoF = ToF - ToF[0]
                DDoF = TDoF*c
                # AP DoD
                xod_AP = angles_AP_x[I[AP_rays[0][0]][0]]/np.pi
                yod_AP = angles_AP_y[I[AP_rays[0][0]][1]]/np.pi
                zod_AP = xod_AP**2 + yod_AP**2
                if zod_AP > 1:
                    xod_AP, yod_AP = xod_AP/np.sqrt(zod_AP), yod_AP/np.sqrt(zod_AP)
                    zod_AP = 0
                else:
                    zod_AP = np.sqrt(1-zod_AP)
                dod_AP = np.array([xod_AP, yod_AP, -zod_AP])
                dod_AP = antenna_AP.uncertainty.apply(dod_AP)
                # RIS DoD
                xod_RIS = angles_RIS_x[I2[RIS_rays[0][0]][0]]/np.pi
                yod_RIS = angles_RIS_y[I2[RIS_rays[0][0]][1]]/np.pi       
                zod_RIS = xod_RIS**2 + yod_RIS**2
                if zod_RIS > 1:
                    xod_RIS, yod_RIS = xod_RIS/np.sqrt(zod_RIS), yod_RIS/np.sqrt(zod_RIS)
                    zod_RIS = 0
                else:
                    zod_RIS = np.sqrt(1-zod_RIS)
                dod_RIS = np.array([xod_RIS, yod_RIS, -zod_RIS])
                dod_RIS = antenna_RIS.uncertainty.apply(dod_RIS)
                # Create direction matrix
                DoD.append(dod_AP)
                DoD.append(dod_RIS)
                DoD = np.array(DoD)
                # DoA                
                bigW_orig = []
                bigW = []
                for cdbk_UE, Linv in zip(cdbks_UE, LLinv):
                    bigW_orig.append(np.dot(Linv,cdbk_UE.conj().T))
                    bigW.append(np.dot(Linv, np.dot(cdbk_UE.conj().T, MgammaM)))                
                bigW_orig = np.vstack(bigW)
                bigW = np.vstack(bigW)

                gain = []
                idx = []
                for ii, a in enumerate(Alpha):
                    BetaW =  np.dot(a[:,np.newaxis].conj().T,np.linalg.pinv(bigW.conj().T))
                    idx.append(np.argmax(np.abs(np.dot(BetaW,A_UE))))
                    theta = np.unravel_index(idx[ii],(A_UE_x.shape[1],A_UE_y.shape[1]))
                    xoa = angles_UE_x[theta[0]]/np.pi
                    yoa = angles_UE_y[theta[1]]/np.pi
                    zoa = xoa**2 + yoa**2
                    if zoa > 1:
                        xoa, yoa = xoa/np.sqrt(zoa), yoa/np.sqrt(zoa)
                        zoa = 0
                    else:
                        zoa = np.sqrt(1-zoa)
                    doa = np.array([xoa, yoa, zoa])
                    DoA.append(doa)
                    matched_filter = np.sqrt(p_t)*np.dot(A_UE[:,idx[ii]][:,np.newaxis].conj().T,bigW)
                    gain.append(np.dot(matched_filter,a)/np.linalg.norm(matched_filter)**2)
                gain = np.array(gain)
                DoA = np.array(DoA)
                DoA2 = DoA
                DoA = antenna_UE.uncertainty.apply(DoA)

                UE_pos_est_tmps = []
                if loc_method == 0:
                    UE_pos_est_tmps.append(localization.localization_2LoS(np.array(AP_pos[0]),np.array(RIS_pos[0]),-DoA,ToF,c))
                elif loc_method == 1:
                    UE_pos_est_tmps.append(localization.localization_2LoS(np.array(AP_pos[0]),np.array(RIS_pos[0]),DoD,ToF,c))

                UE_pos_est_tmps = np.array(UE_pos_est_tmps)
                errors = np.linalg.norm(np.abs(UE_pos_est_tmps-ue_pos), axis=1)
                UE_pos_est_tmp = UE_pos_est_tmps[np.argmin(np.array(errors)),:]
                                
            else:
                Alpha = []
                Power = []
                DoA = []
                DoD = []
                ToF = []
                for a, iii in zip(alpha, I):
                    Alpha.append(a)
                    Power.append(20*np.log10(np.linalg.norm(a)))
                    ii_component = 0
    
                    if scenario == 0:
                        xod = angles_AP_x[iii[ii_component]]/np.pi
                        yod = angles_AP_y[iii[ii_component+1]]/np.pi
                    elif scenario == 1:
                        xod = angles_RIS_x[iii[ii_component]]/np.pi
                        yod = angles_RIS_y[iii[ii_component+1]]/np.pi
                    zod = xod**2 + yod**2
                    if zod > 1:
                        xod, yod = xod/np.sqrt(zod), yod/np.sqrt(zod)
                        zod = 0
                    else:
                        zod = np.sqrt(1-zod)
                    if scenario == 0:
                        dod = np.array([xod, yod, -zod])
                    else:
                        dod = np.array([xod, yod, -zod])
                    DoD.append(dod)
                    ii_component += 2
                    tof = delays[iii[ii_component]]
                    ToF.append(tof)
                Alpha = np.array(Alpha)
                Power = np.array(Power)
                DoD = np.array(DoD)        
                DoD2 = DoD
                if scenario == 0:
                    DoD = antenna_AP.uncertainty.apply(DoD)
                elif scenario == 1:
                    DoD = antenna_RIS.uncertainty.apply(DoD)    
                ToF = np.array(ToF)/B
                TDoF = ToF - ToF[0]
                DDoF = TDoF*c

                bigW_orig = []
                bigW = []
                for cdbk_UE, Linv in zip(cdbks_UE, LLinv):
                    bigW_orig.append(np.dot(Linv,cdbk_UE.conj().T))
                    bigW.append(np.dot(Linv, np.dot(cdbk_UE.conj().T, MgammaM)))                
                bigW_orig = np.vstack(bigW_orig)
                bigW = np.vstack(bigW)

                DoA = []                   
                gain = []
                idx = []
                for ii, a in enumerate(Alpha):
                    BetaW =  np.dot(a[:,np.newaxis].conj().T,np.linalg.pinv(bigW.conj().T))
                    idx.append(np.argmax(np.abs(np.dot(BetaW,A_UE))))
                    theta = np.unravel_index(idx[ii],(A_UE_x.shape[1],A_UE_y.shape[1]))
                    xoa = angles_UE_x[theta[0]]/np.pi
                    yoa = angles_UE_y[theta[1]]/np.pi                 
                    zoa = xoa**2 + yoa**2
                    if zoa > 1:
                        xoa, yoa = xoa/np.sqrt(zoa), yoa/np.sqrt(zoa)
                        zoa = 0
                    else:
                        zoa = np.sqrt(1-zoa)
                    doa = np.array([xoa, yoa, zoa])
                    DoA.append(doa)
                    matched_filter = np.sqrt(p_t)*np.dot(A_UE[:,idx[ii]][:,np.newaxis].conj().T,bigW)
                    gain.append(np.dot(matched_filter,a)/np.linalg.norm(matched_filter)**2)
                gain = np.array(gain)
                DoA = np.array(DoA)
                DoA2 = DoA
                DoA = antenna_UE.uncertainty.apply(DoA)

                if scenario == 0:                 
   
                    UE_pos_est_tmps = []
                    if loc_method == 0:
                        UE_pos_est_tmps.append(localization.localization_single_ap(AP_pos[0], -DoA, DDoF))
                    elif loc_method == 1:
                        UE_pos_est_tmps.append(localization.localization_single_ap(AP_pos[0], DoD, DDoF))
                    UE_pos_est_tmps = np.array(UE_pos_est_tmps)
                    if UE_pos_est_tmps[0] is None:
                        UE_pos_est_tmps[0] = np.array((1000,1000,100))[np.newaxis,:]
                    errors = np.linalg.norm(np.abs(UE_pos_est_tmps-ue_pos), axis=1)
                    UE_pos_est_tmp = UE_pos_est_tmps[np.argmin(np.array(errors)),:]
                    
                elif scenario == 1:

                    UE_pos_est_tmps = []
                    if loc_method == 0:
                        UE_pos_est_tmps.append(localization.localization_single_ap(RIS_pos[0], -DoA, DDoF))
                    elif loc_method == 1:
                        UE_pos_est_tmps.append(localization.localization_single_ap(RIS_pos[0], DoD, DDoF))                     
                    UE_pos_est_tmps = np.array(UE_pos_est_tmps)
                    if UE_pos_est_tmps[0] is None:
                        UE_pos_est_tmps[0] = np.array((1000,1000,100))[np.newaxis,:]
                    errors = np.linalg.norm(np.abs(UE_pos_est_tmps-ue_pos), axis=1)
                    UE_pos_est_tmp = UE_pos_est_tmps[np.argmin(np.array(errors)),:]
            
            x_val = UE_pos_est_tmp[0]
            y_val = UE_pos_est_tmp[1]
            z_val = UE_pos_est_tmp[2]
            
            count = count + 1
            if count == count_th:
                print(UE_pos_est_tmp)
                UE_pos_est_tmp = np.array((1000,1000,1000))
                break        

        UE_pos_est.append(UE_pos_est_tmp)
        estimation.append({
            "Alpha_r": np.real(Alpha).tolist(), "Alpha_i": np.imag(Alpha).tolist(), "Power": Power.tolist(),
            "DoA": DoA.tolist(), "DoD": DoD.tolist(),
            "DDoF": DDoF.tolist(), "CTime": toc,
            "MgammaM_r": np.real(MgammaM).tolist(), "MgammaM_i": np.imag(MgammaM).tolist(),
            "bigW_orig_r": np.real(bigW_orig).tolist(), "bigW_orig_i": np.imag(bigW_orig).tolist(),
            "bigW_r": np.real(bigW).tolist(), "bigW_i": np.imag(bigW).tolist()})

        print(ue_pos)
        print(UE_pos_est[-1])
        print(np.linalg.norm(np.abs(UE_pos_est[-1]-ue_pos)))
        print("\n")