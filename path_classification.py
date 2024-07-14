import numpy as np

# DoA and DoD known
def RX_TX(DoA, DoD, DDoF=None, weights=None, th_az=0.02, th_el=0.02):
    # Angular information
    sina_el = DoA[:, 2]
    cosa_el = np.sqrt(1-np.power(sina_el, 2))
    sind_el = DoD[:, 2]
    cosd_el = np.sqrt(1-np.power(sind_el, 2))
    # Initialization
    classi = []
    # Identify if first path is LoS
    first_LoS = np.linalg.norm(DoA+DoD) < th_az + th_el
    if first_LoS:
        classi.append("s")
        aoa_los, aod_los = DoA[0], DoD[0]
        for ii_path in range(1, len(DoA)):
            if np.dot(DoA[ii_path, :2], DoA[0, :2]) > (1-th_az)*cosa_el[ii_path]*cosa_el[0] and np.dot(DoD[ii_path, :2], DoD[0, :2]) > (1-th_az)*cosd_el[ii_path]*cosd_el[0] and np.abs(sina_el[ii_path] - sind_el[ii_path]) < th_el:
                # Either floor or ceiling
                classi.append("h")
            elif np.abs(sina_el[ii_path] + sind_el[ii_path]) < th_el:
                # Wall
                classi.append("v")
            else:
                classi.append("x")
    else:
        for ii_path in range(len(DoA)):
            if np.dot(DoA[ii_path, :2], DoD[ii_path, :2]) < (th_az-1)*cosa_el[ii_path]*cosd_el[ii_path] and np.abs(sina_el[ii_path] - sind_el[ii_path]) < th_el:
                # Ceiling
                classi.append("h")
            elif np.abs(sina_el[ii_path] + sind_el[ii_path]) < th_el:
                # Wall
                classi.append("v")
            else:
                classi.append("x")
    return classi

# DoA and DoD known, DoD[2] may be flipped
def RX_TXF(DoA, DoD, DDoF=None, weights=None, th_az=0.02, th_el=0.02):
    # Angular information
    sina_el = DoA[:, 2]
    cosa_el = np.sqrt(1-np.power(sina_el, 2))
    sind_el = DoD[:, 2]
    cosd_el = np.sqrt(1-np.power(sind_el, 2))
    # Initialization
    classi = []
    # Identify if first path is LoS
    first_LoS = np.linalg.norm(DoA[0]+DoD[0]) < th_az + th_el
    if first_LoS:
        classi.append("s")
        aoa_los, aod_los = DoA[0], DoD[0]
        for ii_path in range(1, len(DoA)):
            if np.dot(DoA[ii_path, :2], DoA[0, :2]) > (1-th_az)*cosa_el[ii_path]*cosa_el[0] and np.dot(DoD[ii_path, :2], DoD[0, :2]) > (1-th_az)*cosd_el[ii_path]*cosd_el[0]:
                # Either floor or ceiling
                classi.append("h")
            elif sina_el[ii_path] < 0 and sina_el[ii_path] > sina_el[0] and np.abs(sina_el[ii_path] + sind_el[ii_path]) < th_el:
                # Wall
                classi.append("v")
            else:
                classi.append("x")
    else:
        for ii_path in range(len(DoA)):
            if np.dot(DoA[ii_path, :2], DoD[ii_path, :2]) < (th_az-1)*cosa_el[ii_path]*cosd_el[ii_path]:
                # Ceiling
                classi.append("h")
            elif sina_el[ii_path] < 0 and np.abs(sina_el[ii_path] + sind_el[ii_path]) < th_el:
                # Wall
                classi.append("v")
            else:
                classi.append("x")
    return classi

# DoA and DoD known, DoD[:2] may be rotated
def RX_TXR(DoA, DoD, DDoF=None, weights=None, th_az=0.02, th_el=0.02):
    # Angular information
    sina_el = DoA[:, 2]
    cosa_el = np.sqrt(1-np.power(sina_el, 2))
    sind_el = DoD[:, 2]
    cosd_el = np.sqrt(1-np.power(sind_el, 2))
    # Initialization
    classi = []
    # Identify if first path is LoS
    first_LoS = np.abs(sina_el[0] - sind_el[0]) < th_el
    if first_LoS:
        classi.append("s")
        aoa_los, aod_los = DoA[0], DoD[0]
        for ii_path in range(1, len(DoA)):
            if np.dot(DoA[ii_path, :2], DoA[0, :2]) > (1-th_az)*cosa_el[ii_path]*cosa_el[0] and np.dot(DoD[ii_path, :2], DoD[0, :2]) > (1-th_az)*cosd_el[ii_path]*cosd_el[0]:
                # Either floor or ceiling
                classi.append("h")
            elif sina_el[ii_path] < 0 and sina_el[ii_path] > sina_el[0] and np.abs(sina_el[ii_path] + sind_el[ii_path]) < th_el:
                # Wall
                classi.append("v")
            else:
                classi.append("x")
    else:
        for ii_path in range(len(DoA)):
            if np.abs(sina_el[ii_path] - sind_el[ii_path]) < th_el:
                # Ceiling
                classi.append("h")
            elif sina_el[ii_path] < 0 and np.abs(sina_el[ii_path] + sind_el[ii_path]) < th_el:
                # Wall
                classi.append("v")
            else:
                classi.append("x")
    return classi