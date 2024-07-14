import numpy as np

def localization_single_ap_new(AP_pos, dirs, dist_diff, classi, weights=None):
    """Localize the user using a single ap.
    UE_pos = localization_single_ap(AP_pos, dirs, time_diff, weights = None)
    INPUT:
        AP_pos: 3D position of the access point
        dirs: array of unitary path directions (main path should be LoS)
        time_diff: time difference with the main path time_diff[0] = 0
        classi: list with path classification ("s" for LoS, "v" for wall reflection, "h" for floor/ceiling reflection and "x" for spurious path)
        weights: weights (possitive) to be used for the UE_pos estimation, we
        recommend using the path power as weight
    OUTPUT:
        UE_pos: Estimated 3D position of the user
    """
    # Check if first path is LoS
    if classi[0] != "s":
        return
    # Define weights
    if weights is None:
        weights = np.ones(len(dirs))
    # Dump every variable to a numpy array
    AP_pos, dirs, dist_diff, weights = [np.asarray(a) for a in [AP_pos, dirs, dist_diff, weights]]
    # Compute angle functions relative to the LoS
    cosa_el = np.sqrt(1-np.power(dirs[:, 2], 2))
    cosa_diff_az = np.dot(
        dirs[:, :2]/cosa_el[:, np.newaxis],
        dirs[0, :2]/cosa_el[0])
    sina_el = dirs[:, 2]
    # Estimation equations system
    scalar = 0
    norm2 = 0
    for ii_path in range(1, len(dirs)):
        if classi[ii_path] == "h":
            scalar += weights[ii_path]*(cosa_el[0]-cosa_el[ii_path])*cosa_el[ii_path]*dist_diff[ii_path]
            norm2 += weights[ii_path]*np.power((cosa_el[0]-cosa_el[ii_path]), 2)
        elif classi[ii_path] == "v":
            scalar += weights[ii_path]*(sina_el[0]-sina_el[ii_path])*sina_el[ii_path]*dist_diff[ii_path]
            norm2 += weights[ii_path]*np.power((sina_el[0]-sina_el[ii_path]), 2)
    # Ranging
    if norm2 > 0:
        d = scalar/norm2
        return AP_pos+dirs[0, :]*d
    else:
        return

def localization_single_ap(AP_pos, dirs, dist_diff, weights=None, dirs_D=None, z_symm=False, th_cosaz=0.99, th_sinel=0.03):
    """Localize the user using a single ap.

    UE_pos = localization_single_ap(AP_pos, dirs, time_diff, weights = None)

    INPUT:
        AP_pos: 3D position of the access point
        dirs: array of unitary path directions (main path should be LoS)
        time_diff: time difference with the main path time_diff[0] = 0
        weights: weights (possitive) to be used for the UE_pos estimation, we
        recommend using the path power as weight

    OUTPUT:
        UE_pos: Estimated 3D position of the user

    """
    # Define weights
    if weights is None:
        weights = np.ones(len(dirs))
    # Dump every variable to a numpy array
    AP_pos, dirs, dist_diff, weights = [np.asarray(a) for a in [AP_pos, dirs, dist_diff, weights]]
    # Compute angle functions relative to the LoS
    cosa_el = np.sqrt(1-np.power(dirs[:, 2], 2))
    cosa_diff_az = np.dot(
        dirs[:, :2]/cosa_el[:, np.newaxis],
        dirs[0, :2]/cosa_el[0])
    sina_el = dirs[:, 2]
    # Wall vs Ceiling/Floor reflection classifier
    if dirs_D is None:
        def is_floor_reflection(ii_path):
            return cosa_diff_az[ii_path] > th_cosaz
        def is_wall_reflection(ii_path):
            return True
    else:
        cosd_el = np.sqrt(1-np.power(dirs_D[:, 2], 2))
        cosd_diff_az = np.dot(
            dirs_D[:, :2]/cosd_el[:, np.newaxis],
            dirs_D[0, :2]/cosd_el[0])
        sind_el = dirs_D[:, 2]
        if z_symm:
            def is_floor_reflection(ii_path):
                return (np.abs(np.abs(sina_el[ii_path])-np.abs(sind_el[ii_path])) < th_sinel) &\
                    (cosa_diff_az[ii_path] > th_cosaz) &\
                    (cosd_diff_az[ii_path] > th_cosaz)
            def is_wall_reflection(ii_path):
                return np.abs(np.abs(sina_el[ii_path])-np.abs(sind_el[ii_path])) < th_sinel
        else:
            def is_floor_reflection(ii_path):
                return (np.abs(sina_el[ii_path]-sind_el[ii_path]) < th_sinel) &\
                    (cosa_diff_az[ii_path] > th_cosaz) &\
                    (cosd_diff_az[ii_path] > th_cosaz)
            def is_wall_reflection(ii_path):
                return np.abs(sina_el[ii_path]+sind_el[ii_path]) < th_sinel
    # Estimation equations system
    scalar = 0
    norm2 = 0
    for ii_path in range(1, len(dirs)):
        if is_floor_reflection(ii_path):
            scalar += weights[ii_path]*(cosa_el[0]-cosa_el[ii_path])*cosa_el[ii_path]*dist_diff[ii_path]
            norm2 += weights[ii_path]*np.power((cosa_el[0]-cosa_el[ii_path]), 2)
        elif is_wall_reflection(ii_path):
            scalar += weights[ii_path]*(sina_el[0]-sina_el[ii_path])*sina_el[ii_path]*dist_diff[ii_path]
            norm2 += weights[ii_path]*np.power((sina_el[0]-sina_el[ii_path]), 2)
    # Ranging
    if norm2 > 0:
        d = scalar/norm2
        return AP_pos+dirs[0, :]*d
    else:
        return

def localization_2LoS(AP_pos,RIS_pos,dirs,TDoF,c):    
    t0 = 1e-9
    b = AP_pos[:,np.newaxis]
    r = RIS_pos[:,np.newaxis]
    phi_BM = dirs[0,:][:,np.newaxis]
    phi_RM = dirs[1,:][:,np.newaxis]
    tau_BM = TDoF[0]
    tau_RM = TDoF[1]
    diff = (phi_RM-phi_BM)*c
    inverse = np.dot(np.linalg.inv(np.dot(diff.T,diff)),diff.T)
    t0_est = np.dot(inverse,(b-r+phi_BM*c*(tau_BM-t0)-phi_RM*c*(tau_RM-t0)))
    ue_post_est = r + phi_RM*c*((tau_RM-t0)+t0_est)
    # ue_post_est = b + phi_BM*c*((tau_BM-t0)+t0_est)
    ue_post_est = ue_post_est[:,-1]
    return ue_post_est