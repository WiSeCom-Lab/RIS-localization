# RIS-aided joint channel estimation and localization at mmWave
Code to run to estimate the channels and localize the users for the paper "RIS-aided Joint Channel Estimation and Localization at mmWave under Hardware Impairments: A Dictionary Learning-based Approach".

# Dataset
The `data` folder contains the information related to a ray-tracing dataset for an RIS-aided mmWave system operating in an indoor factory environment. The locations of the BS, RIS and MSs are in `data/AP_pos.txt`, `data/RIS_pos.txt` and `data/UE_pos.txt`, respectively. The information related to paths of the channels are contained in the `data/Info_BM.txt`, `data/Info_BR.txt` and `data/Info_RM.txt` for the BS-MS, BS-RIS and RIS-MS channels, respectively. Each path has the following information: Phase of the channel gain, delay, channel gain, azimuth AoA, elevation AoA, azimuth AoD, and elevation AoD.

# Simulation
The `main.py` code can be run to estimate the channels via MOMP algorithm and localiza the users. All the other `.py` files are complementary functions.

Parameters to set for each case:
- Only BS 8x8: `scenario` = 0, `N_M_AP` = 8, `N_est` = 5
- Only RIS 16x16: `scenario` = 1, `N_RIS` = 16, `N_M_AP` = 16, `N_est` = 5
- Only RIS 32x32: `scenario` = 1, `N_RIS` = 32, `N_M_AP` = 32, `N_est` = 5
- BS and RIS 16x16: `scenario` = 2, `N_RIS` = 16, `N_M_AP` = 16, `N_est` = 20
- BS and RIS 32x132: `scenario` = 2, `N_RIS` = 32, `N_M_AP` = 32, `N_est` = 20

# Citation
If you use the dataset or the code, please cite our paper as:
```
@article{Bayraktar2024,
  title={{RIS}-aided joint channel estimation and localization at {mmWave} under hardware impairments: A dictionary learning-based approach},
  author={Bayraktar, Murat and and González-Prelcic, Nuria and Alexandropoulos, George C. and Chen, Hao},
  journal={IEEE Trans. Wireless Commun.},
  year={2024}
}
```
