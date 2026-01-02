import noise_covariance as nc

Rw_dict, info = nc.compute_noise_covariances_from_folder(
    N=100,
    base_path=r"G:\GITFINAL\rf_tdoa\DATA\RAW_BIN",
    downsample_factor=50,
    save_dir=r"CODE\MATCHED_FILTER\NOISE_COVARIANCE",
    verbose=True,
)

print(info)
# Rw_dict["rx0"], Rw_dict["rx1"], Rw_dict["rx2"]
