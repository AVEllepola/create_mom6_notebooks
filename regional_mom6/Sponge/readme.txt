creating damp state file...

1) download the 3d domain data for the whole model run
2) use code from the regional mom6 to process the downloaded data to create an nc file covering the whole time period of the model run with regridded data for the model domain. Functionality of regional mom6 from the step where IC_unprocessed is handled is what we need. 
	   For this step regional mom6 is not needed at all.  `hgrid.nc` and `voord.nc` along with downloaded data are all we need. A simple script to regrid the data to the same structure as `hgrid` and `vcoord` is sufficient. 

3) step 2 creates the damp_state file.
4) next create the damp rate file. this could be done using the script provided by https://github.com/NOAA-GFDL/CEFI-regional-MOM6/tree/main/tools/sponge. write_damping_tgb.py creates a tapering sponge where as the write_damping_data.py creates a non tapering sponge.
5) once the two files are made give their path to the sponge related override section in the MOM_override. 

add the below to the override once the damp state file and the damp rate file are created.

! === Sponge ===

#override SPONGE = True

#override SPONGE_DAMPING_FILE = "damping_tgb_tuv.nc" 
#override SPONGE_STATE_FILE = "state_file_tracers.nc"
#override SPONGE_PTEMP_VAR = "temp"
#override SPONGE_SALT_VAR = "salt"
#override SPONGE_IDAMP_VAR = "Idamp"

#override SPONGE_UV = True

#override SPONGE_UV_STATE_FILE = "state_file_uv.nc"
#override SPONGE_U_VAR = "u"
#override SPONGE_V_VAR = "v"
#override SPONGE_IDAMP_U_var = "Idamp_u"
#override SPONGE_IDAMP_V_var = "Idamp_v"

#override INTERPOLATE_SPONGE_TIME_SPACE = True
#override SPONGE_DATA_ONGRID = True

--------------------------------------------------------------------------


ERROR records

FATAL from PE 0: NetCDF: Invalid dimension ID or name: get_unlimited_dimension_name: file:INPUT/ic_processed_291216_020118_tracers.nc . time dimension has to be unlimited in state files.

FATAL from PE    34: file/field INPUT/rec_out/ic_processed_291216_020118_tracers.nc/temp couldnt recognize axis atts in time_interp_external - axis attribute needs to be added. reffering to the gfdl code file creatinon,


FATAL from PE     1: time_interp_ext, file/field INPUT/rec_out/ic_processed_291216_020118_uv.nc/u x dim doesnt match model -- the uv needs to be on hpoints.