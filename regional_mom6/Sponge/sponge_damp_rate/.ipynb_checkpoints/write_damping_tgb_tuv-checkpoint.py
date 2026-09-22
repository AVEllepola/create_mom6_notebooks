import numpy as np
from os import path
import xarray

def uvt_hgrid(hgrid):
    u = (
        hgrid
        [['x', 'y']]
        .isel(nxp=slice(0, None, 2), nyp=slice(1, None, 2))
        .rename({'y': 'lat', 'x': 'lon', 'nxp': 'xq', 'nyp': 'yh'})
    )

    v = (
        hgrid
        [['x', 'y']]
        .isel(nxp=slice(1, None, 2), nyp=slice(0, None, 2))
        .rename({'y': 'lat', 'x': 'lon', 'nxp': 'xh', 'nyp': 'yq'})
    )

    t = (
        hgrid
        [['x', 'y']]
        .isel(nxp=slice(1, None, 2), nyp=slice(1, None, 2))
        .rename({'y': 'lat', 'x': 'lon', 'nxp': 'xh', 'nyp': 'yh'})
    )

    return u, v, t

def mult(width, total_width=100):
    i = np.arange(1, total_width+1)
    return 1 - np.tanh((2/np.e)*(i-1)/(width-1))

def create_damping(shape, w_width, w_nsponge, e_width, e_nsponge, n_width, n_nsponge, rate):
    # No south band -- south boundary is left undamped.
    assert rate < 1
    mult_west = np.zeros(shape)
    mult_east = np.zeros(shape)
    mult_north = np.zeros(shape)

    # West: boundary at column index 0 -- direct order (max damping at index 0).
    mult_west[:, 0:w_nsponge] = mult(w_width, total_width=w_nsponge)[np.newaxis, 0:w_nsponge]
    # East: boundary at the last column -- flipped order (max damping at the last index).
    mult_east[:, -e_nsponge:] = np.fliplr(mult(e_width, total_width=e_nsponge)[np.newaxis, 0:e_nsponge])
    # North: boundary at the last row -- flipped order.
    mult_north[-n_nsponge:, :] = np.flipud(mult(n_width, total_width=n_nsponge)[0:n_nsponge, np.newaxis])

    combined = np.maximum(mult_west, mult_east)
    combined = np.maximum(combined, mult_north)
    tanh = combined * rate
    return tanh

def write_damping(hgrid, output_dir, w_nsponge, w_width, e_nsponge, e_width, n_nsponge, n_width, rate,
                   fname='damping_tgb.nc'):
    # Writes ONE combined file containing Idamp (T-point), Idamp_u and Idamp_v
    # (U/V-point) all together -- matches how e.g. NEP10's production
    # MOM_input points both SPONGE_DAMPING_FILE and SPONGE_UV_DAMPING_FILE at
    # the same single file.
    target_u, target_v, target_t = uvt_hgrid(hgrid)

    w_dx = hgrid['dx'].isel(nx=0).mean()
    w_width_pts = int(np.round(w_width / 2 / w_dx))

    e_dx = hgrid['dx'].isel(nx=-1).mean()
    e_width_pts = int(np.round(e_width * 2 / e_dx))

    # NOTE: the original NWA12 script averaged n_dy over only part of the north
    # row (nxp=slice(1000, None)) to skip a land-heavy stretch specific to that
    # domain. That index has no meaning for this domain's grid, so it's been
    # replaced here with a plain full-row mean. Worth double-checking this is
    # right for your grid (e.g. if part of your north edge is land/masked).
    n_dy = hgrid['dy'].isel(ny=-1).mean()
    n_width_pts = int(np.round(n_width / n_dy))

    idamp_u = create_damping(
        target_u.lon.shape, w_width_pts, w_nsponge, e_width_pts, e_nsponge, n_width_pts, n_nsponge, rate)
    idamp_v = create_damping(
        target_v.lon.shape, w_width_pts, w_nsponge, e_width_pts, e_nsponge, n_width_pts, n_nsponge, rate)
    idamp_t = create_damping(
        target_t.lon.shape, w_width_pts, w_nsponge, e_width_pts, e_nsponge, n_width_pts, n_nsponge, rate)

    ds = xarray.Dataset(
        data_vars=dict(
            Idamp_u=(['yh', 'xq'], idamp_u),
            Idamp_v=(['yq', 'xh'], idamp_v),
            Idamp=(['yh', 'xh'], idamp_t),
        ),
        coords=dict(
            xh=target_v.xh,
            xq=target_u.xq,
            yh=target_u.yh,
            yq=target_v.yq
        )
    )

    for v in ['Idamp_u', 'Idamp_v', 'Idamp']:
        ds[v].attrs['units'] = 's-1'
        ds[v].attrs['cell_methods'] = 'time: point'

    encodings = {v: {'dtype': np.int32} for v in ['xh', 'xq', 'yh', 'yq']}
    encodings.update({v: {'_FillValue': None} for v in ['Idamp_u', 'Idamp_v', 'Idamp']})

    ds.to_netcdf(
        path.join(output_dir, fname),
        format='NETCDF3_64BIT',
        engine='netcdf4',
        encoding=encodings
    )

if __name__ == '__main__':
    import argparse
    from yaml import safe_load
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = safe_load(file)
    hgrid = xarray.open_dataset(config['filesystem']['ocean_hgrid'])
    write_damping(
        hgrid, './',
        w_nsponge=120, w_width=62e3,
        e_nsponge=40,  e_width=5e3,
        n_nsponge=80,  n_width=20e3,
        rate=1 / (3 * 24 * 3600),
        fname='damping_tgb.nc',
    )
