"""Smoke test: load HBI sample data and print a quick summary."""
from pathlib import Path

import numpy as np

from rsqsim_api.catalogue.catalogue import RsqSimCatalogue
from rsqsim_api.io.hbi_utils import (
    read_hbi_event_file,
    read_hbi_xyz_file,
    read_hbi_eqslip_file,
)


HERE = Path(__file__).resolve().parent
EVENT = HERE / "event1.dat"
XYZ = HERE / "xyz1.dat"
EQSLIP = HERE / "EQslip1.dat"
STL = HERE / "model_1_2_km_scale.stl"

events = read_hbi_event_file(str(EVENT))
xyz = read_hbi_xyz_file(str(XYZ))
eqslip = read_hbi_eqslip_file(str(EQSLIP), n_events=len(events),
                                n_elements=len(xyz))
print(f"events: {len(events)} rows  ({events.columns.tolist()})")
print(events.head(3))
print(f"xyz: shape={xyz.shape}, x range [{xyz[:,0].min():.0f}, {xyz[:,0].max():.0f}] m")
print(f"     z range [{xyz[:,2].min():.0f}, {xyz[:,2].max():.0f}] m")
print(f"eqslip: shape={eqslip.shape}, nnz={(eqslip != 0).sum()} "
        f"({(eqslip != 0).mean()*100:.1f}% filled)")
print(f"        slip range [{eqslip.min():.3f}, {eqslip.max():.3f}] m")

cat = RsqSimCatalogue.from_hbi_files(
    event_file=str(EVENT),
    xyz_file=str(XYZ),
    eqslip_file=str(EQSLIP),
    stl_file=str(STL),
    rake=90.0,
)
print("\ncatalogue_df head:")
print(cat.catalogue_df.head(3))
print(f"\nlist lengths: event_list={len(cat.event_list)}, "
        f"patch_list={len(cat.patch_list)}, "
        f"patch_slip={len(cat.patch_slip)}, "
        f"patch_time={len(cat.patch_time_list)}")
print(f"unique events in event_list: {len(np.unique(cat.event_list))}")
print(f"fault_model patches: {len(cat.fault_model.patch_dic)}")

# Pull one event back via the standard event API and check it
biggest = int(cat.catalogue_df["mw"].idxmax())
ev = cat.events_by_number(biggest, cat.fault_model)[0]
print(f"\nbiggest event idx={biggest}, mw={cat.catalogue_df.loc[biggest, 'mw']:.3f}, "
        f"n_patches={len(ev.patches)}, mean_slip={np.mean(ev.patch_slip):.3f} m")

all_evs = cat.all_events(cat.fault_model)
for i, ev in enumerate(all_evs):
    slip_vtk = "ev{:02d}_slip.vtk".format(i)
    ev.slip_dist_to_vtk(slip_vtk)

