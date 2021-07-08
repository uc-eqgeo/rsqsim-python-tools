import os

from rsqsim_api.fault.multifault import RsqSimMultiFault

run_dir = os.path.dirname(__file__)

bruce_faults = RsqSimMultiFault.read_fault_file_bruce(os.path.join(run_dir, "../../../data/bruce_m7/zfault_Deepen.in"),
                                                    os.path.join(run_dir, "../../../data/bruce_m7/znames_Deepen.in"),
                                                    transform_from_utm=True)

pickle_file = os.path.join(run_dir, "../../../data/bruce_m7/bruce_faults.pkl")
bruce_faults.pickle_model(pickle_file)
pickle_faults = RsqSimMultiFault.read_fault_file_bruce(pickle_file,
                                                    os.path.join(run_dir, "../../../data/bruce_m7/znames_Deepen.in"),
                                                    from_pickle=True)
