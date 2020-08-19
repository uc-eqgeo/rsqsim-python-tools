from rsqsim_api.containers.fault import RsqSimMultiFault

test = RsqSimMultiFault.read_fault_file_bruce("zfault_Deepen.in", "znames_Deepen.in")
f0 = test.faults[0]