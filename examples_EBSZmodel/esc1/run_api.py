from rsqsim_api.containers.fault import RsqSimMultiFault

test = RsqSimMultiFault.read_fault_file_keith("EBSZ_1000.flt")
f0 = test.faults[0]