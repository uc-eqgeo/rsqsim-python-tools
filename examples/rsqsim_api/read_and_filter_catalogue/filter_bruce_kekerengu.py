from rsqsim_api.containers.catalogue import read_bruce

if not all([a in globals() for a in ("bruce_faults", "catalogue")]):
    bruce_faults, catalogue = read_bruce()