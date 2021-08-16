import numpy as np
def array_to_patch(arg_ls: list):
    patch_num, triangle, fault, strike_slip, dip_slip = arg_ls
    triangle3 = triangle.reshape(3, 3)
    patch = RsqSimTriangularPatch(fault, vertices=triangle3, patch_number=patch_num, strike_slip=strike_slip,
                                  dip_slip=dip_slip)
    return patch_num, patch


def array_to_patch_rake(arg_ls: list):
    patch_num, triangle, fault, rake, normalize_slip = arg_ls
    triangle3 = triangle.reshape(3, 3)
    strike_slip = np.cos(np.radians(rake)) * normalize_slip
    dip_slip = np.sin(np.radians(rake)) * normalize_slip
    patch = RsqSimTriangularPatch(fault, vertices=triangle3, patch_number=patch_num, strike_slip=strike_slip,
                                  dip_slip=dip_slip)
    return patch_num, patch