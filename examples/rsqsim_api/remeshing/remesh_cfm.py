from rsqsim_api.io.tsurf import tsurf
import os
import glob

in_directory = "../../../data/cfm/cfm_0_3_tsurf"
out_directory = "../../../data/cfm/cfm_0_3_stl"

cubit_commands = ("""cubit.cmd('import stl "{}" feature_angle 135.00 merge ')\n"""
                  "cubit.cmd('delete mesh surface 1 propagate')\n"
                  "cubit.cmd('surface 1  scheme trimesh geometry approximation angle 15 minimum size 800 ')\n"
                  "cubit.cmd('Trimesher surface gradation 1.3')\n"
                  "cubit.cmd('Trimesher geometry sizing on')\n"
                  "cubit.cmd('surface 1 size 1000')\n"
                  "cubit.cmd('mesh surface 1 ')\n"
                  """cubit.cmd('export stl "{}" surface 1 mesh  overwrite ')\n"""
                  "cubit.cmd('delete Volume 1')\n")

def journal_instructions(in_file, out_file, jou_file):
    filled_commands = cubit_commands.format(in_file, out_file)
    with open(jou_file, "w") as journal:
        journal.write("#!python\n")
        for command in filled_commands:
            journal.write(command)
        journal.write("exit()\n")




# tsurf_files = glob.glob(os.path.join(in_directory, "*.ts"))
# if not os.path.exists(out_directory):
#     os.mkdir(out_directory)
# for tsurf_file in tsurf_files:
#     ts = tsurf(tsurf_file)
#     basename = os.path.basename(tsurf_file)
#     stl_name = basename.split(".ts")[0] + ".stl"
#     stl_out = os.path.join(out_directory, stl_name)
#     try:
#         ts.mesh.write(stl_out, file_format="stl")
#     except:
#         print("Trouble writing: {}".format(stl_name))



