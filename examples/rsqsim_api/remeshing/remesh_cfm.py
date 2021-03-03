from rsqsim_api.io.tsurf import tsurf
import os
import glob

# in_directory = "../../../data/cfm/cfm_0_3_tsurf"
in_directory = "../../../data/cfm/cfm_0_3_stl"
out_directory = "../../../data/cfm/cfm_0_3_remeshed_stl"


def journal_command(in_file, out_file, number):
    cubit_commands = "".join(["""cubit.cmd('import stl "{}" feature_angle 135.00 merge ')\n""".format(in_file),
                              "cubit.cmd('delete mesh surface {:d} propagate')\n".format(number),
                              "cubit.cmd('surface {:d}  scheme trimesh geometry approximation angle 15 minimum size 800 ')\n".format(number),
                              "cubit.cmd('Trimesher surface gradation 1.3')\n",
                              "cubit.cmd('Trimesher geometry sizing on')\n",
                              "cubit.cmd('surface {:d} size 1000')\n".format(number),
                              "cubit.cmd('mesh surface {:d} ')\n".format(number),
                              """cubit.cmd('export stl "{}" surface {:d} mesh  overwrite ')\n""".format(out_file, number),
                              "cubit.cmd('delete Surface {:d}')\n".format(number)])
    return cubit_commands


def journal_instructions(in_file_list, jou_file):
    with open(jou_file, "w") as journal:
        journal.write("#!python\n")
        for i, in_file in enumerate(in_file_list):
            out_name = get_out_file_name(in_file, out_directory)
            journal.write(journal_command(in_file, out_name, i + 1))
        journal.write("exit()\n")


def get_out_file_name(in_file, out_dir: str):
    basename = os.path.basename(in_file)
    return os.path.join(out_dir, basename)




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



if __name__=="__main__":
    in_files = list(glob.glob(os.path.join(in_directory, "*.stl")))
    journal_instructions(in_files, "multi_test.jou")
