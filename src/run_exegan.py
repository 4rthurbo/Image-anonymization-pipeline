import subprocess
import os

# exe-gan is called in a subprocess
def run_exegan():

    print("############################################ \n start EXE-GAN Inpainting...")

    eval_dir = os.path.abspath("images/temp/inpaintings")
    os.makedirs(eval_dir, exist_ok=True)

    subprocess.run([
        "python", "EXE-GAN/test.py",
        "--path", "images/temp/crops",
        "--mask_root", "images/temp/masks",
        "--mask_file_root", "images/temp",
        "--mask_type", "mask_list.txt",
        "--psp_checkpoint_path", "EXE-GAN/pre-train/psp_ffhq_encode.pt",
        "--ckpt", "EXE-GAN/checkpoint/EXE_GAN_model.pt",
        "--size", "256",
        "--batch", "1",
        "--eval_dir", eval_dir
    ])

