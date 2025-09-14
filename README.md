# Image-anonymization-pipeline

Pipeline for automatic anonymization of faces and license plates in images collected from automotive sensor systems.

# Features:  
- YOLO-based detection for faces and license plates  
- Direct anonymization methods: Gaussian blur, pixelation, mask  
- additionally for faces: GAN-based inpainting (via EXE-GAN)
- oriented anonymization for license plates (via PCA)

# Installation:
Requirements:  
- Python 3.10+  
- requirements.txt  
- detection models are not included and need to be downloaded:  
- [yolov11m-face.pt](https://drive.google.com/file/d/1EMUnDFBMSsiryr-gf5rTC_k-HWMv0ac0/view?usp=sharing)    
- [yolov11m-LP.pt](https://drive.google.com/file/d/1QLMYazNXTw-1sUI-ZH2nkBNK024kcwlx/view?usp=sharing)  
- after downloading, put both models in models/

- for the implementation of EXE-GAN (requirements and pre-trained models), we refere to https://github.com/LonglongaaaGo/EXE-GAN  
- our repository already contains a slightly modified version of EXE-GAN code  
- nevertheless, the pretrained checkpoints have to be prepared  
- the code has been tested on cuda 12.8 and the following torch versions (!!!):  
torch: 2.7.0+cu126  
torchaudio: 2.7.0+cu126  
torchvision: 0.22.0+cu126  

# Usage: 
- put your images into images/input/ and run: 
python boenigk2025bsc_test.py  

Example: python boenigk2025bsc_test.py --face_method blur --plate_method mask --oriented activate

# Parameters:  
--input_dir: folder/to/your/images (default: images/input)
--face_method: inpaint | blur | pixelation | mask (default: inpaint)
--plate_method: blur | pixelation | mask (default: blur)
--oriented: activate | deactivate 

(Its not necessary to meet requirements of EXE-GAN if one of the other methods is chosen)

# Output:  
- Temporary results in images/temp  
- Final results (fully anonymized images) in images/export

# Limitations:  
- only works on images (only .png !!!)  

# License:  
- licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
