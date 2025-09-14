import os
import cv2
import argparse
import numpy as np
from src.detect_faces import detect_faces
from src.detect_plates import detect_plates_filtered
from src.anonymize_methods import blur_region, pixelate_region, mask_region

# clear temps
from src.clear_temp import clear_temp
clear_temp()


#---------ArgumentParser---------
parser = argparse.ArgumentParser(description="Face detection")
parser.add_argument("--input_dir", type=str, default='images/input')
parser.add_argument("--face_method",type=str,default="inpaint", choices=["inpaint","blur","pixelation","mask"])
parser.add_argument("--plate_method", type=str, default="blur", choices=["blur", "pixelation", "mask"])
parser.add_argument("--oriented", type=str, default="deactivate", choices=["activate","deactivate"])

args = parser.parse_args()
#---------------------------

#detect faces
checkpoint_dir = "images/temp/checkpoint"
os.makedirs(checkpoint_dir, exist_ok=True)

predictions_faces = detect_faces(args.input_dir)

if args.face_method == "inpaint":
    from src.run_exegan import run_exegan

    # -------- Predict Faces ----------
    all_results = []
    f_masklist = open("images/temp/mask_list.txt", "w")

    #per input image i
    for prediction in predictions_faces:
        image_path = prediction.path
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        boxes = prediction.boxes.xyxy.cpu().numpy()
        img = cv2.imread(image_path)
        img_height, img_width = img.shape[:2]


        boundingboxes = []
        crop_coords = []
        crop_paths = []
        
        #per bounding box j in image i
        for i, box in enumerate(boxes):
            
            #bounding box = mask for GAN-inpainting
            x1, y1, x2, y2 = box
            boundingboxes.append([int(x1),int(y1),int(x2),int(y2)])


            width = x2-x1
            height = y2-y1

            #padding for context:
            pad_x = width * 0.5
            pad_y = height * 0.5

            #make sure the crop is in the image
            crop_x1 = max(0, x1 - pad_x)
            crop_y1 = max(0, y1 - pad_y)
            crop_x2 = min(img_width, x2 + pad_x)
            crop_y2 = min(img_height, y2 + pad_y)
            crop_coords.append([int(crop_x1),int(crop_y1),int(crop_x2),int(crop_y2)])

            #cut out the crop
            crop = img[int(crop_y1):int(crop_y2), int(crop_x1):int(crop_x2)]
            crop_filename = f"{image_name}_crop_{i:02d}.png"
            crop_path = os.path.join("images/temp/crops", crop_filename)

            crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(crop_path, crop)
            crop_paths.append(crop_path)

            #safe mask
            mask = np.zeros((int(crop_y2 - crop_y1), int(crop_x2 - crop_x1)), dtype=np.uint8)

            rel_x1 = int(x1 - crop_x1)
            rel_y1 = int(y1 - crop_y1)
            rel_x2 = int(x2 - crop_x1)
            rel_y2 = int(y2 - crop_y1)

            # color mask white in bb area
            mask[rel_y1:rel_y2, rel_x1:rel_x2] = 255

            # safe everything
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            mask_path = os.path.join("images/temp/masks", crop_filename)
            cv2.imwrite(mask_path, mask)
            f_masklist.write(f"{crop_filename}\n")

        # informationen
        all_results.append({
            "image_name": image_name,
            "image_path": image_path,
            "boundingboxes": boundingboxes,
            "crop_coords": crop_coords,
            "crop_paths": crop_paths
        })
    f_masklist.close()


    #START EXEGAN INPAINTING
    run_exegan()

    #rescale and merge 
    idx = 0
    for result in all_results:
        original = cv2.imread(result["image_path"])
        merged_image = original.copy()

        for j, (x1, y1, x2, y2) in enumerate(result["crop_coords"]):
            inpaint_path = os.path.join("images/temp/inpaintings", f"{str(idx).zfill(6)}_inpaint.png")
            idx += 1

            inpainted = cv2.imread(inpaint_path)
            if inpainted is None:
                print(f"Inpaint missing: {inpaint_path}")
                continue

            inpainted_resized = cv2.resize(inpainted, (x2 - x1, y2 - y1))
            merged_image[y1:y2, x1:x2] = inpainted_resized

        export_path = os.path.join(checkpoint_dir, result["image_name"]+"_cp.png")
        cv2.imwrite(export_path, merged_image)

else:

    # if exe-gan is not the parameter, one of those methods is applied
    #directly on bb area
    for prediction in predictions_faces:
        img_path = prediction.path
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        boxes = prediction.boxes.xyxy.cpu().numpy()
        img = cv2.imread(img_path)

        for (x1, y1, x2, y2) in boxes:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            if args.face_method == "blur":
                img = blur_region(img, x1, y1, x2, y2)
            elif args.face_method == "pixelation":
                img = pixelate_region(img, x1, y1, x2, y2)
            elif args.face_method == "mask":
                img = mask_region(img, x1, y1, x2, y2)

        export_path = os.path.join(checkpoint_dir, img_name + "_cp.png")
        cv2.imwrite(export_path, img)


print("Face-anonymization finished")


# same for license plates:

# first, detection is applied on semi-anonymized images in checkpoint dir
predictions_plates = detect_plates_filtered(checkpoint_dir)

export_dir = "images/export"
oriented_active = args.oriented == "activate"

# direct anonymization methods:
for prediction in predictions_plates:
    img_path = prediction.path
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    boxes = prediction.boxes.xyxy.cpu().numpy()
    img = cv2.imread(img_path)

    for (x1, y1, x2, y2) in boxes:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        if args.plate_method == "blur":
            img = blur_region(img, x1, y1, x2, y2, oriented=oriented_active)
        elif args.plate_method == "pixelation":
            img = pixelate_region(img, x1, y1, x2, y2, oriented=oriented_active)
        elif args.plate_method == "mask":
            img = mask_region(img, x1, y1, x2, y2, oriented=oriented_active)


    export_path = os.path.join(export_dir, img_name + "_anonymized.png")
    cv2.imwrite(export_path, img)

print("License-Plate anonymization finished")

