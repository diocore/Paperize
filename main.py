from multiprocessing import Process
from pathlib import Path
import glob
import os
from typing import List

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# mockupTypes = ["Bedroom", "Kitchen", "Leaning", "Livingroom", "Lobby", "Office"]
mockupTypes = ["Kitchen", "Office"]

ratios = ["1.0"]
# mockupTypes = ["Leaning"]

DEBUG: bool = False

def getImagesFromFolder(path: str):
    images: List[Image] = []
    for mockupType in mockupTypes:
        for f in glob.iglob(path + "/" + mockupType + "/" + "*"):
            images.append(Image.open(f))
    return images


# def paseImageInImage(source: Image, destination: Image) -> Image:
#     # Convert images to NumPy arrays
#     main_image_array = np.array(destination)
#     template_image_array = np.array(Image.new('RGB', (300, 200), (255, 0, 0)))
#
#     # Load the image using Pillow
#     image = destination
#
#     # Convert the image to a NumPy array
#     image_np = np.array(image)
#
#     # Create a mask for the red area (RGB value of (255, 0, 0))
#     red_mask = (image_np[:, :, 0] == 255) & (image_np[:, :, 1] == 0) & (image_np[:, :, 2] == 0)
#
#     # Find the indices of the red area
#     red_indices = np.argwhere(red_mask)
#
#     # Get the bounding box of the red area
#     y_min, x_min = red_indices.min(axis=0)
#     y_max, x_max = red_indices.max(axis=0)
#
#     # Print the bounding box coordinates
#     print(f"Bounding box coordinates: ({x_min}, {y_min}) to ({x_max}, {y_max})")
#
#     size = x_max - x_min + 1, y_max - y_min + 1
#     source = source.resize(size, Image.LANCZOS)
#     destination.paste(source, (x_min, y_min))
#     # 1627 1193
#     return destination

def fixRedPixelValue(image: Image, red: int = 255, fixAll = False) -> Image:
    filename = image.filename
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image_np = np.array(image)

    # # Find all pixels that are [254, 0, 0] and change them to [255, 0, 0]
    # mask = (image_np[:, :, 0] == red) & (image_np[:, :, 1] == 0) & (image_np[:, :, 2] == 0)
    # image_np[mask] = [255, 0, 0]
    changed: bool = False
    if fixAll:
        x = range(190, 255)
        for n in x:
            mask = (image_np[:, :, 0] == n) & (image_np[:, :, 1] == 0) & (image_np[:, :, 2] == 0)
            if np.any(mask):
                image_np[mask] = [255, 0, 0]
                changed = changed | True

    # Convert the NumPy array back to an image
    modified_image = Image.fromarray(image_np)
    # Save or display the modified image
    if changed:
        print("save")
        modified_image.save(filename)

def paseImageInImage(source: Image, destination: Image, red: int = 255, filename:str = "") -> Image:
    filename_dest = destination.filename
    if destination.mode == 'RGBA':
        destination = destination.convert('RGB')
    if source.mode == 'RGBA':
        source = source.convert('RGB')


    main_image = destination
    main_image_np = np.array(main_image)


    # Load the new image that will replace the red area
    new_image = source
    new_image_np = np.array(new_image)

    # Create a mask for the red area
    red_mask = (main_image_np[:, :, 0] == 255) & (main_image_np[:, :, 1] == 0) & (main_image_np[:, :, 2] == 0)
    red_mask = red_mask.astype(np.uint8) * 255

    # Find contours in the red mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the largest contour is the red area
    try:
        red_contour = max(contours, key=cv2.contourArea)
    except Exception as E:
        print(E, filename_dest)
        # print(red)
        # red -= 1
        # filename = destination.filename
        # new_image = fixRedPixelValue(destination, red)
        # new_image.save(filename)
        # destination = Image.open(filename)
        # paseImageInImage(source, destination, red, filename)
        raise Exception



    # # Find the bounding box of the red area
    # x, y, w, h = cv2.boundingRect(red_contour)

    # Find the four corners of the red area using the contour
    epsilon = 0.02 * cv2.arcLength(red_contour, True)
    approx_corners = cv2.approxPolyDP(red_contour, epsilon, True)

    if len(approx_corners) != 4:
        # print(red)
        # red -= 1
        # filename = destination.filename
        # new_image = fixRedPixelValue(destination, red)
        # new_image.save(filename)
        # destination = Image.open(filename)
        # paseImageInImage(source, destination, red, filename)
        # return
        raise ValueError("The red area does not have four corners." , filename_dest)

    if red <= 100:
        raise ValueError("The red area does not have four corners." , filename_dest)

    if "Leaning" in filename_dest:
        ## fix x axis
        approx_corners[0][0][0] -= 4
        approx_corners[1][0][0] -= 4
        approx_corners[2][0][0] += 4
        approx_corners[3][0][0] += 4
        ## fix y axis
        approx_corners[0][0][1] -= 4
        approx_corners[1][0][1] += 4
        approx_corners[2][0][1] += 4
        approx_corners[3][0][1] -= 4

    # Order corners in a consistent order: top-left, top-right, bottom-right, bottom-left
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    approx_corners = approx_corners.reshape(4, 2)
    ordered_corners = order_points(approx_corners)

    # Define the destination points (corners of the new image)
    (h, w) = new_image_np.shape[:2]
    offset:int = 0

    if ("Leaning" in destination.filename):
        offset *= 20
    dst_points = np.array([[offset, 0], [w - offset, 0], [w - offset, h - offset/2], [offset, h - offset/2]], dtype="float32")

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(dst_points, ordered_corners)

    # Apply the perspective transform to the new image
    transformed_new_image = cv2.warpPerspective(new_image_np, M, (main_image_np.shape[1], main_image_np.shape[0]))



    # Create a mask for the transformed new image
    transformed_mask = np.zeros_like(main_image_np, dtype=np.uint8)
    cv2.fillConvexPoly(transformed_mask, ordered_corners.astype(int), (255, 255, 255))

    # Invert the mask to remove the red area
    inverse_mask = cv2.bitwise_not(transformed_mask)

    # Use the mask to remove the red area from the main image
    main_image_np_no_red = cv2.bitwise_and(main_image_np, inverse_mask)

    # Add the transformed new image to the main image
    result_image = cv2.add(main_image_np_no_red, transformed_new_image)

    result_image = cv2.blur(result_image, (3, 3))

    # Convert the result back to a PIL image and save it
    result_image_pil = Image.fromarray(result_image)
    # result_image_pil.filename = source.filename
    return result_image_pil

def placeImageInMockups(imageToPaste, mockups,filename:str, materialType:str, folderPath:str):
    for mockup in mockups:
        try:
            new_image = paseImageInImage(imageToPaste, mockup)
            folder_name = mockup.filename.replace("\\", "/").split('/')[-2]
            name_pasted = mockup.filename.split("/")[len(mockup.filename.split("/")) - 1].split(".")[0]
            name_original = (filename.split("/"))[len(filename.split('/')) -1].split(".")[0]

            name_comp: str = name_original + "_" + name_pasted
            name_comp = name_comp.replace("\\","_").replace("/","_")
            dir_path = os.path.dirname(os.path.realpath(__file__))
            absolute_path : str = dir_path + "\\OUT\\" + name_original + "\\" + materialType  + "\\" + folder_name + "\\"

            path = Path(absolute_path)
            path.mkdir(parents=True, exist_ok=True)

            pathTotal: str = absolute_path + "\\" + name_comp
            print(name_original + "_" + name_pasted)
            if not DEBUG:
                new_image.save(pathTotal + ".png", quality='keep', compression_level=1)
        except Exception as E:
            print(E, mockup.filename)
            raise E


def CreateMockup(ratio:str, material:str, subfolder:str, image:Image, filename:str, folderPath:str):
    images = []
    ratio_str = str(ratio)
    if ratio == 1.0:
        for f in glob.iglob(f'Images/Mockups/{ratio_str}/{material}/{subfolder}'+"/*"):
            images.append(Image.open(f))
        try:
            placeImageInMockups(image, images, filename, material, folderPath)
        except OSError as E:
            print(E, filename)
        except Exception as E:
            print(E, filename)
            raise E
    else:
        for f in glob.iglob(f'Images/Mockups/{ratio_str}/Horizontal/{material}/{subfolder}'+"/*"):
            images.append(Image.open(f))
        try:
            placeImageInMockups(image, images, filename, material, folderPath)
        except OSError as E:
            print(E, filename)
        except Exception as E:
            print(E, filename)
            raise E
        for f in glob.iglob(f'Images/Mockups/{ratio_str}/Vertical/{material}/{subfolder}'+"/*"):
            images.append(Image.open(f))
        try:
            placeImageInMockups(image, images, filename, material, folderPath)
        except OSError as E:
            print(E, filename)
        except Exception as E:
            print(E, filename)
            raise E


def main():
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "/OUT/"
    mockupTypes = ["Kitchen", "Lobby", "Bodroom", "Office", "Livingroom", "Leaning", "Kids", "Teen", "CloseUp"]
    # mockupTypes = ["Leaning"]
    materials = ["Poster", "FramedWoodBlack", "FramedWoodWhite"]
    # materials = ["FramedWoodBlack", "FramedWoodWhite"]

    # for f in glob.iglob(f'Images/Mockups/1.0/*/*/*'):
    #     img = Image.open(f)
    #     fixRedPixelValue(img, fixAll=True)



    # images = []
    # for f in glob.iglob(f'Images/Art/1.0/*'):
    #     images.append(Image.open(f))
    # CreateMockupsMain(dir_path, images, materials, mockupTypes, 1.0)
    #
    images = []
    for f in glob.iglob(f'Images/Art/0.7/*/*'):
        img = Image.open(f)
        images.append(img)

    CreateMockupsMain(dir_path, images, materials, mockupTypes,0.7)

    # images = []
    # for f in glob.iglob(f'Images/Art/0.7/Horizontal/*'):
    #     img = Image.open(f)
    #     # images.append(img)
    # CreateMockupsMain(dir_path, images, materials, mockupTypes,0.7)



    # dir_path = "C:\StableDiffusion\StabilityMatrix\Data\Packages\ComfyUI\output\_Paperize\Mockups"





def CreateMockupsMain(dir_path, images, materials, mockupTypes, ratio):
    for material in materials:
        for image in images:
            filename: str = image.filename.replace("\\", '/')
            folderPath: str = filename.split('/')[-1].replace(".png", "") + '/' + material + '/'

            path = Path(dir_path + "/" + folderPath)
            path.mkdir(parents=True, exist_ok=True)

            print(filename.split('/')[-1] + '/' + material + '/')
            image = Image.open(filename)

            for mockup in mockupTypes:
                # CreateMockup(ratio, material, mockup, image, filename, path)
                Process(target=CreateMockup, args=(ratio, material, mockup,image, filename, path)).start()


if __name__ == '__main__':
    #sys.argv = ["programName.py","--input","test.txt","--output","tmp/test.txt"]
    DEBUG = False
    main()


