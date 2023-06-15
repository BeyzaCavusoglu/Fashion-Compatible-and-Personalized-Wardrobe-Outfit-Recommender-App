# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# import time
# import tensorflow as tf
#
# from utils_my import Read_Img_2_Tensor, Save_Image, Load_DeepFashion2_Yolov3, Draw_Bounding_Box
#
#
# def Detect_Clothes(img, model_yolov3, eager_execution=True):
#     """Detect clothes in an image using Yolo-v3 model trained on DeepFashion2 dataset"""
#     img = tf.image.resize(img, (416, 416))
#
#     t1 = time.time()
#     if eager_execution==True:
#         boxes, scores, classes, nums = model_yolov3(img)
#         # change eager tensor to numpy array
#         boxes, scores, classes, nums = boxes.numpy(), scores.numpy(), classes.numpy(), nums.numpy()
#     else:
#         boxes, scores, classes, nums = model_yolov3.predict(img)
#     t2 = time.time()
#     print('Yolo-v3 feed forward: {:.2f} sec'.format(t2 - t1))
#
#     class_names = ['short_sleeve_top', 'long_sleeve_top', 'short_sleeve_outwear', 'long_sleeve_outwear',
#                   'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short_sleeve_dress',
#                   'long_sleeve_dress', 'vest_dress', 'sling_dress']
#
#     # Parse tensor
#     list_obj = []
#     for i in range(nums[0]):
#         obj = {'label':class_names[int(classes[0][i])], 'confidence':scores[0][i]}
#         obj['x1'] = boxes[0][i][0]
#         obj['y1'] = boxes[0][i][1]
#         obj['x2'] = boxes[0][i][2]
#         obj['y2'] = boxes[0][i][3]
#         list_obj.append(obj)
#
#     return list_obj
#
# def Detect_Clothes_and_Crop(img_tensor, model, threshold=0.5):
#     list_obj = Detect_Clothes(img_tensor, model)
#
#     img = np.squeeze(img_tensor.numpy())
#     img_width = img.shape[1]
#     img_height = img.shape[0]
#
#     # crop out one cloth
#     for obj in list_obj:
#         if obj['label'] == 'short_sleeve_top' and obj['confidence']>threshold:
#             img_crop = img[int(obj['y1']*img_height):int(obj['y2']*img_height), int(obj['x1']*img_width):int(obj['x2']*img_width), :]
#
#     return img_crop
#
# def process_image(filename):
#     img_path = f'./templates/uploads/{filename}'  # Use the filename received from app.py
#
#     # Read image
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img_tensor = Read_Img_2_Tensor(img_path)
#
#     # Clothes detection and crop the image
#     img_crop = Detect_Clothes_and_Crop(img_tensor, model)
#
#     # Transform the image to gray_scale
#     cloth_img = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)
#
#     # Pretrained classifier parameters
#     PEAK_COUNT_THRESHOLD = 0.02
#     PEAK_VALUE_THRESHOLD = 0.01
#
#     # Horizontal bins
#     horizontal_bin = np.mean(cloth_img, axis=1)
#     horizontal_bin_diff = horizontal_bin[1:] - horizontal_bin[0:-1]
#     peak_count = len(horizontal_bin_diff[horizontal_bin_diff > PEAK_VALUE_THRESHOLD]) / len(horizontal_bin_diff)
#     if peak_count >= PEAK_COUNT_THRESHOLD:
#         print("Class 1 (clothes with stripes)")
#     else:
#         print("Class 0 (clothes without stripes)")
#
#     # Plotting
#     plt.imshow(img)
#     plt.title('Input image')
#     plt.show(block=False)  # Show plot in non-blocking mode
#
#     plt.figure()
#     plt.imshow(img_crop)
#     plt.title('Cloth detection and crop')
#     plt.show(block=False)  # Show plot in non-blocking mode
#
#     # Manually manage the GUI event loop
#     while plt.get_fignums():
#         plt.pause(0.1)
#
#     Save_Image(img_crop, f'./templates/uploads/{filename}_crop.jpg')
#
#
#
# if __name__ == '__main__':
#     filename = 'test5.jpg'  # Replace this with the filename obtained from the webpage submitted by the user
#     img = Read_Img_2_Tensor(f'./templates/uploads/{filename}')
#     model = Load_DeepFashion2_Yolov3()
#     list_obj = Detect_Clothes(img, model)
#     img_with_boxes = Draw_Bounding_Box(img, list_obj)
#
#     cv2.imshow("Clothes detection", cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     cv2.imwrite(f"./templates/uploads/{filename}_clothes_detected.jpg", cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)*255)

import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import tensorflow as tf

from utils_my import Read_Img_2_Tensor, Save_Image, Load_DeepFashion2_Yolov3, Draw_Bounding_Box


def Detect_Clothes(img, model_yolov3, eager_execution=True):
    """Detect clothes in an image using Yolo-v3 model trained on DeepFashion2 dataset"""
    img = tf.image.resize(img, (416, 416))

    t1 = time.time()
    if eager_execution==True:
        boxes, scores, classes, nums = model_yolov3(img)
        # change eager tensor to numpy array
        boxes, scores, classes, nums = boxes.numpy(), scores.numpy(), classes.numpy(), nums.numpy()
    else:
        boxes, scores, classes, nums = model_yolov3.predict(img)
    t2 = time.time()
    print('Yolo-v3 feed forward: {:.2f} sec'.format(t2 - t1))

    class_names = ['short_sleeve_top', 'long_sleeve_top', 'short_sleeve_outwear', 'long_sleeve_outwear',
                  'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short_sleeve_dress',
                  'long_sleeve_dress', 'vest_dress', 'sling_dress']

    # Parse tensor
    list_obj = []
    for i in range(nums[0]):
        obj = {'label':class_names[int(classes[0][i])], 'confidence':scores[0][i]}
        obj['x1'] = boxes[0][i][0]
        obj['y1'] = boxes[0][i][1]
        obj['x2'] = boxes[0][i][2]
        obj['y2'] = boxes[0][i][3]
        list_obj.append(obj)



    return list_obj

def Detect_Clothes_and_Crop(img_tensor, model, threshold=0.5):

    list_obj = Detect_Clothes(img_tensor, model)

    img = np.squeeze(img_tensor.numpy())
    img_width = img.shape[1]
    img_height = img.shape[0]

    # crop out one cloth
    for obj in list_obj:
        if obj['label'] == 'short_sleeve_top' and obj['confidence']>threshold:
            img_crop = img[int(obj['y1']*img_height):int(obj['y2']*img_height), int(obj['x1']*img_width):int(obj['x2']*img_width), :]

    return img_crop

def process_image(filename):
    img_path = f'./templates/uploads/{filename}'  # filename received from app.py

    # Read image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = Read_Img_2_Tensor(img_path)

    # Clothes detection and crop the image
    img_crop = Detect_Clothes_and_Crop(img_tensor, model)

    # Transform the image to gray_scale
    cloth_img = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)

    # Pretrained classifier parameters
    PEAK_COUNT_THRESHOLD = 0.02
    PEAK_VALUE_THRESHOLD = 0.01

    # Horizontal bins
    horizontal_bin = np.mean(cloth_img, axis=1)
    horizontal_bin_diff = horizontal_bin[1:] - horizontal_bin[0:-1]
    peak_count = len(horizontal_bin_diff[horizontal_bin_diff > PEAK_VALUE_THRESHOLD]) / len(horizontal_bin_diff)
    if peak_count >= PEAK_COUNT_THRESHOLD:
        print("Class 1 (clothes with stripes)")
    else:
        print("Class 0 (clothes without stripes)")


    # Save the image with boxes
    img_with_boxes = Draw_Bounding_Box(img, list_obj)
    result_filename = f"./templates/uploads/{filename}_clothes_detected.jpg"
    Save_Image(img_with_boxes, result_filename)


    # After saving the result image
    crop_filename = f"./templates/uploads/{filename}_crop.jpg"  # Change the filename as desired
    Save_Image(img_crop, crop_filename)
    print(img_crop)
    # Save the cropped image
    cv2.imwrite(crop_filename, cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR) * 255)

    return result_filename, filename, crop_filename, list_obj



if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        img = Read_Img_2_Tensor(f'./templates/uploads/{filename}')
        model = Load_DeepFashion2_Yolov3()
        list_obj = Detect_Clothes(img, model)
        img_with_boxes = Draw_Bounding_Box(img, list_obj)

        cv2.imwrite(f"./templates/uploads/{filename}_clothes_detected.jpg", cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)*255)

        # Save list_obj to a text file
        filename = f"./templates/uploads/{filename}_list_obj.txt"
        with open(filename, 'w') as file:
            for obj in list_obj:
                line = f"{obj['label']}\n"
                file.write(line)
