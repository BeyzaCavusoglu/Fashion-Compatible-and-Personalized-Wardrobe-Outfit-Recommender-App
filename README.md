# Fashion-Compatible-and-Personalized-Wardrobe-Outfit-Recommender-App
Modisto


Modisto is an app to help users to upload the images that they take from their closet and upload them to a website. Users can upload the pictures, then pretrained clothe detection model works on this image to find the type of clothes from 13 categories below:

Categories: ['short_sleeve_top','long_sleeve_top', 'short_sleeve_outwear', 'long_sleeve_outwear',
                  'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short_sleeve_dress',
                  'long_sleeve_dress', 'vest_dress', 'sling_dress']

Users can also filter the images by liking and unliking by pressing heearth-shaped button like instagram. This improves the interaction between user and UI. Also, closet can be filtered by clothe types.

Technologies:
- Website: JS, HTML, CSS
- Website Backend: Flask
- Clothe Detection: Yolo-v3
- Dataset: Deep Fashion 2


Demo:

Login page:
![image](https://github.com/BeyzaCavusoglu/Fashion-Compatible-and-Personalized-Wardrobe-Outfit-Recommender-Modisto/assets/45294641/355314ac-ff16-438a-afcc-0e5273e0607d)

Users can only sign in with saved username/passwords, so there is active password checking. (use admin and password for a free pass! :) )



Home page:
![image](https://github.com/BeyzaCavusoglu/Fashion-Compatible-and-Personalized-Wardrobe-Outfit-Recommender-Modisto/assets/45294641/8c1990a5-abe9-4fec-a6cc-2e719499630f)




Upload image page:
![image](https://github.com/BeyzaCavusoglu/Fashion-Compatible-and-Personalized-Wardrobe-Outfit-Recommender-Modisto/assets/45294641/a7b55d1e-e77d-4a66-a5f4-714f375dfe7a)




The result of cloth detection:
![image](https://github.com/BeyzaCavusoglu/Fashion-Compatible-and-Personalized-Wardrobe-Outfit-Recommender-Modisto/assets/45294641/7536e1af-9276-4ecc-b8b1-b3f6a1b38839)




As it can be seen, we can detect the category of clothe and get the type as string to print. We will use this to filter the closet with clothing types.




Calendar page:
![image](https://github.com/BeyzaCavusoglu/Fashion-Compatible-and-Personalized-Wardrobe-Outfit-Recommender-Modisto/assets/45294641/3d8a17ff-9bf9-41c8-89e6-e5167ccfcf7b)






The biggest challenge was integrating Flask backend and trying to do the image processing on a web browser from a python code. It took a huge amount of time for me to figure out. I have learned a lot about web development, image processing, object detection, using flask etc.

My suggestions to contributors is keeping developing the model, adding more datasets and training on them. Also, the UI can be improved to be more aesthetic, more features like chatbox etc can be added to improve the app.
