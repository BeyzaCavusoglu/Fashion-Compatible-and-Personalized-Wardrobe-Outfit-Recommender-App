from flask import Flask, request, render_template, send_file
from flask import redirect, url_for

import os
import subprocess

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.template_folder = os.path.abspath('templates')
app.config['UPLOAD_FOLDER'] = os.path.join(app.template_folder, 'uploads')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')



@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')

    # Perform your login logic here
    # Check if the username and password are valid

    # Example validation: If username is "admin" and password is "password", redirect to home.html
    if username == 'admin' and password == 'password':
        return redirect(url_for('home'))

    # If login fails, you can display an error message or redirect back to the login page
    return 'Invalid username or password'

@app.route('/home', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/calendar', methods=['GET'])
def calendar():
    return render_template('calendar.html')

@app.route('/closet', methods=['GET'])
def closet():
    # Get a list of all the uploaded image files in the UPLOAD_FOLDER directory
    image_files = os.listdir(app.config['UPLOAD_FOLDER'])

    # Create a list to store the words from the list_obj.txt file
    words_list = []

    # Iterate over the image files
    for filename in image_files:
        if not filename.endswith('_clothes_detected.jpg'):
            # Construct the path to the list_obj.txt file
            list_obj_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{filename}_list_obj.txt")

            # Check if the list_obj.txt file exists
            if os.path.isfile(list_obj_path):
                # Open the list_obj.txt file and read the words
                with open(list_obj_path, 'r') as file:
                    for line in file:
                        words_list.extend(line.strip().split())

    # Render the 'my_outfits.html' template and pass the image_files list to it
    return render_template('closet.html', image_files=image_files, words_list=words_list)


@app.route('/cloth_recognition', methods=['GET'])
def cloth_recognition():
    return render_template('cloth_recognition.html')

@app.route('/my_outfits', methods=['GET'])
def my_outfits():
    # Get a list of all the uploaded image files in the UPLOAD_FOLDER directory
    image_files = os.listdir(app.config['UPLOAD_FOLDER'])

    # Create a list to store the words from the list_obj.txt file
    words_list = []

    # Iterate over the image files
    for filename in image_files:
        if not filename.endswith('_clothes_detected.jpg'):
            # Construct the path to the list_obj.txt file
            list_obj_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{filename}_list_obj.txt")

            # Check if the list_obj.txt file exists
            if os.path.isfile(list_obj_path):
                # Open the list_obj.txt file and read the words
                with open(list_obj_path, 'r') as file:
                    for line in file:
                        words_list.extend(line.strip().split())

    # Render the 'my_outfits.html' template and pass the image_files list to it
    return render_template('my_outfits.html', image_files=image_files, words_list=words_list)


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' in request.files:
        image = request.files['image']
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(image_path)
        subprocess.run(['python', 'main.py', image.filename])  # Run main.py using subprocess

        list_obj_path = f"./templates/uploads/{image.filename}_list_obj.txt"
        words_list = []

        with open(list_obj_path, 'r') as file:
            for line in file:
                words_list.extend(line.strip().split())


        return render_template('result.html', image_path=image.filename, result_image_path=f'{image.filename}_clothes_detected.jpg', words_list=words_list)

    else:
        return 'No image found in the request.'


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))


if __name__ == '__main__':
    app.run(debug=True)
