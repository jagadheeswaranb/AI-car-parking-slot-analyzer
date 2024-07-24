
from flask import Flask, render_template, request, session, Response

import cv2

import pickle

import cvzone

import numpy as np

import sqlite3

import re


app = Flask(__name__)
app.secret_key = 'a'

# Create a connection function to be called for each request
def get_db():
    return sqlite3.connect('mydatabase.db')

# Connect to SQLite database (create a new file if it doesn't exist)
conn = sqlite3.connect('mydatabase.db')
cursor = conn.cursor()

# Create a table if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS REGISTER (
        name TEXT PRIMARY KEY,
        email TEXT,
        password TEXT
    )
''')
conn.commit()

print("Connected to SQLite database")


@app.route('/')
def project():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route("/logout")
def logout():
    session.pop("username", None)
    return render_template('index.html')

@app.route('/hero')
def home():
    return render_template('index.html')

@app.route('/model')
def model():
    return render_template('model.html')

@app.route ('/login')
def login():
    return render_template('login.html')

@app.route ('/register')
def register():
    return render_template('signup.html')

# Modify the signup function to use SQLite
@app.route("/reg", methods=['POST', 'GET'])
def signup():
    msg = ''
    if request.method == 'POST':
        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]

        # Check if the username already exists
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM REGISTER WHERE name=?", (name,))
            account = cursor.fetchone()

        if account:
            return render_template('login.html', error=True)
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = "Invalid Email Address!"
        else:
            # Insert the new user into the SQLite database
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO REGISTER VALUES (?, ?, ?)", (name, email, password))

            msg = "You have successfully registered !"
    return render_template('login.html', msg=msg)


# Modify the login1 function to use SQLite
@app.route("/log", methods=['POST', 'GET'])
def login1():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        # Check if the credentials exist in the SQLite database
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM REGISTER WHERE EMAIL=? AND PASSWORD=?", (email, password))
            account = cursor.fetchone()

        if account:
            session['Loggedin'] = True
            session['id'] = account[1]  # Assuming 'name' is the first column
            session['email'] = account[2]  # Assuming 'email' is the second column
            return render_template('model.html')
        else:
            msg = "Incorrect Email/password"
            return render_template('login.html', msg=msg)
    else:
        return render_template('login.html')


# Load parking slot positions
with open('parkingSlotPosition', 'rb') as f:
    posList = pickle.load(f)

width, height = 107, 48

# Video feed
cap = cv2.VideoCapture('static/carParkingInput.mp4')
loop_video = True  # Variable to control the video loop


def check_parking_space(img_pro,image):
    space_counter = 0
    img_with_rectangles = image.copy()  # Create a copy of the image for drawing rectangles

    for pos in posList:
        x, y = pos

        img_crop = img_pro[y:y + height, x:x + width]
        count = cv2.countNonZero(img_crop)

        if count < 900:
            color = (0, 255, 0)
            thickness = 5
            space_counter += 1
        else:
            color = (0, 0, 255)
            thickness = 2

        cv2.rectangle(img_with_rectangles, pos, (pos[0] + width, pos[1] + height), color, thickness)
        cvzone.putTextRect(img_with_rectangles, str(count), (x, y + height - 3), scale=1,
                           thickness=2, offset=0, colorR=color)


    cvzone.putTextRect(img_with_rectangles, f'Free: {space_counter}/{len(posList)}', (100, 50), scale=3,
                       thickness=5, offset=20, colorR=(0, 200, 0))

    return img_with_rectangles  # Return the image with drawn rectangles


def generate_frames():
    global loop_video  # Use the global variable for loop control

    while loop_video:
        success, img = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to the beginning when reaching the end of the video
            continue  # Skip the current iteration and go to the next one

        # Apply parking space detection logic to each frame
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 1)
        img_threshold = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY_INV, 25, 16)
        img_median = cv2.medianBlur(img_threshold, 5)
        kernel = np.ones((3, 3), np.uint8)
        img_dilate = cv2.dilate(img_median, kernel, iterations=1)

        img_with_rectangles = check_parking_space(img_dilate, img)
        cv2.waitKey(5)
        _, buffer = cv2.imencode('.jpg', img_with_rectangles)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route for displaying the processed output
@app.route('/display')
def display():
    return render_template('display.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_loop')
def stop_loop():
    global loop_video
    loop_video = False  # Set loop_video to False to stop the video loop
    return render_template('index.html')



if __name__ == "__main__":
    app.run(debug=True)