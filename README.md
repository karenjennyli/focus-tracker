# Focus Tracker App

A web application to track your focus time, distractions and productivity during work sessions. This is a capstone project by Arnav Arora, Karen Li, and Rohan Sonecha for 18-500 ECE Design Experience at Carnegie Mellon University.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

You will need to have Node.js and npm installed on your machine. You can download Node.js and npm from [here](https://nodejs.org/en/download/).

### Installing

Clone the repository to your local machine.

```
git clone https://github.com/karenjennyli/focus-tracker.git
```

Navigate to the project directory and install the required Python packages in a virtual environment.

```
cd [path to focus-tracker]
pip install -r requirements.txt
```

Navigate to the frontend directory and install the required Node.js packages.

```
cd [path to focus-tracker]/django-focus-tracker-react/frontend
npm install
```

Navigate to the backend directory and run the following commands to create the database.
```
cd [path to focus-tracker]/django-focus-tracker-react/backend
python manage.py makemigrations
python manage.py migrate
```

##  Running the Application

Navigate to the frontend directory and run the following command. This will start the React development server and open the application in your default web browser.

```
cd [path to focus-tracker]/django-focus-tracker-react/frontend
npm start
```

Navigate to the backend directory. Set the environment variables for the Python backend scripts, then run the server.

```
cd [path to focus-tracker]/django-focus-tracker-react/backend
export VENV_PATH='[path to virtual environment]'
export FOCUS_TRACKER_DIR='[path to focus-tracker]'
python3 manage.py runserver
```

Navigate to the video processing directory and run the following command. This will start the video processing backend.
```
cd [path to focus-tracker]/video_processing
python3 run.py
```

To access the application, open your web browser and go to [http://localhost:3000/](http://localhost:3000/).

## Developing

Set environment variables.
```
export FOCUS_TRACKER_DIR='[path to focus-tracker]'
export BACKEND_DIR='$FOCUS_TRACKER_DIR/django-focus-tracker-react/backend'
export FRONTEND_DIR='$FOCUS_TRACKER_DIR/django-focus-tracker-react/frontend'
export VIDEO_PROCESSING_DIR='$FOCUS_TRACKER_DIR/video_processing'
export VENV_PATH='[path to virtual environment]'
```

Run the setup script.
```
cd $FOCUS_TRACKER_DIR
./setup.sh
```

Run the backend server.
```
cd $BACKEND_DIR
python3 manage.py runserver
```

Run the frontend server.
```
cd $FRONTEND_DIR
npm start
```

Open the application in your web browser at [http://localhost:3000/](http://localhost:3000/).

## Built With

* [React](https://reactjs.org/)
* [Django](https://www.djangoproject.com/)
* [Node.js](https://nodejs.org/en/)
* [npm](https://www.npmjs.com/)
* [OpenCV](https://opencv.org/)
* [MediaPipe](https://mediapipe.dev/)
* [Inference](https://inference.roboflow.com/)
* [Supervision](https://supervision.roboflow.com/)

## Authors

* **Arnav Arora** - [arnavarora1111](https://github.com/arnavarora1111)
* **Karen Li** - [karenjennyli](https://github.com/karenjennyli)
* **Rohan Sonecha** - [rohansonecha](https://github.com/rohansonecha)

## Additional Information

For more information, please refer to our [project website](http://course.ece.cmu.edu/~ece500/projects/s24-teame0/).