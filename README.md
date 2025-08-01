# Handwritten Digit Recognition Game

This is a web-based game that challenges you to draw a digit, which is then sent to a Python backend for recognition. The game tracks your score, provides predictions, and includes a live leaderboard using Firebase.

## Features

- **Real-time Digit Recognition:** Uses a pre-trained model on a Flask backend to predict digits drawn on a canvas.
- **Interactive Canvas:** A responsive drawing pad where you can draw digits.
- **Scoring System:** Tracks your correct predictions and provides instant feedback.
- **Firebase Integration:** Persists user data and a global leaderboard in Firestore.
- **Modern UI:** Built with React and styled with Tailwind CSS for a clean, responsive design.

## Tech Stack

**Frontend:**
- **React:** A JavaScript library for building user interfaces.
- **Vite:** A fast build tool for modern web projects.
- **Tailwind CSS v4:** A utility-first CSS framework for rapid styling.
- **Firebase:** A platform for building web and mobile applications.

**Backend:**
- **Flask:** A Python web framework.
- **TensorFlow/Keras:** The machine learning library used for the digit recognition model.

## Data and Model

Due to file size, the pre-trained model is not included in this repository.

**To run the backend, you must first download the model file:**

1.  Download `model.h5` from [**YOUR_MODEL_DOWNLOAD_LINK_HERE**].
2.  Place the `model.h5` file in your `backend/` directory.

## Setup and Installation

Follow these steps to get the project running on your local machine.

### Prerequisites

- Node.js & npm
- Python 3.x
- pip

### 1. Backend Setup

1.  Navigate to the backend directory:
    ```bash
    cd ../backend
    ```
2.  Create and activate a virtual environment (recommended):
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
3.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
4.  Start the Flask server on port 5000:
    ```bash
    flask run
    ```
    Your backend should now be running at `http://127.0.0.1:5000`.

### 2. Frontend Setup

1.  Navigate to the frontend directory:
    ```bash
    cd handwritten_digit_recognition/frontend
    ```
2.  Install the Node.js dependencies:
    ```bash
    npm install
    ```
3.  Open `src/App.jsx` and update the `backendUrl` variable to point to your local Flask server:
    ```javascript
    const backendUrl = "[http://127.0.0.1:5000](http://127.0.0.1:5000)";
    ```
4.  Start the Vite development server:
    ```bash
    npm run dev
    ```
The application should now be running in your browser at `http://localhost:5173/`.