from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
from ultralytics import YOLO
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = YOLO(r'D:\VirtualEye\Flask\best.pt')  # Load the YOLO model

# Define upload and prediction folders
UPLOAD_FOLDER = 'static/uploads'
PREDICTION_FOLDER = 'static/predictions'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4', 'avi', 'mkv'}

# Configure Flask app folders
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICTION_FOLDER'] = PREDICTION_FOLDER

# Ensure the necessary directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICTION_FOLDER, exist_ok=True)

# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route: Home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

# Route: About page
@app.route('/about')
def about():
    return render_template('about.html')

# Route: Contact page
@app.route('/contact')
def contact():
    return render_template('contact.html')

# Route: Prediction page
@app.route('/prediction_page', endpoint='prediction_page')
def prediction_page():
    return render_template('prediction-page.html')

# Route: Results page (static)
@app.route('/results')
def results():
    return render_template('results.html')

# Route: Handle file uploads and YOLO predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Initialize detection flag
        drowning_detected = False

        # Run YOLO prediction based on file type
        results = model(filepath, save=True, project=app.config['PREDICTION_FOLDER'], exist_ok=True)

        for result in results:
            if result.names and result.boxes:
                classes = [result.names[int(cls)] for cls in result.boxes.cls]

                if any(cls.lower() == 'drowning' for cls in classes):
                    drowning_detected = True
                    break  # Exit early if drowning is detected

        # Redirect with detection flag as query param
        return redirect(url_for('result', original_filename=filename, drowning=drowning_detected))
       

    return "Invalid file type"

# Route: Display prediction results
@app.route('/result/<original_filename>')
def result(original_filename):
    drowning = request.args.get('drowning', 'False') == 'True'
    # Find the latest prediction directory
    prediction_dirs = [
        os.path.join(app.config['PREDICTION_FOLDER'], d)
        for d in os.listdir(app.config['PREDICTION_FOLDER'])
        if os.path.isdir(os.path.join(app.config['PREDICTION_FOLDER'], d))
    ]
    if not prediction_dirs:
        return "No prediction directories found"

    latest_folder = max(prediction_dirs, key=os.path.getctime)

    # Find the first prediction file
    prediction_files = os.listdir(latest_folder)
    if not prediction_files:
        return "No predictions found"

    # Return the prediction result
    latest_file = prediction_files[0]
    
    result_file_url = url_for('static', filename=f'predictions/{os.path.basename(latest_folder)}/{latest_file}')
    return render_template('results.html', result_file=result_file_url, drowning_detected=drowning)


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=False)
