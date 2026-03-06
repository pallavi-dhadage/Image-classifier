import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from utils.predictor import YOLOPredictor

app = Flask(__name__)

# Configurations
UPLOAD_FOLDER = os.path.join('static', 'uploads')
RESULT_FOLDER = os.path.join('static', 'results')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Initialize predictor
predictor = YOLOPredictor()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result_filename = 'result_' + filename
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        
        detections = predictor.predict(filepath, result_path)
        
        return render_template('result.html', 
                               original_image=filename, 
                               result_image=result_filename, 
                               detections=detections)
    
    return redirect(request.url)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
