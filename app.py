from flask import Flask, render_template, request, redirect, send_from_directory
import numpy as np
import json
import uuid
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")

label = [
'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust',
'Apple___healthy','Background_without_leaves','Beans___Angular_LeafSpot',
'Beans___Anthracnose','Beans___Healthy','Beans___Rust',
'Blueberry___healthy','Cherry___healthy','Cherry___Powdery_mildew',
'Corn___Cercospora_leaf_spot Gray_leaf_spot','Corn___Common_rust',
'Corn___healthy','Corn___Northern_Leaf_Blight','Grape___Black_rot',
'Grape___Esca_(Black_Measles)','Grape___healthy','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
'Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy',
'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Pigeon_pea___Healthy',
'Pigeon_pea___Leaf_Spot','Pigeon_pea___Leaf_webber','Pigeon_pea___Sterlic_mosaic',
'Potato___Early_blight','Potato___healthy','Potato___Late_blight','Raspberry___healthy',
'Soybean___healthy','Squash___Powdery_mildew','Strawberry___healthy','Strawberry___Leaf_scorch',
'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold',
'Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite',
'Tomato___Target_Spot','Tomato___Tomato_mosaic_virus','Tomato___Tomato_Yellow_Leaf_Curl_Virus',
'Tomato___healthy'
]

with open("plant_disease.json", 'r', encoding='utf-8') as f:
    plant_disease_list = json.load(f)

disease_lookup = {e['name']: e for e in plant_disease_list}


@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory('./uploadimages', filename)


@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')


def extract_features(image_path):
    image   = tf.keras.utils.load_img(image_path, target_size=(160, 160))
    feature = tf.keras.utils.img_to_array(image)
    return np.array([feature])


def model_predict(image_path):
    img    = extract_features(image_path)
    preds  = model.predict(img)        # shape: (1, num_classes)

    # Flatten to 1-D array of scores, one per class
    scores = preds[0].flatten()        # shape: (num_classes,)

    num_labels = len(label)            # 39

    # Only consider scores for indices we have labels for
    safe_scores = scores[:num_labels]

    top_idx   = int(np.argmax(safe_scores))
    top_score = float(safe_scores[top_idx])

    # Confidence: percentage of this class relative to total output
    total      = float(np.sum(safe_scores))
    confidence = round((top_score / total) * 100, 1) if total > 0 else round(top_score * 100, 1)
    confidence = min(max(confidence, 0.1), 99.9)

    predicted_label = label[top_idx]
    disease_info    = disease_lookup.get(predicted_label, {
        'name': predicted_label,
        'display_name':        {'en': predicted_label,       'sw': predicted_label},
        'symptoms':            {'en': 'Data not available.', 'sw': 'Taarifa haipatikani.'},
        'preventive_measures': {'en': 'Data not available.', 'sw': 'Taarifa haipatikani.'},
        'treatment':           {'en': 'Data not available.', 'sw': 'Taarifa haipatikani.'},
    })
    return disease_info, confidence


@app.route('/upload/', methods=['POST', 'GET'])
def uploadimage():
    if request.method == "POST":
        image     = request.files['img']
        temp_name = f"uploadimages/temp_{uuid.uuid4().hex}"
        save_path = f'{temp_name}_{image.filename}'
        image.save(save_path)
        prediction, confidence = model_predict(f'./{save_path}')
        return render_template('home.html',
                               result=True,
                               imagepath=f'/{save_path}',
                               prediction=prediction,
                               confidence=confidence)
    return redirect('/')


if __name__ == "__main__":
    app.run(debug=True)
