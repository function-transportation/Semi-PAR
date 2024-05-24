from flask import Flask, request, render_template, redirect, url_for, session
from google.cloud.storage import Client
from google.oauth2 import service_account
from google import auth
from google.auth.transport import requests
import os
from datetime import timedelta

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # セッション管理のための秘密鍵

# GCSの設定
BUCKET_NAME = "sh_person_image"
IMAGE_DIR = 'test'
INITIAL_LABELS_FILE = "test/annotation/initial_label.txt"
ALREADY_LABELS_FILE = "test/annotation/already_label.txt"

if "GOOGLE_APPLICATION_CREDENTIAL" in os.environ:
    key = os.environ["GOOGLE_APPLICATION_CREDENTIAL"]
    credentials = service_account.Credentials.from_service_account_file(
        key,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    storage_client = Client(credentials=credentials)
else:
    credentials, _ = auth.default()
    if credentials.token is None:
        # Perform a refresh request to populate the access token of the
        # current credentials.
        credentials.refresh(requests.Request())
    storage_client = Client()

ATTRIBUTES = ['Hat', 'Glasses', 'ShortSleeve', 'LongSleeve', 'UpperStride', 'UpperLogo', 'UpperPlaid', 'UpperSplice', 
              'LowerStripe', 'LowerPattern', 'LongCoat', 'Trousers', 'Shorts', 'Skirt&Dress', 'Boots', 'HandBag', 
              'ShoulderBag', 'Backpack', 'HoldObjectsInFront', 'AgeOver60', 'Age18-60', 'AgeLess18', 'Female', 'Front', 
              'Side', 'Back']

def download_blob_to_string(bucket_name, blob_name):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.download_as_text()

# GCSにファイルをアップロードする関数
def upload_string_to_blob(bucket_name, content, blob_name):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(content)

# 署名付きURLを生成する関数
def generate_signed_url(bucket_name, blob_name, expiration=timedelta(minutes=15)):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    url = blob.generate_signed_url(
        version="v4",
        expiration=expiration, 
        service_account_email=credentials.service_account_email,
        access_token=credentials.token,
        method="GET",
    )
    return url

# アノテーションがまだされていない画像のリストを取得する関数
def get_annotation_candidates(initial_labels, already_labels):
    candidates = []
    for line in initial_labels:
        if line.strip():
            image_path = line.split(',')[0]
            if not any(image_path in al for al in already_labels) or any(al.split(',')[1:][ATTRIBUTES.index(attr)] == '-1' for al in already_labels if image_path in al for attr in session['selected_attributes']):
                candidates.append(image_path)
    return candidates

@app.route('/')
def index():
    return render_template('select_attributes.html', attributes=ATTRIBUTES)

@app.route('/start_annotation', methods=['POST'])
def start_annotation():
    session['selected_attributes'] = request.form.getlist('attributes')
    initial_labels_content = download_blob_to_string(BUCKET_NAME, INITIAL_LABELS_FILE)
    already_labels_content = download_blob_to_string(BUCKET_NAME, ALREADY_LABELS_FILE)
    
    initial_labels = initial_labels_content.split('\n')
    already_labels = already_labels_content.split('\n')
    
    candidates = get_annotation_candidates(initial_labels, already_labels)
    
    if candidates:
        session['candidates'] = candidates
        session['current_index'] = 0
        session['initial_labels'] = initial_labels
        session['already_labels'] = already_labels
        return redirect(url_for('annotate'))
    else:
        return "No images left to annotate."

@app.route('/annotate', methods=['GET', 'POST'])
def annotate():
    if request.method == 'POST':
        image_path = session['candidates'][session['current_index']]
        annotations = request.form.getlist('annotations')
        
        new_already_labels = []
        updated = False
        for line in session['already_labels']:
            if line.startswith(image_path):
                existing_labels = line.strip().split(',')
                for attr in session['selected_attributes']:
                    index = ATTRIBUTES.index(attr) + 1
                    existing_labels[index] = '1' if attr in annotations else '0'
                new_already_labels.append(','.join(existing_labels))
                updated = True
            else:
                new_already_labels.append(line.strip())
        
        if not updated:
            new_entry = [image_path] + ['-1'] * len(ATTRIBUTES)
            for attr in session['selected_attributes']:
                index = ATTRIBUTES.index(attr) + 1
                new_entry[index] = '1' if attr in annotations else '0'
            new_already_labels.append(','.join(new_entry))
        
        session['already_labels'] = new_already_labels
        upload_string_to_blob(BUCKET_NAME, '\n'.join(new_already_labels), ALREADY_LABELS_FILE)
        
        # 次の画像に進む
        session['current_index'] += 1
        if session['current_index'] >= len(session['candidates']):
            return "All images annotated!"
    
    # 現在の画像を取得
    current_index = session.get('current_index', 0)
    image_path = session['candidates'][current_index]
    signed_url = generate_signed_url(BUCKET_NAME, image_path)
    
    # 現在のアノテーションを取得
    current_annotations = ['-1'] * len(ATTRIBUTES)
    for line in session['already_labels']:
        if line.startswith(image_path):
            current_annotations = line.strip().split(',')[1:]
            break

    # 初期ラベルを反映
    for line in session['initial_labels']:
        if line.startswith(image_path):
            initial_annotations = line.strip().split(',')[1:]
            for i, value in enumerate(initial_annotations):
                if value == '1' and current_annotations[i] == '-1':
                    current_annotations[i] = '1'
            break
    
    print('signed_url:', signed_url)  # URLをコンソールに出力して確認
    return render_template('annotate.html', image_url=signed_url, image_path=image_path, current_annotations=current_annotations, selected_attributes=session['selected_attributes'], all_attributes=ATTRIBUTES)

@app.route('/next')
def next_image():
    if 'current_index' in session:
        session['current_index'] += 1
        if session['current_index'] >= len(session['candidates']):
            session['current_index'] = len(session['candidates']) - 1
    return redirect(url_for('annotate'))

@app.route('/prev')
def prev_image():
    if 'current_index' in session:
        session['current_index'] -= 1
        if session['current_index'] < 0:
            session['current_index'] = 0
    return redirect(url_for('annotate'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
