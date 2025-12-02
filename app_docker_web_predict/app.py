from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from PIL import Image
import io
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import threading
import json
import time
import sys
import logging
import tempfile
import shutil


class CNNMultiTask(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten_dim = 128 * (TARGET_SIZE[0] // 8) * (TARGET_SIZE[1] // 8)
        self.shared_fc = nn.Sequential(nn.Linear(self.flatten_dim, 256), nn.ReLU())

        self.fc_glasses = nn.Linear(256, 1)
        self.fc_beard = nn.Linear(256, 1)
        self.fc_mustache = nn.Linear(256, 1)
        self.fc_color = nn.Linear(256, 5)
        self.fc_hair = nn.Linear(256, 3)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.shared_fc(x)
        return (
            self.fc_glasses(x),
            self.fc_beard(x),
            self.fc_mustache(x),
            self.fc_color(x),
            self.fc_hair(x),
        )


# ---------------- PREPROC ----------------
def segment_and_resize_images(input_folder, output_folder, target_size=(64, 64)):
    os.makedirs(output_folder, exist_ok=True)
    t0 = time.time()
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(".png"):
            continue
        img = Image.open(os.path.join(input_folder, filename)).convert("RGB")
        gray = img.convert("L")
        arr = np.array(gray)
        mask = arr < 250
        if not mask.any():
            cropped = img
        else:
            ys, xs = np.where(mask)
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            cropped = img.crop((x_min, y_min, x_max + 1, y_max + 1))
        resized = cropped.resize(target_size)
        resized.save(os.path.join(output_folder, filename))
    t1 = time.time()
    print(f"[PREPROC] images processed in {t1 - t0:.2f}s")


class FaceDatasetNoLabels(Dataset):
    """
    Dataset qui lit uniquement les images d'un dossier (jpg/png)
    et renvoie :
      - image : tensor (C,H,W)
      - filename : nom du fichier image
    """

    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        if len(self.images) == 0:
            raise RuntimeError(f"Aucune image trouvée dans {image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        else:
            # même comportement que dans script.py si transform=None
            image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0

        return {"image": image, "filename": img_name}


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
# Data folder is mounted at /app/data inside the container (we mount ../data:/app/data)
DATA_ROOT = os.path.join(APP_ROOT, 'data')

TARGET_SIZE = (64, 64)

app = Flask(__name__)

# Reduce noisy access logs from the development WSGI server (werkzeug)
# This will suppress lines like: "172.21.0.1 - - [..] \"GET /images/...\" 200 -"
try:
    werkzeug_logger = logging.getLogger('werkzeug')
    werkzeug_logger.setLevel(logging.WARNING)
except Exception:
    pass

# Ensure Python output is unbuffered so `print()` shows up immediately in docker logs
# and configure the root logger to stream to stdout.
try:
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
except Exception:
    pass
try:
    # Python 3.7+: make stdout line-buffered when redirected (e.g. in docker)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
except Exception:
    try:
        sys.stdout.flush()
    except Exception:
        pass

try:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
except Exception:
    pass

def list_subfolders(root):
    if not os.path.isdir(root):
        return []
    return [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]


def preprocess_image(path):
    img = Image.open(path).convert('RGB')
    img = img.resize(TARGET_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def predict_using_dataset(model, source_folder, target_size=(64, 64), batch_size=32, device='cpu', progress_callback=None):
    """
    Use `segment_and_resize_images` (from `train_ml_flow`) to produce a resized folder,
    then wrap it with `FaceDatasetNoLabels` (from `predict_mlflow`) and run batched inference.
    Returns list of dicts with keys: filename, glasses, beard, mustache, color, hair
    """
    
    # The source folder may be mounted read-only inside the container (docker compose :ro).
    # Create a temporary writable directory under APP_ROOT for resized images.
    resized_dir = None
    tmp_parent = APP_ROOT if os.path.isdir(APP_ROOT) else None
    try:
        resized_dir = tempfile.mkdtemp(prefix='resized_', dir=tmp_parent)
    except Exception:
        # fallback to system temp directory
        resized_dir = tempfile.mkdtemp(prefix='resized_')

    # produce resized images into tmp dir
    segment_and_resize_images(source_folder, resized_dir, target_size)

    ds = FaceDatasetNoLabels(resized_dir, transform=None)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    results = []
    total = len(ds)
    model.to(device).eval()
    with torch.no_grad():
        processed = 0
        for batch in loader:
            images = batch['image'].to(device)
            filenames = batch['filename']
            outs = model(images)
            g, b, m, c, h = outs
            prob_g  = (torch.sigmoid(g).view(-1) > 0.5).int().cpu().numpy()
            prob_b = torch.sigmoid(b).view(-1).cpu().numpy()
            prob_m = torch.sigmoid(m).view(-1).cpu().numpy()
            pred_c = torch.argmax(c, dim=1).cpu().numpy()
            pred_h = torch.argmax(h, dim=1).cpu().numpy()

            for i, fname in enumerate(filenames):
                results.append({
                    'filename': fname,
                    'glasses': float(prob_g[i]),
                    'beard': float(prob_b[i]),
                    'mustache': float(prob_m[i]),
                    'color': int(pred_c[i]),
                    'hair': int(pred_h[i]),
                })
            processed += images.size(0)
            # call progress callback if provided
            if progress_callback is not None:
                try:
                    progress_callback(processed, total)
                except Exception:
                    pass
    # cleanup temporary resized folder
    try:
        shutil.rmtree(resized_dir, ignore_errors=True)
    except Exception:
        pass
    return results


@app.route('/', methods=['GET'])
def index():
    sub = list_subfolders(DATA_ROOT)
    return render_template('index.html', folders=sub)


@app.route('/predict', methods=['POST'])
def predict():
    folder = request.form.get('folder')
    if not folder:
        return redirect(url_for('index'))
    folder_path = os.path.join(DATA_ROOT, folder)
    # If the folder doesn't exist, go back to index
    if not os.path.isdir(folder_path):
        return redirect(url_for('index'))

    # Prefer a model inside the selected folder if present
    user_model_path = request.form.get('model_path')
    local_model_candidate = os.path.join(folder_path, 'cnn_multitask_best_final.pth')
    if user_model_path and os.path.exists(user_model_path):
        model_path = user_model_path
    elif os.path.exists(local_model_candidate):
        model_path = local_model_candidate
    else:
        model_path = os.path.join(APP_ROOT, 'cnn_multitask_best_final.pth')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # check model class availability
    if CNNMultiTask is None:
        return {'status': 'error', 'message': 'CNNMultiTask not available: cannot load model. Ensure train_ml_flow.py is present and importable.'}, 500
    # load model
    model = CNNMultiTask().to(device)

    def robust_load_state(m, path, device):
        """Try several common checkpoint formats and adapt state_dict keys if needed."""
        if not os.path.exists(path):
            print('No model file found at:', path)
            return False
        try:
            ck = torch.load(path, map_location=device)
        except Exception as e:
            print('torch.load failed for', path, '->', e)
            return False

        # If ck is a dict, look for common keys
        if isinstance(ck, dict):
            for key in ('state_dict', 'model_state_dict', 'model'):
                if key in ck and isinstance(ck[key], dict):
                    sd = ck[key]
                    break
            else:
                sd = ck
        else:
            sd = ck

        # If state_dict keys use 'module.' prefix (DataParallel), strip it
        if isinstance(sd, dict):
            new_sd = {}
            for k, v in sd.items():
                new_k = k
                if k.startswith('module.'):
                    new_k = k[len('module.'):]
                new_sd[new_k] = v
            sd = new_sd

        try:
            m.load_state_dict(sd)
            print(f'Loaded model from: {path}')
            return True
        except Exception as e:
            print('load_state_dict failed:', e)
            # debug: show expected vs provided keys counts
            try:
                exp_keys = set(m.state_dict().keys())
                prov_keys = set(sd.keys()) if isinstance(sd, dict) else set()
                print(f'expected keys: {len(exp_keys)}, provided keys: {len(prov_keys)}')
                # show small sample
                print('expected sample keys:', list(exp_keys)[:5])
                print('provided sample keys:', list(prov_keys)[:5])
            except Exception:
                pass
            return False

    loaded_ok = robust_load_state(model, model_path, device)
    if not loaded_ok:
        print('Model failed to load or incompatible state dict; using freshly initialized model (this may explain uniform predictions).')

    # start prediction in background thread and return immediately
    progress_path = os.path.join(APP_ROOT, 'progress.json')

    def write_progress(state: dict):
        try:
            with open(progress_path, 'w', encoding='utf-8') as pf:
                json.dump(state, pf)
        except Exception as e:
            print('Failed to write progress:', e)

    def progress_cb(processed, total):
        pct = int(processed / total * 100) if total > 0 else 0
        write_progress({'status': 'running', 'processed': int(processed), 'total': int(total), 'percent': pct, 'message': f'{processed}/{total}'})

    def run_predict_and_write():
        try:
            write_progress({'status': 'running', 'processed': 0, 'total': 0, 'percent': 0, 'message': 'starting'})
            # call batched prediction
            results_local = []
            results_local = predict_using_dataset(model, folder_path, target_size=TARGET_SIZE, batch_size=32, device=device, progress_callback=progress_cb)
          
            # save results to a CSV under app folder
            import csv
            out_csv = os.path.join(APP_ROOT, 'last_results.csv')
            with open(out_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['filename', 'glasses', 'beard', 'mustache', 'color', 'hair'])
                writer.writeheader()
                for r in results_local:
                    writer.writerow(r)

            write_progress({'status': 'done', 'processed': len(results_local), 'total': len(results_local), 'percent': 100, 'message': 'finished', 'folder': folder})
        except Exception as e:
            print('Error during background prediction:', e)
            write_progress({'status': 'error', 'message': str(e)})

    # clear previous progress and start background thread
    write_progress({'status': 'queued', 'processed': 0, 'total': 0, 'percent': 0, 'message': 'queued'})
    t = threading.Thread(target=run_predict_and_write, daemon=True)
    t.start()
    return {'status': 'started', 'folder': folder}


@app.route('/progress')
def progress():
    progress_path = os.path.join(APP_ROOT, 'progress.json')
    if os.path.exists(progress_path):
        try:
            with open(progress_path, 'r', encoding='utf-8') as pf:
                data = json.load(pf)
            return data
        except Exception as e:
            return {'status': 'error', 'message': f'cannot read progress: {e}'}
    return {'status': 'idle', 'message': 'no active job'}


@app.route('/results')
def results():
    folder = request.args.get('folder')
    out_csv = os.path.join(APP_ROOT, 'last_results.csv')
    rows = []
    if os.path.exists(out_csv):
        import csv
        with open(out_csv, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                # convert types
                r['glasses'] = float(r['glasses'])
                r['beard'] = float(r['beard'])
                r['mustache'] = float(r['mustache'])
                r['color'] = int(r['color'])
                r['hair'] = int(r['hair'])
                rows.append(r)
    # attributes counts
    attrs = {
        'glasses': sum(1 for r in rows if r['glasses']>0.5),
        'beard': sum(1 for r in rows if r['beard']>0.5),
        'mustache': sum(1 for r in rows if r['mustache']>0.5),
    }
    return render_template('results.html', rows=rows, attrs=attrs, folder=folder)


@app.route('/images/<folder>/<filename>')
def serve_image(folder, filename):
    folder_path = os.path.join(DATA_ROOT, folder)
    return send_from_directory(folder_path, filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
