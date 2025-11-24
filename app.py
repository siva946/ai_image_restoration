from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from PIL import Image
import base64
import torch
from diffusers import StableDiffusionInpaintPipeline
import os
import logging

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

logging.basicConfig(level=logging.INFO)

device = "cuda" if torch.cuda.is_available() else "cpu"
inpaint_pipe = None

def load_inpaint_model():
    global inpaint_pipe
    if inpaint_pipe is None:
        try:
            print(f"Loading inpaint model on {device}...")
            inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                safety_checker=None,
            )
            if device == "cuda":
                inpaint_pipe.to("cuda")
                inpaint_pipe.enable_model_cpu_offload()
            try:
                inpaint_pipe.enable_xformers_memory_efficient_attention()
            except:
                pass
            inpaint_pipe.enable_attention_slicing(1)
            inpaint_pipe.enable_vae_slicing()
            print("✓ Model loaded")
        except Exception as e:
            print(f"✗ Model load failed: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        image_file = request.files['image']
        denoise = request.form.get('denoise') == 'true'
        upscale = request.form.get('upscale') == 'true'
        enhance = request.form.get('enhance') == 'true'
        face_enhance = request.form.get('faceEnhance') == 'true'
        prompt = request.form.get('prompt', '')
        mask_file = request.files.get('mask', None)

        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_shape = image.shape[:2]

        if denoise:
            image = cv2.fastNlMeansDenoisingColored(image, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)

        if mask_file and prompt:
            load_inpaint_model()
            if inpaint_pipe is not None:
                mask_bytes = mask_file.read()
                mask = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
                
                def resize_and_pad(img, size=512, is_mask=False):
                    h, w = img.shape[:2]
                    scale = size / max(h, w)
                    nh, nw = int(h * scale), int(w * scale)
                    resized_img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA)
                    if len(img.shape) == 2 or is_mask:
                        padded_img = np.zeros((size, size), dtype=img.dtype)
                    else:
                        padded_img = np.zeros((size, size, img.shape[2]), dtype=img.dtype)
                    top = (size - nh) // 2
                    left = (size - nw) // 2
                    padded_img[top:top+nh, left:left+nw] = resized_img
                    return padded_img, top, left, nh, nw

                image_resized, top, left, nh, nw = resize_and_pad(image, 512, is_mask=False)
                mask_resized, _, _, _, _ = resize_and_pad(mask, 512, is_mask=True)

                if np.sum(mask_resized) > 0:
                    image_pil = Image.fromarray(image_resized)
                    mask_pil = Image.fromarray(mask_resized)
                    with torch.inference_mode():
                        inpaint_result = inpaint_pipe(
                            prompt=prompt,
                            image=image_pil,
                            mask_image=mask_pil,
                            num_inference_steps=10,
                            guidance_scale=8.0
                        )
                    image_resized = np.array(inpaint_result.images[0])
                    if device == "cuda":
                        torch.cuda.empty_cache()

                cropped_img = image_resized[top:top+nh, left:left+nw]
                image = cv2.resize(cropped_img, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_AREA)

        if enhance:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
            image = cv2.filter2D(image, -1, kernel)

        if upscale:
            target_shape = (original_shape[1] * 4, original_shape[0] * 4)
            image = cv2.resize(image, target_shape, interpolation=cv2.INTER_LANCZOS4)
        else:
            target_shape = (original_shape[1], original_shape[0])
            image = cv2.resize(image, target_shape, interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        restored_image = base64.b64encode(buffer).decode('utf-8')

        return jsonify({'restored_image': restored_image})
    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        return jsonify({'error': 'Processing failed'}), 500

if __name__ == '__main__':
    app.run(debug=True)
