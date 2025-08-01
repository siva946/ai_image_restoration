from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from PIL import Image
import io
import base64
import torch
from diffusers import StableDiffusionInpaintPipeline
from gfpgan import GFPGANer
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'  # Replace with a secure key

# Initialize models
device = "cuda" if torch.cuda.is_available() else "cpu"

inpaint_pipe = None
try:
    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    inpaint_pipe = inpaint_pipe.to(device)
    if device == "cuda":
        inpaint_pipe.enable_model_cpu_offload()
        inpaint_pipe.unet = inpaint_pipe.unet.half()
        inpaint_pipe.vae = inpaint_pipe.vae.half()
        try:
            inpaint_pipe.unet = torch.compile(inpaint_pipe.unet)
            inpaint_pipe.vae = torch.compile(inpaint_pipe.vae)
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}")
except Exception as e:
    print(f"Error loading inpaint pipeline: {e}")

gfpgan= GFPGANer(model_path='weights/GFPGANv1.4.pth',upscale=1,arch='clean',channel_multiplier=2
)

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
        mask_bytes = None
        if mask_file:
            mask_bytes = mask_file.read()
            mask_file.seek(0)

        # Read image
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_shape = image.shape[:2]

        # Denoising
        if denoise:
            image = cv2.fastNlMeansDenoisingColored(image, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)

        # Generative fill
        if mask_bytes and prompt and inpaint_pipe is not None:
            mask = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
            # Resize image and mask to 512x512 for inpainting with aspect ratio preservation and padding
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
                inpaint_result = inpaint_pipe(
                    prompt=prompt,
                    image=image_pil,
                    mask_image=mask_pil,
                    num_inference_steps=10,
                    guidance_scale=8.0
                )
                image_resized = np.array(inpaint_result.images[0])

            # Crop the padded area and resize back to original size
            cropped_img = image_resized[top:top+nh, left:left+nw]
            image = cv2.resize(cropped_img, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_AREA)

        # Enhancement
        if enhance:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
            image = cv2.filter2D(image, -1, kernel)

        # Face enhancement
        if face_enhance:
            _, _, image = gfpgan.enhance(image, has_aligned=False, only_center_face=False, paste_back=True)

        # Resize to original size or upscaled size
        if upscale:
            # Optional: integrate Real-ESRGAN for better upscaling if available
            target_shape = (original_shape[1] * 4, original_shape[0] * 4)  # 4x original size
            image = cv2.resize(image, target_shape, interpolation=cv2.INTER_LANCZOS4)
        else:
            target_shape = (original_shape[1], original_shape[0])  # Original size
            image = cv2.resize(image, target_shape, interpolation=cv2.INTER_AREA)

        # Encode result
        _, buffer = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        restored_image = base64.b64encode(buffer).decode('utf-8')

        return jsonify({'restored_image': restored_image})
    except Exception as e:
        # Log error internally and return generic message
        print(f"Error processing image: {e}")
        return jsonify({'error': 'An error occurred during image processing.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
