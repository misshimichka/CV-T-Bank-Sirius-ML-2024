import os

import torch
import argparse
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import pipeline

from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator

device = "cuda"

sam = sam_model_registry["default"](checkpoint="/kaggle/working/sam_vit_h_4b8939.pth").to(device=device)
predictor = SamPredictor(sam)
mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=32, points_per_batch=64)

inpaint = AutoPipelineForInpainting.from_pretrained("Lykon/dreamshaper-8-inpainting", torch_dtype=torch.float16).to(
    device=device
)
inpaint.enable_model_cpu_offload()

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
model = model.to(device)

pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ru", device=device)


def load_file(image_url):
    image = cv2.imread(image_url)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    new_size = (image.shape[1] // 8 * 8, image.shape[0] // 8 * 8)
    resized_image = cv2.resize(image, new_size)

    return resized_image


def find_best_mask(masks):
    idx = 0 if masks[0]["bbox"][0] < 10 else 1
    best_mask = np.uint8(masks[idx]["segmentation"] * 1)

    return best_mask


def prepare_mask(mask):
    fill_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))

    filled_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, fill_kernel)
    cleaned_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_OPEN, clean_kernel)
    blurred_mask = cv2.GaussianBlur(cleaned_mask, (35, 35), 0)

    inverted_mask = np.logical_not(blurred_mask) * 1

    pil_mask = Image.fromarray(blurred_mask * 255)

    return pil_mask, inverted_mask


def rgb_to_rgba(image, mask):
    image = cv2.bitwise_and(image, image, mask=mask.astype("uint8"))
    rgba_image = np.dstack(
        (image, np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8) + 255)
    )
    alpha_channel = (rgba_image[:, :, 0:3] == [0, 0, 0]).all(2)
    rgba_image[alpha_channel] = (0, 0, 0, 0)

    pil_rgba_image = Image.fromarray(rgba_image)

    return pil_rgba_image


def generate_mask(filename):
    image = load_file(filename)

    masks = mask_generator.generate(image)

    best_mask = find_best_mask(masks)

    return prepare_mask(best_mask)


def generate_background(filename, mask, inverted_mask):
    image = load_file(filename)
    tensor_image = load_image(Image.fromarray(image))
    tensor_mask = load_image(mask)

    prompt = "an object in a smooth background, smooth texture, realistic"
    negative_prompt = "ugly, extra items"

    generated_image = inpaint(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=tensor_image,
        mask_image=tensor_mask,
        guidance_scale=15,
        num_inference_steps=20,
        height=tensor_image.height,
        width=tensor_image.width,
        strength=0.9,
    ).images[0]

    rgba_image = rgb_to_rgba(image, inverted_mask)

    generated_image = generated_image.convert("RGBA")
    generated_image.paste(rgba_image, (0, 0), rgba_image)

    return generated_image


def generate_caption(image):
    prompt = ""
    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda", torch.float16)

    outputs = model.generate(**inputs)
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

    return generated_text


def translate_caption(caption):
    translation = pipe(caption)

    return translation[0]["translation_text"]


def main():
    #     parser = argparse.ArgumentParser(
    #         prog='Marketplace Photo Editor',
    #         description='This program highlights the background of the image, changes it and adds a caption.',
    #         epilog='Wish you choco biscuits')

    #     parser.add_argument('file_directory', default="/kaggle/input/sirius-2024-cv/sirius_data")

    #     args = parser.parse_args()
    work_dir = "/kaggle/input/sirius-2024-cv/sirius_data/"

    files = os.listdir(work_dir)
    os.makedirs("./output/masks", exist_ok=True)
    os.makedirs("./output/images", exist_ok=True)

    for filename in tqdm(files):
        pillow_mask, inverted_mask = generate_mask(work_dir + filename)
        pillow_mask.save(f"./output/masks/{filename}.png")
        torch.cuda.empty_cache()

        generated_image = generate_background(work_dir + filename, pillow_mask, inverted_mask)
        generated_image.save(f"./output/images/{filename}.png")
        torch.cuda.empty_cache()

        caption = generate_caption(generated_image)
        translated_caption = translate_caption(caption)
        torch.cuda.empty_cache()

        f = open("./output/captions.txt", mode="a+", encoding="utf-8")
        f.write(translated_caption)
        f.write("\n")
        f.close()

    print("Mission completed!")
    print(f"Files saved in {os.getcwd() + '/output'}.")


if __name__ == "__main__":
    main()
