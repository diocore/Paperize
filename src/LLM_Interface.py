import ollama
from PIL import Image
from io import BytesIO


def find_closest_aspect_ratio(aspect_ratio, aspect_ratios):
  closest_ratio = min(aspect_ratios, key=lambda x: abs(aspect_ratio - x))
  return closest_ratio

path:str = '../Images/Art/random_wallart_x2_-upscaled-high fidelity v2-4x-2.png'

with Image.open(path) as img:
  # Scale the image (e.g., to half its original size)
  img = img.resize((img.width // 2, img.height // 2))

  # Convert the image to bytes
  img_byte_arr = BytesIO()
  img.save(img_byte_arr, format='JPEG')
  img_bytes = img_byte_arr.getvalue()

  style = ["abstract", "minimal", "modern", "surrealism", "wildlife"]

  response = ollama.chat(model='llava-llama3', messages=[
    {
      'role': 'user',
      'content': f'Assign styles to this images bases on the styles in this list {style}. Only list the styles in a list ordered by the most fitting to least fitting and only use applicable styles',
      'images': [img_bytes]
    },
  ])
  print(response['message']['content'])
  image_aspect_ratio = 3456 / 4864
  aspect_ratio_values = [0.7,1.0,1.33,1.78]
  print(image_aspect_ratio)
  closest_ratio_value = find_closest_aspect_ratio(image_aspect_ratio, aspect_ratio_values)
  print(closest_ratio_value)


