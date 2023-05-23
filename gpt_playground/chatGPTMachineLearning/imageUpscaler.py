import numpy as np
from PIL import Image

# Load input image
input_image = Image.open('input.jpg')
input_tensor = torch.from_numpy(np.array(input_image))

# Upscale image
output_tensor = model(input_tensor)

# Convert output tensor to numpy array and save as image
output_array = output_tensor.detach().numpy()
output_image = Image.fromarray(np.uint8(output_array))
output_image.save('output.jpg')
