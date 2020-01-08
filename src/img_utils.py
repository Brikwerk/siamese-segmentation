from PIL import Image
import torchvision.transforms.functional as F
import numpy as np


class ResizePad(object):
    def __init__(self, size):
        self.size = (size, size)
    
    def __call__(self, image, target):
        # Padding/resizing image
        resize_image = square_pad(image).resize(self.size, Image.BILINEAR)
        resize_image = F.to_tensor(resize_image)

        # Padding/resizing masks
        masks = target["masks"]
        new_masks = []
        for i in range(len(masks)):
            mask = target["masks"][i]
            pil_mask = Image.fromarray(mask.numpy(), "L")
            resize_mask = square_pad(pil_mask, image_type="L").resize(self.size, Image.NEAREST)
            resize_mask = np.array(resize_mask, dtype=np.uint8)
            resize_mask = np.where(resize_mask > 0, 1, 0)
            new_masks.append(F.to_tensor(resize_mask).squeeze())

        target["masks"] = new_masks

        return resize_image, target


def square_pad(img, image_type="RGB"):
    width, height = img.size
    size = max(width, height)
    square_img = Image.new(image_type, (size, size))

    width_center = int((size - width) / 2)
    height_center = int((size - height) / 2)
    square_img.paste(img, (width_center, height_center))

    return square_img
