import math
from PIL import Image, ImageDraw, ImageFont

def build_collage(images: list[Image.Image], max_width: int = 1600) -> Image.Image:
    """
    Takes a list of PIL Images and stitches them into an optimal grid.
    Adds a red index label to the top-left of each image.
    """
    if not images:
        return Image.new('RGB', (100, 100), color='black')

    num_images = len(images)
    cols = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)

    # Standardize individual cell size (e.g. downscale to fit 1600px width limit)
    cell_w = max_width // cols
    # Maintain aspect ratio of the first image approx
    aspect = images[0].height / float(images[0].width)
    cell_h = int(cell_w * aspect)

    master_w = cols * cell_w
    master_h = rows * cell_h

    collage = Image.new('RGB', (master_w, master_h), color='black')

    for i, img in enumerate(images):
        col = i % cols
        row = i // cols
        
        # Resize and crop image to fit cell
        img_resized = img.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
        
        x = col * cell_w
        y = row * cell_h
        
        collage.paste(img_resized, (x, y))
        
        # Draw red number label 
        draw = ImageDraw.Draw(collage)
        
        # Draw background box for text visibility
        box_w, box_h = 60, 60
        draw.rectangle([x, y, x + box_w, y + box_h], fill=(255, 0, 0))
        
        # We don't have guaranteed TTF fonts, so we use default and scale via small image if needed,
        # but for simplicity default PIL font is used. To make it bigger, we manually draw thicker lines or multiple text pieces.
        # Actually, draw.text is fine for basic recognition, but let's draw large polygons if we really wanted to.
        # Default font:
        draw.text((x + 20, y + 20), str(i + 1), fill=(255, 255, 255))
        
    return collage
