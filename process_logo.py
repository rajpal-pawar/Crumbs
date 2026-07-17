import sys
from PIL import Image

def process_image(input_path, output_path):
    img = Image.open(input_path).convert("RGBA")
    
    # Crop to just the main logo, skipping the bottom sentence
    # bbox: x1=230, y1=300, x2=1320, y2=620
    img = img.crop((230, 300, 1320, 620))
    
    data = img.getdata()
    
    new_data = []
    for item in data:
        # item is (R, G, B, A)
        r, g, b, a = item
        # If it's close to white, make it transparent
        if r > 240 and g > 240 and b > 240:
            new_data.append((255, 255, 255, 0))
        else:
            # It's part of the logo. 
            lum = (r * 0.299 + g * 0.587 + b * 0.114)
            intensity = 1.0 - (lum / 255.0)
            
            # The Crumbs theme accent is (224, 168, 96)
            out_r = int(224)
            out_g = int(168)
            out_b = int(96)
            out_a = int(a * intensity)
            
            # If the original was fully transparent, keep it.
            if a == 0:
                new_data.append(item)
            else:
                new_data.append((out_r, out_g, out_b, out_a))

    img.putdata(new_data)
    img.save(output_path, "PNG")
    print(f"Saved {output_path}")

if __name__ == "__main__":
    process_image("ui/public/logo.png", "ui/public/logo-transparent.png")
