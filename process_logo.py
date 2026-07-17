import sys
from PIL import Image

def process_image(input_path, output_path):
    img = Image.open(input_path).convert("RGBA")
    data = img.getdata()
    
    new_data = []
    for item in data:
        # item is (R, G, B, A)
        r, g, b, a = item
        # If it's close to white, make it transparent
        if r > 240 and g > 240 and b > 240:
            new_data.append((255, 255, 255, 0))
        else:
            # It's part of the logo. Let's make it an orange-ish color to match the theme
            # or just invert the darkness so it's bright.
            # Let's tint it with #e0a860 (224, 168, 96)
            # If the original text is dark (e.g., black), r,g,b are low.
            # We can use the luminance to determine the opacity of the new color.
            lum = (r * 0.299 + g * 0.587 + b * 0.114)
            # if lum is 0 (black), we want it fully colored.
            # if lum is 255 (white), it's already filtered, but just in case.
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
