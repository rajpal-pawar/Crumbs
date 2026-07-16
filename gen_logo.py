from PIL import Image, ImageDraw

img = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
# Draw a big white circle
d.ellipse([(64, 64), (448, 448)], fill='white')
# Draw a smaller transparent circle inside to make a ring
d.ellipse([(128, 128), (384, 384)], fill=(0, 0, 0, 0))
img.save('logo.png')
