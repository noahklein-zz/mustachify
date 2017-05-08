import cv2
from PIL import Image
import os
import random
import sys
import requests
import tempfile

MUSTACHES = [Image.open('mustaches/' + m).convert('RGBA')
             for m in os.listdir('mustaches')
             if not m.startswith('.')]
FACE_CASCADE = cv2.CascadeClassifier('frontalface.xml')


def fetch_img(url):
    temp = tempfile.NamedTemporaryFile(prefix='mustachify_')
    response = requests.get(url)
    temp.write(response.content)
    return temp


def faces(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
    return (
        (x, int(y + h * .65), x + w, y + h)
        for (x, y, w, h) in faces
    )


def resize_with_aspect_ratio(img, new_width):
    old_width, old_height = img.size
    new_height = int(new_width * old_height / old_width)
    return img.resize((new_width, new_height), Image.ANTIALIAS)


def paste_mustache(img, pos):
    (x1, y1, x2, _) = pos
    mustache = random.choice(MUSTACHES)
    mustache_resize = resize_with_aspect_ratio(mustache, x2 - x1)
    img.paste(mustache_resize, (x1, y1), mustache_resize)


tmp_img = fetch_img(sys.argv[1])
img_path = tmp_img.name
img = Image.open(img_path)
for face_pos in faces(img_path):
    paste_mustache(img, face_pos)

img.save('out.jpg', format="JPEG")
# img.show()
