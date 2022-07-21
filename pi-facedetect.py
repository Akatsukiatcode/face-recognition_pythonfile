import face_detection
from PIL import Image, ImageDraw 
import numpy as np

shiv_image = face_detection.load_image_file("Shiv.jpeg")

shiv_face_encoding = face_detection.face_encodings(shiv_image)[0]

rahul_image = face_detection.load_image_file("Rahul.jpeg")

rahul_face_encoding = face_detection.face_encodings(rahul_image)[0]

known_face_names = [
    "Shiv.jpeg",
    "Rahul.jpeg"
]

unknown_image = face_detection.load_image_file("f1.jpg")

face_locations = face_detection.face_locations(unknown_image)
face_encodings = face_detection.face_encodings(unknown_image, face_locations)


pil_image = Image.fromarray(unknown_image)
draw = ImageDraw.Draw(pil_image)


for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    
    matches = face_detection.compare_faces(face_encodings, face_encoding)

    name = "rahul.jpeg"

    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]

    
    draw.rectangle(((left, top), (right, bottom)), outline=(48, 63, 159))

    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(48, 63, 159), outline=(48, 63, 159))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 0))


pil_image.show()

# pil_image.save("image_with_boxes.jpg")