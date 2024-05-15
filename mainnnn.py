import tkinter as tk
from tkinter import filedialog, font as tkfont
from PIL import Image, ImageTk
import face_recognition
import numpy as np
import os

def load_known_faces():
    person_images = {
        'Zaid Zitawi': ['images/Training Set/zaid zitawi kjfkd.jpg', 'images/Training Set/Zaid Zitawi.jpg', 'images/Training Set/zaidz3j.jpg'],
        'Elon Musk': ['images/Training Set/Elon Musk.jpg', 'images/Training Set/elonmusk4.jpg', 'images/Training Set/elon_musk.jpg', 'images/Training Set/Elon-Musk.jpg'],
        'The GOOOAT': ['images/Training Set/THE GOAAAAT.jpg', 'images/Training Set/THE GOAAAT.JPG', 'images/Training Set/THE GOOAT.jpg']
    }
    known_face_encodings = []
    known_face_names = []
    for person_name, image_paths in person_images.items():
        for image_path in image_paths:
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            if face_encodings:
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(person_name)
    return known_face_encodings, known_face_names

known_face_encodings, known_face_names = load_known_faces()

def load_testing_faces(directory='images/Testing Set'):
    test_encodings = []
    test_names = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(directory, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                test_encodings.append(encodings[0])
                test_names.append(filename.split('.')[0])
    return test_encodings, test_names

test_face_encodings, test_face_names = load_testing_faces()

def calculate_distance_matrix(known_encodings, test_encodings):
    distance_matrix = np.zeros((len(test_encodings), len(known_encodings)))
    for i, test_encoding in enumerate(test_encodings):
        for j, known_encoding in enumerate(known_encodings):
            distance = np.linalg.norm(test_encoding - known_encoding)
            distance_matrix[i, j] = distance
    return distance_matrix

distance_matrix = calculate_distance_matrix(known_face_encodings, test_face_encodings)
threshold = np.percentile(distance_matrix, 15)

def evaluate_fmr_fnmr(distance_matrix, threshold):
    fmr = 0
    fnmr = 0
    total_genuine = 0
    total_imposter = 0

    for i in range(len(test_face_encodings)):
        min_distance = np.min(distance_matrix[i])
        genuine_index = np.argmin(distance_matrix[i])
        is_genuine = test_face_names[i] == known_face_names[genuine_index]

        if is_genuine:
            total_genuine += 1
            if min_distance > threshold:
                fnmr += 1
        else:
            total_imposter += 1
            if min_distance <= threshold:
                fmr += 1

    fnmr_rate = fnmr / total_genuine if total_genuine else 0
    fmr_rate = fmr / total_imposter if total_imposter else 0
    return fnmr_rate, fmr_rate

fnmr_rate, fmr_rate = evaluate_fmr_fnmr(distance_matrix, threshold)
print(f"FMR: {fmr_rate}, FNMR: {fnmr_rate}")
print(threshold)

# GUI setup
root = tk.Tk()
root.title("Face Recognition System")
root.geometry("800x600")

title_font = tkfont.Font(family='Helvetica', size=18, weight="bold")
result_font = tkfont.Font(family='Helvetica', size=14)
title_label = tk.Label(root, text="Welcome to the Face Recognition System", font=title_font)
title_label.pack()

img_label = None

def select_image():
    global img_label, threshold
    file_path = filedialog.askopenfilename()
    if file_path:
        unknown_image = face_recognition.load_image_file(file_path)
        unknown_face_encodings = face_recognition.face_encodings(unknown_image)
        name = "Unknown"
        if unknown_face_encodings:
            face_distances = face_recognition.face_distance(known_face_encodings, unknown_face_encodings[0])
            best_match_index = np.argmin(face_distances)
            distance = face_distances[best_match_index]
            authentication_status = "OK" if distance <= threshold else "NOT OK"
            name = known_face_names[best_match_index] if authentication_status == "OK" else "Unknown"
        load = Image.open(file_path)
        load.thumbnail((300, 300))
        render = ImageTk.PhotoImage(load)
        if img_label is None:
            img_label = tk.Label(image=render)
            img_label.pack(pady=10)
        else:
            img_label.config(image=render)
        img_label.image = render
        result_label.config(text=f"Recognized: {name} - Authentication: {authentication_status}")

load_button = tk.Button(root, text="Load Image", command=select_image, font=result_font)
load_button.pack(pady=10)

result_label = tk.Label(root, text="", font=result_font)
result_label.pack()

root.mainloop()