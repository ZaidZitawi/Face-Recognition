import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import face_recognition
import numpy as np
import os
import matplotlib.pyplot as plt

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

def calculate_distance_matrix(known_encodings, test_encodings):
    distance_matrix = np.zeros((len(test_encodings), len(known_encodings)))
    for i, test_encoding in enumerate(test_encodings):
        for j, known_encoding in enumerate(known_encodings):
            distance = np.linalg.norm(test_encoding - known_encoding)
            distance_matrix[i, j] = distance
    return distance_matrix

def plot_roc_and_find_eer(distances):
    thresholds = np.linspace(np.min(distances), np.max(distances), 100)
    fmr_list, fnmr_list = [], []
    
    for threshold in thresholds:
        fmr = np.sum(distances <= threshold) / np.size(distances)
        fnmr = np.sum(distances > threshold) / np.size(distances)
        fmr_list.append(fmr)
        fnmr_list.append(fnmr)
    
    print_lists(fmr_list, fnmr_list, thresholds)
    print(f"\n\nDistances between faces are:{distance_matrix}\n\n")
    print(f"flatten distance matrix:{flat_distances}")

    fmr_array = np.array(fmr_list)
    fnmr_array = np.array(fnmr_list)
    eer_index = np.nanargmin(np.abs(fmr_array - fnmr_array))
    eer = (fmr_array[eer_index] + fnmr_array[eer_index]) / 2
    
    plt.figure()
    plt.plot(fmr_array, fnmr_array, label='ROC Curve')
    plt.scatter(fmr_array[eer_index], fnmr_array[eer_index], color='red', label=f'EER: {eer:.4f}')
    plt.title('ROC Curve and EER')
    plt.xlabel('False Match Rate (FMR)')
    plt.ylabel('False Non-Match Rate (FNMR)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return eer

def print_lists(fmr_list, fnmr_list, thresholds):
    print("\nDetailed FMR, FNMR, and Thresholds:")
    for i, (fmr, fnmr, threshold) in enumerate(zip(fmr_list, fnmr_list, thresholds)):
        print(f"Index {i:03d}: FMR={fmr:.6f}, FNMR={fnmr:.6f}, Threshold={threshold:.4f}")

known_face_encodings, known_face_names = load_known_faces()
test_face_encodings, test_face_names = load_testing_faces()
distance_matrix = calculate_distance_matrix(known_face_encodings, test_face_encodings)
flat_distances = distance_matrix.flatten()
eer = plot_roc_and_find_eer(flat_distances)
print(f"Equal Error Rate (EER): {eer}")

# GUI setup
root = tk.Tk()
root.title("Face Recognition System")
root.geometry("1000x600")

style = ttk.Style()
style.configure('TButton', font=('Helvetica', 12))

title_frame = tk.Frame(root)
title_frame.pack(fill=tk.X)

title_label = tk.Label(title_frame, text="Face Recognition System", font=('Helvetica', 18, 'bold'))
title_label.pack(pady=20)

content_frame = tk.Frame(root)
content_frame.pack(fill=tk.BOTH, expand=True)

left_frame = tk.Frame(content_frame, width=200)
left_frame.pack(side=tk.LEFT, fill=tk.Y)

right_frame = tk.Frame(content_frame)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        unknown_image = face_recognition.load_image_file(file_path)
        unknown_face_encodings = face_recognition.face_encodings(unknown_image)
        if unknown_face_encodings:
            face_distances = face_recognition.face_distance(known_face_encodings, unknown_face_encodings[0])
            best_match_index = np.argmin(face_distances)
            distance = face_distances[best_match_index]
            authentication_status = "OK" if distance <= eer else "NOT OK"
            name = known_face_names[best_match_index] if authentication_status == "OK" else "Unknown"
            update_result(name, authentication_status, file_path)

def update_result(name, status, image_path):
    result_label.config(text=f"Recognized: {name}\nAuthentication: {status}")
    img = Image.open(image_path)
    img.thumbnail((250, 250))
    photo = ImageTk.PhotoImage(img)
    img_label.config(image=photo)
    img_label.image = photo

load_button = ttk.Button(left_frame, text="Load Image", command=select_image)
load_button.pack(pady=10)

result_label = tk.Label(right_frame, text="", font=('Helvetica', 14))
result_label.pack(pady=20)

img_label = tk.Label(right_frame)
img_label.pack(pady=20)

root.mainloop()
