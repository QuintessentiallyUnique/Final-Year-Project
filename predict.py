import tensorflow as tf
import numpy as np
import os

IMG_SIZE = (192, 192)
CLASS_NAMES = [
    'chicken_curry', 'grilled_salmon', 'pizza', 'samosa', 'garlic_bread',
    'steak', 'french_fries', 'hamburger', 'spaghetti_bolognese', 'falafel'
]

print("Loading Ensemble (5 Models)... Please wait.")
models = []

def load_inference_model(model_path):
    return tf.keras.models.load_model(model_path, compile=False)

for i in range(1, 6):
    model_path = f"model_fold_{i}.h5"
    if os.path.exists(model_path):
        try:
            models.append(load_inference_model(model_path))
            print(f"  - Model {i} loaded.")
        except Exception as e:
            print(f"  - Failed to load {model_path}: {e}")
    else:
        print(f"  - Warning: {model_path} missing.")

if not models:
    raise SystemExit("CRITICAL: No models could be loaded for inference.")

def process_and_predict(image_path):
    try:
        img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(img) / 255.0 
        img_array = tf.expand_dims(img_array, 0) 

        all_scores = [m.predict(img_array, verbose=0)[0] for m in models]
        
        ensemble_score = np.mean(all_scores, axis=0)
        
        best_idx = np.argmax(ensemble_score)
        print(f"\n>>> RESULT: {CLASS_NAMES[best_idx].upper()} ({ensemble_score[best_idx]:.2%})")
        
        top_2_idx = np.argsort(ensemble_score)[-3:-1][::-1]
        print(f"    Alternatives: {CLASS_NAMES[top_2_idx[0]]} ({ensemble_score[top_2_idx[0]]:.2%}), "
              f"{CLASS_NAMES[top_2_idx[1]]} ({ensemble_score[top_2_idx[1]]:.2%})")

    except Exception as e:
        print(f"Error processing image: {e}")


print("Drop an image in this folder and type the filename.")
print("Type 'exit' or 'q' to stop.")

while True:
    user_input = input("\nEnter image filename (e.g., test.jpg): ").strip()
    
    if user_input.lower() in ['exit', 'q']:
        print("Shutting down engine...")
        break
    
    if os.path.exists(user_input):
        process_and_predict(user_input)
    else:
        print(f"File '{user_input}' not found. Try again.")
