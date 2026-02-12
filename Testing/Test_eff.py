import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# --- CONFIGURATION ---
CLASS_NAMES = ['angular_leaf_spot', 'bean_rust', 'healthy']
# Automatically find the model file relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, '..', 'Models', 'efficientnet_plants.pth') 
# (Note: I added .pth extension, make sure your file has it!)

def load_model():
    print(f"Loading EfficientNet from: {MODEL_PATH}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Recreate the Architecture (Must match training!)
    model = models.efficientnet_b0(pretrained=False)
    # Resize the final layer to 3 classes
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    
    # 2. Load the Weights
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        exit()
        
    model = model.to(device)
    model.eval() # Set to evaluation mode
    return model, device

def predict_image(image_path):
    model, device = load_model()
    
    # 3. Define Transform (Resize -> Tensor -> Normalize)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 4. Open and Predict
    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
        # 5. Interpret Results
        top_prob, top_class_id = torch.topk(probabilities, 1)
        class_name = CLASS_NAMES[top_class_id.item()]
        confidence = top_prob.item() * 100
        
        print(f"\n🌿 Result: {class_name.upper()}")
        print(f"📊 Confidence: {confidence:.2f}%")
        return class_name
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")

if __name__ == "__main__":
    # Allow user to type image path manually
    img_path = input("Enter the path to your leaf image (e.g. C:/Users/Downloads/leaf.jpg): ")
    # Remove quotes if user copied as path
    img_path = img_path.strip('"').strip("'")
    predict_image(img_path)