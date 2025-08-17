import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import numpy as np
import uuid

# Conjunto de dados personalizado para imagens faciais (para treinamento a partir de imagens salvas)
class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        
        # Load images and labels
        for idx, class_name in enumerate(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                self.class_to_idx[class_name] = idx
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.png')):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(idx)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Função para capturar imagens da webcam para criação de conjunto de dados
def capture_images(output_dir, person_name, num_images=100):
    if not os.path.exists(os.path.join(output_dir, person_name)):
        os.makedirs(os.path.join(output_dir, person_name))
    
    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    count = 0
    
    print(f"Capturing {num_images} images for {person_name}. Press 'q' to stop early.")
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            # Extract face ROI
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (224, 224))  # Resize to match model input
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert to RGB
            face_pil = Image.fromarray(face)
            
            # Save image
            img_path = os.path.join(output_dir, person_name, f"{uuid.uuid4()}.jpg")
            face_pil.save(img_path)
            count += 1
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.imshow('Capturing Faces', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {count} images for {person_name}.")

# Função principal de treinamento
def train_face_recognition(root_dir, num_epochs=10, batch_size=32, learning_rate=0.001):
    # Transforms
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Dataset and DataLoader
    dataset = FaceDataset(root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Number of classes (people)
    num_classes = len(dataset.class_to_idx)
    
    # Load pre-trained model
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
    
    # Save the model
    torch.save(model.state_dict(), 'face_recognition_model.pth')
    print("Training complete. Model saved as 'face_recognition_model.pth'")

#Main do treinamento
if __name__ == "__main__":
    dataset_dir = "face_dataset"
    
    print("Digite o nome do usuario!")
    nome = input("Digite seu nome: ")

    # Step 1: Capture images for each person
    capture_images(dataset_dir, nome, num_images=50)
    
    # Step 2: Train the model
    train_face_recognition(dataset_dir, num_epochs=10)

    print("Treinamento finalizado!")