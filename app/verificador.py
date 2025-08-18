import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import os

#comunicação arduino
import serial
import time
import serial.tools.list_ports

# Encontrar a porta do Arduino
def find_arduino_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        # Ajustei aqui para pegar automaticamente /dev/ttyACM*
        if "Arduino" in port.description or "CH340" in port.description or "USB" in port.description or "ACM" in port.device:
            return port.device
    return None

# Carregar o modelo treinado
def load_trained_model(model_path, num_classes):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Funcao para pegar nomes do dataset directory
def get_class_names(root_dir):
    class_names = []
    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if os.path.isdir(class_dir):
            class_names.append(class_name)
    return class_names

def commit_status(status, ser):

    print("Teste")
    ser.write(status)
    data = ser.readline().decode(errors="ignore").strip()
    if data:
        print(f"📩 Recebido do Arduino: {data}")
    time.sleep(0.1)
    



# Real-time da webcam
def real_time_face_recognition(model_path, root_dir, ser):

    # Carrega nomes
    class_names = get_class_names(root_dir)
    num_classes = len(class_names)
    
    # carrega modelos
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(model_path, num_classes)
    model.to(device)
    
    # Transform for input images
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Carregar Haar Cascade para detecção de rosto
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print("Starting real-time face recognition. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        # Converter para escala de cinza para detecção de rosto
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        memory = 'r'

        for (x, y, w, h) in faces:
            # Extrair ROI facial
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (224, 224))  
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face)
            
            # Aplicar transforms
            face_tensor = transform(face_pil).unsqueeze(0).to(device)
            
            # Verifica
            with torch.no_grad():
                output = model(face_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                predicted_class = class_names[predicted.item()]
                confidence_score = confidence.item() * 100
                 
                # Exibir somente se a confiança estiver acima de um limite (por exemplo, 70%)
                if confidence_score > 70 and predicted_class != "Unknown":
                    label = f"{predicted_class} ({confidence_score:.1f}%)"
                    if(memory == 'r'):
                        commit_status(b"q",ser)
                        memory = 'q'
                else:
                    label = "Unknown"
                
                # Desenhar retângulo e label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
        commit_status(b"r",ser)
        memory = 'r'
        #Mostra o quadro
        cv2.imshow('Real-Time Face Recognition', frame)
        
        
        # Interromper o loop ao pressionar a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    print("Reconhecimento facial em Real-time fechado!")

# Variáveis de controle
last_state = None
ser = None



# Examplo de uso
if __name__ == "__main__":

    #carrega serial arduino
    port = find_arduino_port()
    if not port:
        raise Exception("Arduino não encontrado. Verifique a conexão USB.")
    
    ser = serial.Serial(port, 9600, timeout=1)
    print(f"✅ Conectado ao Arduino na porta {port}")
    time.sleep(0.3)
    
    model_path = "face_recognition_model.pth"
    dataset_dir = "face_dataset"
    try:
        real_time_face_recognition(model_path, dataset_dir,ser)
    except Exception as e:
        print(f"❌ Erro: {e}")
    finally:
        # Liberar recursos
        if ser and ser.is_open:
            ser.close()
            print("🔌 Conexão serial fechada.")