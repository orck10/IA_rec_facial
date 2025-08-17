import serial
import time
import serial.tools.list_ports
from pynput import keyboard

# Função para encontrar a porta do Arduino
def find_arduino_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        # Ajustei aqui para pegar automaticamente /dev/ttyACM*
        if "Arduino" in port.description or "CH340" in port.description or "USB" in port.description or "ACM" in port.device:
            return port.device
    return None

# Variáveis de controle
last_state = None
ser = None

# Função chamada quando uma tecla é pressionada
def on_press(key):
    global last_state, ser
    try:
        if key.char == 'q':
            if last_state != 'q':
                ser.write(b'q')
                print("Enviado: q")
                last_state = 'q'
    except AttributeError:
        pass

# Função chamada quando uma tecla é liberada
def on_release(key):
    global last_state, ser
    try:
        if key.char == 'q':
            if last_state != 'r':
                ser.write(b'r')
                print("Enviado: r")
                last_state = 'r'
    except AttributeError:
        pass
    except Exception as e:
        print(f"Erro ao liberar tecla: {e}")

# Main
try:
    port = find_arduino_port()
    if not port:
        raise Exception("Arduino não encontrado. Verifique a conexão USB.")
    
    ser = serial.Serial(port, 9600, timeout=1)
    print(f"✅ Conectado ao Arduino na porta {port}")
    time.sleep(2)

    # Inicia o listener do teclado
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        while True:
            if ser.in_waiting > 0:
                data = ser.readline().decode(errors="ignore").strip()
                if data:
                    print(f"📩 Recebido do Arduino: {data}")
            time.sleep(0.1)

except Exception as e:
    print(f"❌ Erro: {e}")
finally:
    if ser and ser.is_open:
        ser.close()
        print("🔌 Conexão serial fechada.")
