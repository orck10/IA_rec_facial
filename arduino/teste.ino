void setup() {
  Serial.begin(9600); // Inicia a comunicação serial com baud rate de 9600
  while (!Serial) {
    ; // Aguarda a conexão serial (necessário para alguns modelos como Leonardo)
  }
  pinMode(2, OUTPUT); // Configura pino 2 como saída
  pinMode(3, OUTPUT); // Configura pino 3 como saída
  digitalWrite(2, LOW); // Inicialmente, pino 2 desligado
  digitalWrite(3, HIGH); // Inicialmente, pino 3 ligado (tecla 'q' não pressionada)
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read(); // Lê o comando recebido
    if (command == 'q') {
      digitalWrite(2, HIGH); // Liga o pino 2
      digitalWrite(3, LOW);  // Desliga o pino 3
      Serial.println("Pino 2 ligado, pino 3 desligado");
    } else if (command == 'r') {
      digitalWrite(2, LOW);  // Desliga o pino 2
      digitalWrite(3, HIGH); // Liga o pino 3
      Serial.println("Pino 2 desligado, pino 3 ligado");
    }
  }
}