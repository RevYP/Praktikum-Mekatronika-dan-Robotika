// ==========================================
// DEFINISI PIN SENSOR (Sesuaikan dengan PCB Anda)
// ==========================================
const int numSensors = 8;
// Masukkan pin ESP32 yang terhubung ke 8 sensor (dari kiri ke kanan)
int sensorPins[numSensors] = {13, 12, 14, 27, 26, 25, 33, 32}; 
int sensorValues[numSensors];

// ==========================================
// DEFINISI PIN MOTOR DRIVER (Sesuaikan dengan PCB Anda)
// ==========================================
// Motor Kiri
const int ENA = 19;  // Pin PWM Kecepatan Motor Kiri
const int IN1 = 18;  // Arah Motor Kiri
const int IN2 = 5;

// Motor Kanan
const int ENB = 23;  // Pin PWM Kecepatan Motor Kanan
const int IN3 = 17;  // Arah Motor Kanan
const int IN4 = 16;

// ==========================================
// PARAMETER PID (Silakan Tuning Sesuai Lapangan)
// ==========================================
float Kp = 4.5;   // Proporsional: Koreksi utama berdasarkan jarak dari tengah
float Ki = 0.0;   // Integral: Mengatasi akumulasi error (opsional, seringkali 0)
float Kd = 12.0;  // Derivative: Meredam gerakan berayun/goyang

int lastError = 0;
int integral = 0;

// Kecepatan Base Robot (Range PWM ESP32: 0 - 255)
const int maxSpeed = 200; 
const int baseSpeed = 150; 

void setup() {
  Serial.begin(115200);

  // Setup Pin Sensor
  for (int i = 0; i < numSensors; i++) {
    pinMode(sensorPins[i], INPUT);
  }

  // Setup Pin Motor
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  // Setup PWM untuk ESP32 (Menggunakan LEDC jika di ESP32 jadul, atau analogWrite untuk core terbaru)
  // Catatan: Arduino IDE versi terbaru untuk ESP32 sudah mendukung analogWrite() langsung.
}

void loop() {
  int position = readSensors();
  
  // Jika posisi 0, berarti robot pas di tengah garis hitam
  int error = position - 0; 

  // Menghitung nilai PID
  int proportional = error;
  integral += error;
  int derivative = error - lastError;
  
  int controlSignal = (Kp * proportional) + (Ki * integral) + (Kd * derivative);
  lastError = error;

  // Tentukan kecepatan motor kanan dan kiri
  int leftMotorSpeed = baseSpeed + controlSignal;
  int rightMotorSpeed = baseSpeed - controlSignal;

  // Batasi kecepatan agar tidak melewati batas PWM (0 - maxSpeed)
  leftMotorSpeed = constrain(leftMotorSpeed, 0, maxSpeed);
  rightMotorSpeed = constrain(rightMotorSpeed, 0, maxSpeed);

  // Jalankan motor
  moveMotors(leftMotorSpeed, rightMotorSpeed);
}

// ==========================================
// FUNGSI MEMBACA SENSOR & MENGHITUNG POSISI
// ==========================================
int readSensors() {
  int activeSensors = 0;
  long sum = 0;

  // Ganti ke '1' jika garis Anda PUTIH dan background HITAM
  // Contoh di bawah berasumsi garis HITAM (sensor bernilai HIGH/1 saat kena hitam)
  for (int i = 0; i < numSensors; i++) {
    sensorValues[i] = digitalRead(sensorPins[i]);
    
    // Pembobotan sensor: Sensor tengah bernilai kecil, sensor ujung bernilai besar
    // Sensor:   S0    S1    S2    S3   S4   S5   S6   S7
    // Bobot:   -35   -25   -15   -5    5    15   25   35
    int weight = (i * 10) - 35; 

    if (sensorValues[i] == HIGH) { 
      sum += weight;
      activeSensors++;
    }
  }

  // Jika tidak ada sensor yang mendeteksi garis (robot keluar jalur)
  if (activeSensors == 0) {
    if (lastError < 0) return -40; // Terakhir di kiri, paksa belok kiri keras
    else return 40;                // Terakhir di kanan, paksa belok kanan keras
  }

  return sum / activeSensors;
}

// ==========================================
// FUNGSI PENGGERAK MOTOR
// ==========================================
void moveMotors(int leftSpeed, int rightSpeed) {
  // Motor Kiri Maju
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, leftSpeed);

  // Motor Kanan Maju
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  analogWrite(ENB, rightSpeed);
}