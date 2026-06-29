#include <Arduino.h>
#include <WiFi.h>
#include <Firebase_ESP_Client.h>

// Menyertakan helper code token & database dari library
#include "addons/TokenHelper.h"
#include "addons/RTDBHelper.h"

// ====================== KONFIGURASI ======================
const char* WIFI_SSID   = "Kos Graha";        // SSID WiFi Anda
const char* WIFI_PASS   = "kosgraha14";        // Password WiFi Anda

// Kredensial Firebase
#define FIREBASE_HOST   "monitoring-suhu-036-default-rtdb.asia-southeast1.firebasedatabase.app" // URL database tanpa https://
#define FIREBASE_AUTH   "t6HdI831BxftAVnAhAwcnZEOF73gXcygEmpUv4Yr"                   // Database Secret Key dari Firebase Console

#define LM35_PIN      3      // Pin Vout LM35 terhubung ke GPIO 3 (ADC1)
#define LED_PIN       15     // LED Onboard LOLIN S2 Mini (GPIO 15)
#define UPLOAD_MS     5000   // Interval upload data ke Firebase (5 detik)
// =========================================================

// ===== Objek Firebase =====
FirebaseData fbdo;
FirebaseData streamData;
FirebaseAuth auth;
FirebaseConfig config;

// ===== Shared Data & Mutex (Thread-Safe) =====
SemaphoreHandle_t dataMutex;
float g_temp      = 0.0;
bool  g_ledState  = false;
long  g_rssi      = 0;
bool  g_fbReady   = false;

// ===== Fungsi Helper: Kendali LED Onboard Active-Low =====
void updatePhysicalLED(bool state) {
  // LOLIN S2 Mini LED aktif saat LOW, dan mati saat HIGH
  digitalWrite(LED_PIN, state ? LOW : HIGH);
}

// ===== Fungsi Helper: Baca LM35 dengan Oversampling =====
float bacaSuhuLM35(int pin) {
  long totalAdc = 0;
  const int jumlahSampel = 20;
  for (int i = 0; i < jumlahSampel; i++) {
    totalAdc += analogRead(pin);
    delay(10);
  }
  float rataRataAdc = (float)totalAdc / jumlahSampel;
  
  // Rumus konversi ADC 12-bit (0-4095) ke Suhu Celcius (LM35)
  float temp = (rataRataAdc * 330.0) / 4095.0;
  return temp;
}

// ===== Firebase Stream Callback (Menerima Perintah LED secara Realtime) =====
void streamCallback(FirebaseStream data) {
  Serial.printf("Stream Event: Path=%s, Type=%s, Value=%s\n",
                data.streamPath().c_str(),
                data.dataType().c_str(),
                data.value().c_str());

  if (data.dataType() == "boolean") {
    bool state = data.boolData();
    xSemaphoreTake(dataMutex, portMAX_DELAY);
    g_ledState = state;
    xSemaphoreGive(dataMutex);
    updatePhysicalLED(state);
    Serial.printf("→ LED diubah via Firebase ke: %s\n", state ? "ON" : "OFF");
  }
}

void streamTimeoutCallback(bool timeout) {
  if (timeout) {
    Serial.println("Stream timeout, resume stream...");
  }
}

// ===== Task 1: Baca Sensor (FreeRTOS Core 0) =====
void sensorTask(void *pvParams) {
  for (;;) {
    float t = bacaSuhuLM35(LM35_PIN);
    
    xSemaphoreTake(dataMutex, portMAX_DELAY);
    g_temp      = t;
    g_rssi      = WiFi.RSSI();
    xSemaphoreGive(dataMutex);
    
    vTaskDelay(pdMS_TO_TICKS(2000)); // Baca sensor tiap 2 detik
  }
}

// ===== Task 2: Upload Firebase (FreeRTOS Core 0) =====
void firebaseUploadTask(void *pvParams) {
  for (;;) {
    // Pastikan WiFi dan Firebase siap
    if (WiFi.status() == WL_CONNECTED && g_fbReady) {
      float t; bool led; long rssi;
      xSemaphoreTake(dataMutex, portMAX_DELAY);
      t = g_temp; led = g_ledState; rssi = g_rssi;
      xSemaphoreGive(dataMutex);

      String basePath = "/device/ESP32S2_Lolin_01";
      
      // Kirim data sensor & status ke Realtime Database secara asinkron
      if (Firebase.RTDB.setFloatAsync(&fbdo, basePath + "/temperature", t)) {
        Serial.printf("→ Sent Temperature: %.1f °C\n", t);
      } else {
        Serial.printf("✗ Gagal kirim suhu: %s\n", fbdo.errorReason().c_str());
      }
      
      Firebase.RTDB.setIntAsync(&fbdo, basePath + "/rssi", rssi);
      Firebase.RTDB.setIntAsync(&fbdo, basePath + "/uptime", millis() / 1000);
    }
    
    vTaskDelay(pdMS_TO_TICKS(UPLOAD_MS)); // Upload setiap 5 detik
  }
}

void setup() {
  Serial.begin(115200);
  
  // Konfigurasi ADC & LED
  analogReadResolution(12);
  pinMode(LM35_PIN, INPUT);
  
  pinMode(LED_PIN, OUTPUT);
  updatePhysicalLED(g_ledState); // Matikan LED di awal
  
  dataMutex = xSemaphoreCreateMutex();
  
  Serial.println("\n========== LOLIN S2 Mini Modul 6 (Firebase) ==========");
  
  // Menghubungkan ke WiFi
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Menghubungkan ke WiFi");
  while (WiFi.status() != WL_CONNECTED) { 
    delay(500); 
    Serial.print("."); 
  }
  Serial.println("\n✓ WiFi Terhubung!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());
  
  // Setup Firebase
  config.host = FIREBASE_HOST;
  config.signer.tokens.legacy_token = FIREBASE_AUTH;
  
  // Inisialisasi Firebase Client
  Firebase.begin(&config, &auth);
  Firebase.reconnectWiFi(true);
  
  g_fbReady = true;
  Serial.println("✓ Firebase client diinisialisasi.");
  
  // Mulai Realtime Stream untuk mendengarkan status LED
  String ledPath = "/device/ESP32S2_Lolin_01/led";
  if (!Firebase.RTDB.beginStream(&streamData, ledPath)) {
    Serial.printf("✗ Gagal memulai stream: %s\n", streamData.errorReason().c_str());
  } else {
    Firebase.RTDB.setStreamCallback(&streamData, streamCallback, streamTimeoutCallback);
    Serial.println("✓ Stream LED diaktifkan.");
  }
  
  // Membuat Task FreeRTOS pada Core 0 (ESP32-S2 hanya memiliki Core 0)
  xTaskCreatePinnedToCore(sensorTask,         "SensorTask",   4096, NULL, 1, NULL, 0);
  xTaskCreatePinnedToCore(firebaseUploadTask, "FirebaseTask", 8192, NULL, 2, NULL, 0);
  
  Serial.println("✓ FreeRTOS Tasks aktif di Core 0!");
}

void loop() {
  // Kosong karena semua proses dilakukan oleh FreeRTOS di background
}
