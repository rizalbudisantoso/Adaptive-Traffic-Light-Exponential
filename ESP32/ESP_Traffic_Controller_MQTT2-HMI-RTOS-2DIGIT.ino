#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

// === MODE CONFIGURATION ===
#define DEBUG_MODE false

// WiFi Configuration
const char* ssid = "iPhone Rizal";
const char* password = "rzlbdsntso";

// MQTT Configuration
const char* mqtt_server = "172.20.10.3";
const int mqtt_port = 1883;
const char* mqtt_topic = "trafficlight/durations";
const char* mqtt_client_id = "ESP32_TrafficLight_Wirosaban";

// Pin lampu lalu lintas
const int merah[] = {23, 33, 27, 19};
const int kuning[] = {22, 25, 14, 18};
const int hijau[] = {32, 26, 12, 5};
const char titik[] = {'A', 'B', 'C', 'D'};

// Durasi dalam milidetik (mendukung presisi tinggi)
const unsigned long durasiDarurat[4] = {30000, 45000, 30000, 45000};
unsigned long durasi[4] = {30000, 45000, 30000, 45000};

bool emergencyMode = false;

// Timer
unsigned long faseMulai = 0;
int indexFase = 0;

// ✅ Countdown menggunakan float untuk presisi desimal
float countdown[4] = {0.0, 0.0, 0.0, 0.0};
unsigned long lastUpdate = 0;

// MQTT objects
WiFiClient espClient;
PubSubClient mqttClient(espClient);

// RTT measurement
unsigned long lastMQTTReceived = 0;
unsigned long lastRequestTime = 0;
unsigned long mqttLatency = 0;

// === FreeRTOS Handles ===
TaskHandle_t taskTrafficLightHandle = NULL;
TaskHandle_t taskMQTTConnectionHandle = NULL;
TaskHandle_t taskSerialPrintHandle = NULL;

// Queue untuk serial print (non-blocking)
QueueHandle_t serialQueue;
#define SERIAL_QUEUE_SIZE 20  // ✅ Diperbesar untuk 10ms update

// Semaphore untuk proteksi shared variables
SemaphoreHandle_t xDurasiMutex;
SemaphoreHandle_t xCountdownMutex;
SemaphoreHandle_t xMQTTMutex;

// Struktur data untuk serial print queue
struct SerialMessage {
  char message[256];
  bool isDebug;
};

// === LOGGING FUNCTIONS ===
void logDebug(String message) {
  if (DEBUG_MODE) {
    SerialMessage msg;
    strncpy(msg.message, message.c_str(), 255);
    msg.message[255] = '\0';
    msg.isDebug = true;
    xQueueSend(serialQueue, &msg, 0);
  }
}

void logInfo(String message) {
  SerialMessage msg;
  strncpy(msg.message, message.c_str(), 255);
  msg.message[255] = '\0';
  msg.isDebug = false;
  xQueueSend(serialQueue, &msg, 0);
}

// === MQTT CALLBACK ===
void mqttCallback(char* topic, byte* payload, unsigned int length) {
  unsigned long receiveTime = millis();
  
  String debugInfo = "";
  if (DEBUG_MODE) {
    debugInfo = "\n====================================================\n";
    debugInfo += "📩 DATA RECEIVED FROM MQTT BROKER\n";
    debugInfo += "====================================================\n";
    debugInfo += "⏰ Received at: " + String(receiveTime) + " ms\n";
    logDebug(debugInfo);
  }
  
  String message;
  for (unsigned int i = 0; i < length; i++) {
    message += (char)payload[i];
  }
  
  StaticJsonDocument<1000> doc;
  DeserializationError err = deserializeJson(doc, message);
  
  if (!err && doc.containsKey("Barat")) {
    if (xSemaphoreTake(xDurasiMutex, portMAX_DELAY) == pdTRUE) {
      float barat = doc["Barat"];
      float selatan = doc["Selatan"];
      float timur = doc["Timur"];
      float utara = doc["Utara"];
      
      durasi[0] = max((unsigned long)(barat * 1000.0), 5000UL);
      durasi[1] = max((unsigned long)(selatan * 1000.0), 5000UL);
      durasi[2] = max((unsigned long)(timur * 1000.0), 5000UL);
      durasi[3] = max((unsigned long)(utara * 1000.0), 5000UL);
      
      xSemaphoreGive(xDurasiMutex);
    }
    
    if (lastRequestTime > 0) {
      mqttLatency = receiveTime - lastRequestTime;
      lastRequestTime = 0;
    }
    
    emergencyMode = false;
    
    String durInfo = "Durasi dari server: ";
    durInfo += "A=" + String(durasi[0]/1000.0, 2) + "s, ";
    durInfo += "B=" + String(durasi[1]/1000.0, 2) + "s, ";
    durInfo += "C=" + String(durasi[2]/1000.0, 2) + "s, ";
    durInfo += "D=" + String(durasi[3]/1000.0, 2) + "s";
    logInfo(durInfo);
    
    lastMQTTReceived = millis();
  } else {
    logInfo("Gagal parsing JSON dari server");
    if (DEBUG_MODE && err) {
      logDebug("JSON Error: " + String(err.c_str()));
    }
  }
}

// === TASK 1: TRAFFIC LIGHT CONTROL (Priority: Highest) ===
void taskTrafficLight(void *parameter) {
  TickType_t xLastWakeTime = xTaskGetTickCount();
  const TickType_t xFrequency = 10 / portTICK_PERIOD_MS;  // ✅ 10ms update cycle
  
  for (;;) {
    unsigned long sekarang = millis();
    unsigned long currentDurasi[4];
    
    if (xSemaphoreTake(xDurasiMutex, 5 / portTICK_PERIOD_MS) == pdTRUE) {
      memcpy(currentDurasi, durasi, sizeof(durasi));
      xSemaphoreGive(xDurasiMutex);
    }
    
    // ✅ Update countdown setiap 10ms dengan presisi tinggi
    if (sekarang - lastUpdate >= 10) {
      float deltaTime = (sekarang - lastUpdate) / 1000.0;
      lastUpdate = sekarang;
      
      if (xSemaphoreTake(xCountdownMutex, 5 / portTICK_PERIOD_MS) == pdTRUE) {
        for (int i = 0; i < 4; i++) {
          countdown[i] -= deltaTime;
          if (countdown[i] < 0) countdown[i] = 0;
        }
        xSemaphoreGive(xCountdownMutex);
      }
    }
    
    updateLampuTransisi();
    
    if (sekarang - faseMulai >= currentDurasi[indexFase]) {
      indexFase = (indexFase + 1) % 4;
      faseMulai = sekarang;
      
      if (indexFase == 0) {
        if (xSemaphoreTake(xMQTTMutex, 0) == pdTRUE) {
          if (mqttClient.connected()) {
            logDebug("🔄 Cycle completed - requesting update");
            lastRequestTime = millis();
            mqttClient.publish("trafficlight/request", "update");
          }
          xSemaphoreGive(xMQTTMutex);
        }
      }
      
      inisialisasiCountdown();
      setLampuLaluLintas();
    }
    
    vTaskDelayUntil(&xLastWakeTime, xFrequency);
  }
}

// === TASK 2: MQTT CONNECTION (Priority: Medium) ===
void taskMQTTConnection(void *parameter) {
  TickType_t xLastWakeTime = xTaskGetTickCount();
  const TickType_t xFrequency = 100 / portTICK_PERIOD_MS;
  
  connectWiFi();
  
  if (WiFi.status() == WL_CONNECTED) {
    vTaskDelay(1000 / portTICK_PERIOD_MS);
    
    mqttClient.setServer(mqtt_server, mqtt_port);
    mqttClient.setCallback(mqttCallback);
    mqttClient.setKeepAlive(60);
    mqttClient.setSocketTimeout(30);
    mqttClient.setBufferSize(1024);
    
    reconnectMQTT();
  }
  
  for (;;) {
    unsigned long sekarang = millis();
    
    if (xSemaphoreTake(xMQTTMutex, portMAX_DELAY) == pdTRUE) {
      if (!mqttClient.connected()) {
        if (sekarang - lastMQTTReceived > 10000) {
          xSemaphoreGive(xMQTTMutex);
          reconnectMQTT();
          lastMQTTReceived = sekarang;
          if (xSemaphoreTake(xMQTTMutex, portMAX_DELAY) != pdTRUE) {
            vTaskDelayUntil(&xLastWakeTime, xFrequency);
            continue;
          }
        }
      } else {
        mqttClient.loop();
      }
      xSemaphoreGive(xMQTTMutex);
    }
    
    vTaskDelayUntil(&xLastWakeTime, xFrequency);
  }
}

// === TASK 3: SERIAL PRINT (Priority: Lowest) ===
void taskSerialPrint(void *parameter) {
  SerialMessage msg;
  TickType_t xLastWakeTime = xTaskGetTickCount();
  // ✅ Update setiap 10ms untuk presisi maksimal
  const TickType_t xFrequency = 10 / portTICK_PERIOD_MS;
  
  for (;;) {
    // Proses message queue dengan limit untuk menghindari bottleneck
    int processedCount = 0;
    while (xQueueReceive(serialQueue, &msg, 0) == pdTRUE && processedCount < 5) {
      Serial.println(msg.message);
      processedCount++;
    }
    
    kirimSerialCountdown();
    
    vTaskDelayUntil(&xLastWakeTime, xFrequency);
  }
}

// === SETUP ===
void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("\n====================================================");
  Serial.println("🚦 Traffic Light Controller dengan FreeRTOS");
  Serial.println("====================================================");
  Serial.println("Version: 3.3 - Ultra High Precision (10ms Update)");
  Serial.println("Real-time Display with 0.01s Resolution");
  Serial.println("====================================================\n");
  
  for (int i = 0; i < 4; i++) {
    pinMode(merah[i], OUTPUT);
    pinMode(kuning[i], OUTPUT);
    pinMode(hijau[i], OUTPUT);
    
    digitalWrite(merah[i], HIGH);
    delay(200);
    digitalWrite(merah[i], LOW);
  }
  
  serialQueue = xQueueCreate(SERIAL_QUEUE_SIZE, sizeof(SerialMessage));
  xDurasiMutex = xSemaphoreCreateMutex();
  xCountdownMutex = xSemaphoreCreateMutex();
  xMQTTMutex = xSemaphoreCreateMutex();
  
  faseMulai = millis();
  lastUpdate = millis();
  inisialisasiCountdown();
  setLampuLaluLintas();
  
  xTaskCreatePinnedToCore(taskTrafficLight, "TrafficLight", 4096, NULL, 3, &taskTrafficLightHandle, 1);
  xTaskCreatePinnedToCore(taskMQTTConnection, "MQTTConnection", 8192, NULL, 2, &taskMQTTConnectionHandle, 0);
  xTaskCreatePinnedToCore(taskSerialPrint, "SerialPrint", 4096, NULL, 1, &taskSerialPrintHandle, 0);
  
  Serial.println("✅ FreeRTOS Tasks Created Successfully!");
  Serial.println("====================================================\n");
}

void loop() {
  vTaskDelay(1000 / portTICK_PERIOD_MS);
}

// === FUNGSI WIFI ===
void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  vTaskDelay(100 / portTICK_PERIOD_MS);
  
  logInfo("Mencoba menghubungkan ke WiFi...");
  WiFi.begin(ssid, password);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    vTaskDelay(500 / portTICK_PERIOD_MS);
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    String info = "Terhubung ke WiFi - IP: " + WiFi.localIP().toString();
    logInfo(info);
  } else {
    logInfo("Gagal terhubung ke WiFi");
    emergencyMode = true;
  }
}

// === FUNGSI MQTT ===
void reconnectMQTT() {
  int attempts = 0;
  
  while (!mqttClient.connected() && attempts < 5) {
    if (WiFi.status() != WL_CONNECTED) {
      logInfo("WiFi terputus. Mencoba reconnect WiFi...");
      connectWiFi();
      if (WiFi.status() != WL_CONNECTED) {
        break;
      }
      vTaskDelay(1000 / portTICK_PERIOD_MS);
    }
    
    if (mqttClient.state() != MQTT_DISCONNECTED) {
      mqttClient.disconnect();
      vTaskDelay(100 / portTICK_PERIOD_MS);
    }
    
    espClient.flush();
    espClient.stop();
    vTaskDelay(100 / portTICK_PERIOD_MS);
    
    if (mqttClient.connect(mqtt_client_id)) {
      logInfo("MQTT Connected!");
      
      vTaskDelay(100 / portTICK_PERIOD_MS);
      if (mqttClient.subscribe(mqtt_topic, 0)) {
        logDebug("Subscribed to topic");
      }
      
      vTaskDelay(100 / portTICK_PERIOD_MS);
      mqttClient.publish("trafficlight/status", "online", false);
      
      vTaskDelay(200 / portTICK_PERIOD_MS);
      lastRequestTime = millis();
      
      if (mqttClient.publish("trafficlight/request", "init", false)) {
        logDebug("Initial request sent");
      }
      
      emergencyMode = false;
      lastMQTTReceived = millis();
      break;
      
    } else {
      int delayTime = 2000 * (attempts + 1);
      if (attempts < 4) {
        vTaskDelay(delayTime / portTICK_PERIOD_MS);
      }
    }
    
    attempts++;
  }
  
  if (!mqttClient.connected()) {
    logInfo("Gagal koneksi MQTT. Mode darurat aktif.");
    emergencyMode = true;
    
    if (xSemaphoreTake(xDurasiMutex, portMAX_DELAY) == pdTRUE) {
      for (int i = 0; i < 4; i++) {
        durasi[i] = durasiDarurat[i];
      }
      xSemaphoreGive(xDurasiMutex);
    }
  }
}

// === FUNGSI TRAFFIC LIGHT ===
void updateLampuTransisi() {
  unsigned long sekarang = millis();
  unsigned long currentDurasi[4];
  
  if (xSemaphoreTake(xDurasiMutex, 5 / portTICK_PERIOD_MS) == pdTRUE) {
    memcpy(currentDurasi, durasi, sizeof(durasi));
    xSemaphoreGive(xDurasiMutex);
  }
  
  unsigned long sisa = currentDurasi[indexFase] - (sekarang - faseMulai);
  int faseSelanjutnya = (indexFase + 1) % 4;
  
  for (int i = 0; i < 4; i++) {
    if (i != indexFase && i != faseSelanjutnya) {
      digitalWrite(kuning[i], LOW);
      digitalWrite(hijau[i], LOW);
      digitalWrite(merah[i], HIGH);
    }
  }
  
  if (sisa <= 2000) {
    digitalWrite(hijau[indexFase], LOW);
    digitalWrite(kuning[indexFase], HIGH);
    digitalWrite(merah[indexFase], LOW);
    
    digitalWrite(merah[faseSelanjutnya], HIGH);
    digitalWrite(kuning[faseSelanjutnya], HIGH);
    digitalWrite(hijau[faseSelanjutnya], LOW);
  } else {
    setLampuLaluLintas();
  }
}

void setLampuLaluLintas() {
  for (int i = 0; i < 4; i++) {
    digitalWrite(hijau[i], i == indexFase ? HIGH : LOW);
    digitalWrite(merah[i], i == indexFase ? LOW : HIGH);
    digitalWrite(kuning[i], LOW);
  }
}

void inisialisasiCountdown() {
  unsigned long currentDurasi[4];
  
  if (xSemaphoreTake(xDurasiMutex, 5 / portTICK_PERIOD_MS) == pdTRUE) {
    memcpy(currentDurasi, durasi, sizeof(durasi));
    xSemaphoreGive(xDurasiMutex);
  }
  
  if (xSemaphoreTake(xCountdownMutex, 5 / portTICK_PERIOD_MS) == pdTRUE) {
    for (int i = 0; i < 4; i++) {
      if (i == indexFase) {
        countdown[i] = currentDurasi[i] / 1000.0;
      } else {
        float total = 0.0;
        int pos = indexFase;
        while (pos != i) {
          total += currentDurasi[pos] / 1000.0;
          pos = (pos + 1) % 4;
        }
        countdown[i] = total;
      }
    }
    xSemaphoreGive(xCountdownMutex);
  }
}

void kirimSerialCountdown() {
  float currentCountdown[4];
  
  if (xSemaphoreTake(xCountdownMutex, 5 / portTICK_PERIOD_MS) == pdTRUE) {
    memcpy(currentCountdown, countdown, sizeof(countdown));
    xSemaphoreGive(xCountdownMutex);
  }
  
  // ✅ Format dengan 2 desimal untuk presisi 10ms
  String output = "Countdown: ";
  for (int i = 0; i < 4; i++) {
    const char* warna = "Merah";
    if (digitalRead(hijau[i]) == HIGH) warna = "Hijau";
    else if (digitalRead(kuning[i]) == HIGH) warna = "Kuning";
    
    output += String(titik[i]) + String(warna) + "=" + String(currentCountdown[i], 2);
    if (i < 3) output += ", ";
  }
  
  // ✅ Status info setiap 100 iterasi (setiap 1 detik pada 10ms update)
  static int counter = 0;
  if (++counter % 100 == 0) {
    bool mqttConnected = false;
    if (xSemaphoreTake(xMQTTMutex, 5 / portTICK_PERIOD_MS) == pdTRUE) {
      mqttConnected = mqttClient.connected();
      xSemaphoreGive(xMQTTMutex);
    }
    
    output += " | MQTT:" + String(mqttConnected ? "OK" : "DISC");
    output += " RTT:" + String(mqttLatency) + "ms";
    output += " Mode:" + String(emergencyMode ? "DARURAT" : "NORMAL");
  }
  
  Serial.println(output);
}
