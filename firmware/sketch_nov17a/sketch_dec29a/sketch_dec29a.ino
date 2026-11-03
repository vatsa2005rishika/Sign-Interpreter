/*
 * ESP32 GLOVE - OFFLINE MODE (Local Hotspot Only)
 *
 * * Sends sensor data via HTTP POST to local laptop.
 * * NO Internet/NTP required.
 * * Uses millis() for relative timestamping.
 */

// --- LIBRARIES ---
#include <WiFi.h>
#include <Wire.h>
#include <ArduinoJson.h>
#include <HTTPClient.h>
// REMOVED: #include "time.h" and <sys/time.h> (Not needed for offline)

// MPU6050 Libraries
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

// --- 1. CONFIGURATION ---
const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASS = "YOUR_WIFI_PASSWORD";

// Your Laptop's Local IP (Still needed!)
// Check this via ipconfig every time you toggle the hotspot
const char* SERVER_IP = "YOUR_LAPTOP_IP";

// Device
const char* DEVICE_ID = "glove01";

// Sensors
#define SAMPLE_RATE_HZ 50

// Your exact pins
const int HALL_PIN_THUMB  = 32;
const int HALL_PIN_INDEX  = 36; // VP
const int HALL_PIN_MIDDLE = 39; // VN
const int HALL_PIN_RING   = 34;
const int HALL_PIN_PINKY  = 35;

// --- END OF CONFIGURATION ---

// --- GLOBAL VARS & OBJECTS ---
const long SAMPLE_INTERVAL_MS = 1000 / SAMPLE_RATE_HZ;
unsigned long lastSampleTime = 0;

Adafruit_MPU6050 mpu;
WiFiClient espClient;
HTTPClient http;
char serverEndpoint[100];
uint32_t seq = 0;

// --- SETUP ---
void setup() {
  Serial.begin(115200);
  Serial.println("Booting ESP32 (Offline Mode)...");

  snprintf(serverEndpoint, sizeof(serverEndpoint), "http://%s:8000/data", SERVER_IP);

  // Optimization: Reuse connection
  http.setReuse(true);

  Wire.begin();
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    while (1) { delay(10); }
  }
  Serial.println("MPU6050 Found!");
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  // Connect to Wi-Fi (Local only)
  setupWifi();

  // REMOVED: setupNTP(); -- We don't need internet time!
}

// --- MAIN LOOP ---
void loop() {
  unsigned long currentTime = millis();

  if (currentTime - lastSampleTime >= SAMPLE_INTERVAL_MS) {
    lastSampleTime = currentTime;
    if (WiFi.status() == WL_CONNECTED) {
      postTelemetry();
    } else {
      setupWifi();
    }
  }
}

// --- TASK FUNCTIONS ---
void postTelemetry() {
  // 1. Read sensors
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  // 2. Prepare JSON
  StaticJsonDocument<384> doc;

  doc["device_id"] = DEVICE_ID;
  doc["seq"] = seq++;

  // CHANGED: Send milliseconds since boot instead of complex date
  // This is enough for the model to know the "order" of events
  doc["ts_device"] = millis();

  JsonObject mpu6050 = doc.createNestedObject("mpu6050");
  mpu6050.createNestedObject("acc")["x"] = a.acceleration.x;
  mpu6050["acc"]["y"] = a.acceleration.y;
  mpu6050["acc"]["z"] = a.acceleration.z;
  mpu6050.createNestedObject("gyro")["x"] = g.gyro.x;
  mpu6050["gyro"]["y"] = g.gyro.y;
  mpu6050["gyro"]["z"] = g.gyro.z;

  JsonArray hallsData = doc.createNestedArray("halls");
  hallsData.add(analogRead(HALL_PIN_THUMB));
  hallsData.add(analogRead(HALL_PIN_INDEX));
  hallsData.add(analogRead(HALL_PIN_MIDDLE));
  hallsData.add(analogRead(HALL_PIN_RING));
  hallsData.add(analogRead(HALL_PIN_PINKY));

  doc["sample_rate_hz"] = SAMPLE_RATE_HZ;

  String payload;
  serializeJson(doc, payload);

  // 3. Send HTTP POST request (Fire & Forget)
  if (http.begin(espClient, serverEndpoint)) {
    http.addHeader("Content-Type", "application/json");

    int httpCode = http.POST(payload);

    // We don't care about the response right now, just speed
    if (httpCode <= 0) {
      Serial.print("E"); // Error indicator
    }

    http.end();
  }
}

// --- UTILITY FUNCTIONS ---

void setupWifi() {
  if (WiFi.status() == WL_CONNECTED) return;
  Serial.println("Connecting to Local Hotspot...");
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  int retries = 0;
  while (WiFi.status() != WL_CONNECTED && retries < 20) {
    delay(500); Serial.print(".");
    retries++;
  }

  if(WiFi.status() == WL_CONNECTED) {
    Serial.println("\nConnected to Local Network.");
    Serial.print("IP: "); Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nConnection failed.");
  }
}