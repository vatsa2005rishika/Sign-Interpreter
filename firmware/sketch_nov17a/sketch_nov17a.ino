/*
 * SIGN SPEAK GLOVE FIRMWARE
 * -------------------------
 * Features:
 * 1. Custom I2C Pins (SDA=25, SCL=26) to free up defaults.
 * 2. High-speed JSON transmission (50Hz).
 * 3. Robust Wi-Fi reconnection logic.
 * 4. Optimized memory management (No Strings).
 */

#include <WiFi.h>
#include <Wire.h>
#include <ArduinoJson.h>
#include <HTTPClient.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

// --- 1. WIFI & SERVER CONFIGURATION ---
const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASS = "YOUR_WIFI_PASSWORD";
const char* SERVER_IP = "YOUR_LAPTOP_IP";
const int   SERVER_PORT = 8000;

// --- 2. PIN CONFIGURATION ---
// Alternative I2C Pins (Non-Default)
#define I2C_SDA_PIN 18
#define I2C_SCL_PIN 19

// Hall Sensor Pins
const int HALL_PIN_THUMB  = 32;
const int HALL_PIN_INDEX  = 36; // VP
const int HALL_PIN_MIDDLE = 39; // VN
const int HALL_PIN_RING   = 34;
const int HALL_PIN_PINKY  = 35;

// Status LED
#define LED_PIN 2

// --- 3. SYSTEM SETTINGS ---
#define SAMPLE_RATE_HZ 50
const char* DEVICE_ID = "glove01";

// --- GLOBAL OBJECTS ---
Adafruit_MPU6050 mpu;
WiFiClient espClient;
HTTPClient http;

char serverUrl[128];
unsigned long lastSampleTime = 0;
const unsigned long SAMPLE_INTERVAL_MS = 1000 / SAMPLE_RATE_HZ;
uint32_t packet_seq = 0;

// Static Buffer for JSON (Prevents Heap Fragmentation)
char jsonBuffer[1024];

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  Serial.println("\n--- SignSpeak Glove Booting ---");

  // 1. Disable WiFi Power Saving (Crucial for high speed)
  WiFi.setSleep(false);

  // 2. Setup I2C with CUSTOM PINS
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);

  // 3. Initialize MPU6050
  if (!mpu.begin()) {
    Serial.println("❌ MPU6050 Not Found! Check wiring (SDA=25, SCL=26)");
    while (1) {
      digitalWrite(LED_PIN, !digitalRead(LED_PIN));
      delay(100);
    }
  }
  Serial.println("✅ MPU6050 Connected");

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  // 4. Setup Hall Sensors
  pinMode(HALL_PIN_THUMB, INPUT);
  pinMode(HALL_PIN_INDEX, INPUT);
  pinMode(HALL_PIN_MIDDLE, INPUT);
  pinMode(HALL_PIN_RING, INPUT);
  pinMode(HALL_PIN_PINKY, INPUT);

  // 5. Prepare Server URL
  snprintf(serverUrl, sizeof(serverUrl), "http://%s:%d/data", SERVER_IP, SERVER_PORT);

  // 6. Connect to Wi-Fi
  connectToWifi();

  // 7. Configure HTTP Client timeouts (Prevent freezing)
  http.setReuse(true);
  http.setTimeout(100); // 100ms timeout
}

void loop() {
  unsigned long currentMillis = millis();

  if (currentMillis - lastSampleTime >= SAMPLE_INTERVAL_MS) {
    lastSampleTime = currentMillis;

    if (WiFi.status() == WL_CONNECTED) {
      sendSensorData();
    } else {
      Serial.println("⚠️ Wi-Fi Lost. Reconnecting...");
      connectToWifi();
    }
  }
}

void sendSensorData() {
  // --- A. READ SENSORS ---
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  int h1 = analogRead(HALL_PIN_THUMB);
  int h2 = analogRead(HALL_PIN_INDEX);
  int h3 = analogRead(HALL_PIN_MIDDLE);
  int h4 = analogRead(HALL_PIN_RING);
  int h5 = analogRead(HALL_PIN_PINKY);

  // --- B. BUILD JSON (Using Static Buffer) ---
  StaticJsonDocument<768> doc;

  doc["device_id"] = DEVICE_ID;
  doc["seq"] = packet_seq++;
  doc["ts_device"] = millis();
  doc["sample_rate_hz"] = SAMPLE_RATE_HZ;

  JsonObject mpuObj = doc.createNestedObject("mpu6050");
  JsonObject accObj = mpuObj.createNestedObject("acc");
  accObj["x"] = a.acceleration.x;
  accObj["y"] = a.acceleration.y;
  accObj["z"] = a.acceleration.z;

  JsonObject gyroObj = mpuObj.createNestedObject("gyro");
  gyroObj["x"] = g.gyro.x;
  gyroObj["y"] = g.gyro.y;
  gyroObj["z"] = g.gyro.z;

  JsonArray hallsArr = doc.createNestedArray("halls");
  hallsArr.add(h1);
  hallsArr.add(h2);
  hallsArr.add(h3);
  hallsArr.add(h4);
  hallsArr.add(h5);

  // Serialize directly to char array (Faster, safer)
  serializeJson(doc, jsonBuffer, sizeof(jsonBuffer));

  // --- C. SEND HTTP POST ---
  // Re-initiating begin() every time ensures we recover from socket closures
  if(http.begin(espClient, serverUrl)) {
    http.addHeader("Content-Type", "application/json");

    int responseCode = http.POST(jsonBuffer);

    if (responseCode > 0) {
      // Toggle LED on success to show activity
      digitalWrite(LED_PIN, !digitalRead(LED_PIN));
    } else {
      Serial.print("E"); // Error code: http.errorToString(responseCode).c_str()
    }
    http.end(); // Free resources immediately
  } else {
    Serial.print("C"); // Connection failed locally
  }
}

void connectToWifi() {
  Serial.print("Connecting to Wi-Fi: ");
  Serial.println(WIFI_SSID);

  WiFi.begin(WIFI_SSID, WIFI_PASS);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    digitalWrite(LED_PIN, !digitalRead(LED_PIN)); // Blink fast while connecting
    delay(250);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    digitalWrite(LED_PIN, HIGH); // Solid LED on connect
    Serial.println("\n✅ Wi-Fi Connected");
    Serial.print("   IP: "); Serial.println(WiFi.localIP());
  } else {
    Serial.println("\n❌ Wi-Fi Failed.");
  }
}