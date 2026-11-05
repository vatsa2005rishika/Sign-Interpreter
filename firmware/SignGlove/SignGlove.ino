/*
 * SignGlove.ino  -  ESP32 glove firmware (steady-rate version)
 *
 * What changed compared to the old sketch, and why:
 *  1. Sensors are read by their OWN task on an exact clock (100 Hz).
 *     Before, reading and sending were in one function, so a slow network made the
 *     glove sample slowly (you measured ~20 Hz with 200 ms gaps).
 *  2. A second task sends readings in small batches (5 readings = 50 ms).
 *     The reading task never waits for the network. If the network stalls, the
 *     queue fills and readings are dropped; the server SEES that as gaps in "seq".
 *  3. Nobody waits for the server to think. The server only stores the batch.
 *  4. Wi-Fi power saving is OFF (WiFi.setSleep(false)). The default saving mode makes
 *     the radio nap between beacons, which adds random delays of 100+ ms.
 *  5. Each finger value is the average of 4 analog reads (less noise).
 *  6. I2C runs at 400 kHz (faster MPU6050 reads).
 *  7. Wi-Fi name/password live in secrets.h (kept out of the repository).
 *
 * Libraries (Arduino Library Manager): Adafruit MPU6050, Adafruit Unified Sensor.
 * ArduinoJson is no longer needed.
 *
 * Row format sent to the server (one array per reading, always this order):
 *   [seq, ts_ms, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, hall0..hall4]
 */
#include <WiFi.h>
#include <Wire.h>
#include <HTTPClient.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include "secrets.h"

// ---- settings ---------------------------------------------------------------
#define SAMPLE_RATE_HZ   100
#define BATCH_SIZE       5      // readings per HTTP request (5 @ 100 Hz = a request every 50 ms)
#define QUEUE_LEN        256    // ~2.5 s of readings can wait if the network stalls
#define HALL_OVERSAMPLE  4
const char* DEVICE_ID = "glove01";
const int HALL_PINS[5] = {32, 36, 39, 34, 35};   // thumb, index, middle, ring, pinky (all ADC1)

// ---- data -------------------------------------------------------------------
struct Sample {
  uint32_t seq;
  uint32_t ts;
  float ax, ay, az, gx, gy, gz;
  uint16_t h[5];
};

Adafruit_MPU6050 mpu;
QueueHandle_t sampleQueue;
WiFiClient espClient;
HTTPClient http;
char endpoint[96];
volatile uint32_t droppedSamples = 0;
volatile uint32_t postFailures = 0;

// ---- task 1: read sensors on an exact clock, never wait for anything else ----
void sampleTask(void*) {
  const TickType_t period = pdMS_TO_TICKS(1000 / SAMPLE_RATE_HZ);
  TickType_t wake = xTaskGetTickCount();
  uint32_t seq = 0;
  for (;;) {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    Sample s;
    s.seq = seq++;                  // increases even when a reading is dropped -> server sees the gap
    s.ts = millis();
    s.ax = a.acceleration.x; s.ay = a.acceleration.y; s.az = a.acceleration.z;
    s.gx = g.gyro.x;         s.gy = g.gyro.y;         s.gz = g.gyro.z;
    for (int i = 0; i < 5; i++) {
      uint32_t sum = 0;
      for (int j = 0; j < HALL_OVERSAMPLE; j++) sum += analogRead(HALL_PINS[i]);
      s.h[i] = (uint16_t)(sum / HALL_OVERSAMPLE);
    }

    if (xQueueSend(sampleQueue, &s, 0) != pdTRUE) droppedSamples++;   // never block here
    vTaskDelayUntil(&wake, period);                                   // exact 10 ms spacing
  }
}

// ---- Wi-Fi helper: try to reconnect at most every 3 s ----
static bool wifiReady() {
  if (WiFi.status() == WL_CONNECTED) return true;
  static uint32_t lastTry = 0;
  if (millis() - lastTry > 3000) {
    lastTry = millis();
    WiFi.disconnect();
    WiFi.begin(WIFI_SSID, WIFI_PASS);
  }
  return false;
}

// ---- task 2: send batches. Slow network only delays DELIVERY, not sampling ----
void sendTask(void*) {
  static char body[BATCH_SIZE * 120 + 200];
  for (;;) {
    if (!wifiReady()) { vTaskDelay(pdMS_TO_TICKS(100)); continue; }   // readings wait in the queue

    int len = snprintf(body, sizeof(body),
                       "{\"dev\":\"%s\",\"hz\":%d,\"drop\":%lu,\"rows\":[",
                       DEVICE_ID, SAMPLE_RATE_HZ, (unsigned long)droppedSamples);
    for (int i = 0; i < BATCH_SIZE; i++) {
      Sample s;
      xQueueReceive(sampleQueue, &s, portMAX_DELAY);
      len += snprintf(body + len, sizeof(body) - len,
                      "%s[%lu,%lu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%u,%u,%u,%u,%u]",
                      i ? "," : "", (unsigned long)s.seq, (unsigned long)s.ts,
                      s.ax, s.ay, s.az, s.gx, s.gy, s.gz,
                      (unsigned)s.h[0], (unsigned)s.h[1], (unsigned)s.h[2],
                      (unsigned)s.h[3], (unsigned)s.h[4]);
    }
    snprintf(body + len, sizeof(body) - len, "]}");

    http.setReuse(true);          // keep the TCP connection open between batches
    http.setTimeout(800);         // never hang for the default 5 s
    if (http.begin(espClient, endpoint)) {
      http.addHeader("Content-Type", "application/json");
      int code = http.POST((uint8_t*)body, strlen(body));
      if (code != 200) postFailures++;
      http.end();
    } else {
      postFailures++;
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println("\nSignGlove booting...");
  snprintf(endpoint, sizeof(endpoint), "http://%s:8000/data-batch", SERVER_IP);

  Wire.begin();
  Wire.setClock(400000);
  if (!mpu.begin()) {
    Serial.println("MPU6050 not found - check wiring");
    while (1) delay(500);
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  analogReadResolution(12);

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);            // no radio naps -> no random 100+ ms delays
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  for (int i = 0; i < 30 && WiFi.status() != WL_CONNECTED; i++) { delay(500); Serial.print("."); }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("\nWi-Fi OK, glove IP: "); Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nWi-Fi not connected yet (will keep trying in the background)");
  }

  sampleQueue = xQueueCreate(QUEUE_LEN, sizeof(Sample));
  xTaskCreatePinnedToCore(sampleTask, "sample", 4096, NULL, 3, NULL, 1);  // core 1: sensors
  xTaskCreatePinnedToCore(sendTask,   "send",   8192, NULL, 1, NULL, 0);  // core 0: network
}

void loop() {
  // Status every 5 s. If "queue" keeps growing or "dropped" rises, the network is too slow.
  Serial.printf("queue=%u dropped=%lu postFail=%lu rssi=%d\n",
                (unsigned)uxQueueMessagesWaiting(sampleQueue),
                (unsigned long)droppedSamples, (unsigned long)postFailures, WiFi.RSSI());
  delay(5000);
}
