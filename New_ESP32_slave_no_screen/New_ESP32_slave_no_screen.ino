#include <WiFi.h>
#include <Wire.h>
#include <SPI.h>
#include <RadioLib.h>
#include "mbedtls/aes.h"

// -------------------------------------------------------------
// DX-LR30-900M22S (SX1262) LORA PINOUT
// -------------------------------------------------------------
#define LORA_CS    5
#define LORA_DIO1 27
#define LORA_RST  14
#define LORA_BUSY 25

SX1262 radio = new Module(LORA_CS, LORA_DIO1, LORA_RST, LORA_BUSY);

// UART Communication Markers (Unified)
#define START_MARKER 0x3C
#define STOP_MARKER  0x3E

// Packet Type Identifiers (Must match Master)
#define PKT_HANDSHAKE_REQ  0xA1
#define PKT_HANDSHAKE_ACK  0xA2
#define PKT_DATA_CONTROL   0xB1
#define PKT_TELEMETRY      0xB2

// AES-128 ENCRYPTION KEY (Must match Master exactly)
const uint8_t aesKey[16] = { 'G', 'Y', 'e', 'o', 'k', 'k', 'L', 'e', 'i', 'a', 'W', 'n', 'a', 'g', 'n', 0 }; 

// Standard GIGA Control Struct (16 bytes sent over Serial2)
typedef struct __attribute__((packed)) struct_control {
  int32_t x;
  int32_t y;
  int32_t mode;
  int32_t counter;
} struct_control;

// Compact LoRa Control Packet to fit with 1-byte header inside 16-byte AES block
typedef struct __attribute__((packed)) lora_control_packet {
  int16_t x;       // 2 bytes
  int16_t y;       // 2 bytes
  int8_t mode;     // 1 byte
  uint32_t counter;// 4 bytes
} lora_control_packet;

// Optimized GPS Struct to fit with 1-byte header inside 16-byte AES block (Total data = 12 bytes + 1 byte header = 13 bytes)
typedef struct __attribute__((packed)) struct_gps {
  float latitude;      // 4 bytes
  float longitude;     // 4 bytes
  uint8_t satellites;  // 1 byte
  bool fix;            // 1 byte
  int16_t heading;     // 2 bytes
} struct_gps;

struct_control incomingControl;
struct_gps telemetryData = {0.0, 0.0, 0, false, 0};

unsigned long lastLoRaRx = 0;
const unsigned long LORA_TIMEOUT_MS = 3500;
bool isConnected = false;

volatile bool receivedFlag = false;

void setFlag(void) {
  receivedFlag = true;
}
// -------------------------------------------------------------
// AES CRYPTOGRAPHY HELPERS
// -------------------------------------------------------------
void decryptLoRaPayload(uint8_t *cipherText, uint8_t *outputData, uint16_t outputSize) {
  uint8_t decrypted[16] = {0};

  mbedtls_aes_context aes;
  mbedtls_aes_init(&aes);
  
  int keyRet  = mbedtls_aes_setkey_dec(&aes, aesKey, 128);
  int cryptRet = mbedtls_aes_crypt_ecb(&aes, MBEDTLS_AES_DECRYPT, cipherText, decrypted);
  
  mbedtls_aes_free(&aes);

  memcpy(outputData, decrypted, outputSize);
}

void encryptAndSendLoRa(uint8_t pktType, uint8_t *data, uint16_t size) {
  uint8_t inputBuffer[16] = {0};
  inputBuffer[0] = pktType; 
  memcpy(&inputBuffer[1], data, min((size_t)size, (size_t)15));

  uint8_t cipherText[16] = {0};
  mbedtls_aes_context aes;
  mbedtls_aes_init(&aes);
  mbedtls_aes_setkey_enc(&aes, (const unsigned char*)aesKey, 128);
  mbedtls_aes_crypt_ecb(&aes, MBEDTLS_AES_ENCRYPT, inputBuffer, cipherText);
  mbedtls_aes_free(&aes);

  // Transmit the 16-byte encrypted block over LoRa
  int state = radio.transmit(cipherText, 16);
  
  if (state != RADIOLIB_ERR_NONE) {
    Serial.printf("[LoRa TX Error] Code: %d\n", state);
  }

  delayMicroseconds(1000); // Crucial RX/TX turnaround stabilization
  radio.getPacketLength();  // Clear any local buffer flags
  radio.startReceive();     // CRITICAL: Put radio back into listening mode!
}

// -------------------------------------------------------------
// SETUP SECTION
// -------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  Serial2.begin(115200, SERIAL_8N1, 16, 17); // Connected to Arduino GIGA RX/TX
    
  pinMode(LORA_CS, OUTPUT);
  digitalWrite(LORA_CS, HIGH);

  SPI.begin(18, 19, 23, 5); 

  Serial.println(F("Init LoRa Slave with Handshake Support..."));

  int state = radio.begin(923.0, 125.0, 7, 5, RADIOLIB_SX126X_SYNC_WORD_PRIVATE, 14);

  if (state == RADIOLIB_ERR_NONE) {
    Serial.println(F("LoRa Slave Ready!"));
  } else {
    Serial.printf("LoRa Radio Init Fail, code %d\n", state);
    while (true);
  }

  radio.setPacketReceivedAction(setFlag);
  radio.startReceive();
  
  Serial.println(F("[INFO] Slave Synchronized Firmware Loaded!"));
  delay(1000);
}

// -------------------------------------------------------------
// MAIN LOOP
// -------------------------------------------------------------
void loop() {
  // 1. Read incoming telemetry stream from Arduino GIGA via Serial2
  static uint8_t gpsBuffer[16];
  static int gpsIndex = 0;
  static bool receivingGPS = false;

  while (Serial2.available() > 0) {
    uint8_t incomingByte = Serial2.read();
    if (!receivingGPS) {
      if (incomingByte == START_MARKER) {
        receivingGPS = true;
        gpsIndex = 0;
      }
    } else {
      if (incomingByte == STOP_MARKER) {
        if (gpsIndex == sizeof(struct_gps)) {
          memcpy(&telemetryData, gpsBuffer, sizeof(struct_gps));
          
          Serial.printf("[GPS RX] Sats: %d | Hdg: %d | Lat: %.6f | Lon: %.6f\n", 
                        telemetryData.satellites, telemetryData.heading, telemetryData.latitude, telemetryData.longitude);
        } else {
          Serial.printf("[GPS Error] Size mismatch: %d\n", gpsIndex);
        }
        receivingGPS = false;
      } else if (gpsIndex < sizeof(gpsBuffer)) {
        gpsBuffer[gpsIndex++] = incomingByte;
      } else {
        receivingGPS = false; 
      }
    }
  }

  // 2. Process Incoming Encrypted Packets from Master
  if (receivedFlag) {
    receivedFlag = false; 
    
    uint8_t rxBuffer[16] = {0};
    int state = radio.readData(rxBuffer, 16);
    
    if (state == RADIOLIB_ERR_NONE) {
      uint8_t decryptedRaw[16] = {0};
      decryptLoRaPayload(rxBuffer, decryptedRaw, 16);

      uint8_t pktType = decryptedRaw[0]; 

      if (pktType == PKT_HANDSHAKE_REQ) {
        Serial.println(F("[LoRa] Handshake SYN Received. Sending ACK..."));
        uint8_t dummy = 0;
        encryptAndSendLoRa(PKT_HANDSHAKE_ACK, &dummy, 1);
      }
      else if (pktType == PKT_DATA_CONTROL) {
        lora_control_packet loraCtrl;
        memcpy(&loraCtrl, &decryptedRaw[1], sizeof(lora_control_packet));

        // Map compact packet up to full 32-bit struct for the GIGA
        incomingControl.x = loraCtrl.x;
        incomingControl.y = loraCtrl.y;
        incomingControl.mode = loraCtrl.mode;
        incomingControl.counter = loraCtrl.counter;
        
        lastLoRaRx = millis(); 
        
        Serial.printf("[LoRa Control RX] Mode: %ld | X: %ld | Y: %ld\n", 
                      (long)incomingControl.mode, (long)incomingControl.x, (long)incomingControl.y);

        // Forward control command to Arduino GIGA
        Serial2.write(START_MARKER);
        Serial2.write((uint8_t*)&incomingControl, sizeof(struct_control));
        Serial2.write(STOP_MARKER);
        
        delay(5);
        
        // Reply with live telemetry
        encryptAndSendLoRa(PKT_TELEMETRY, (uint8_t*)&telemetryData, sizeof(struct_gps));
      }
    } else {
      Serial.printf("[Radio Read Error]: %d\n", state);
      radio.startReceive(); // Ensure we resume listening on error
    }
  }

  // 3. Connection watchdog tracking
  isConnected = (lastLoRaRx != 0) && (millis() - lastLoRaRx <= LORA_TIMEOUT_MS);
  if (!isConnected) {
    incomingControl.x = 512;  
    incomingControl.y = 512;
    incomingControl.mode = 11; // Default back to autonomous/safe mode on link loss
  }

  vTaskDelay(10 / portTICK_PERIOD_MS);
}