#include <WiFi.h>
#include <Wire.h>
#include <SPI.h>
#include <RadioLib.h>
#include "mbedtls/aes.h"

// -------------------------------------------------------------
// DX-LR30-900M22S (SX1262) LORA PINOUT
// -------------------------------------------------------------
#define LORA_CS   5
#define LORA_DIO1 27
#define LORA_RST  14
#define LORA_BUSY 25

SX1262 radio = new Module(LORA_CS, LORA_DIO1, LORA_RST, LORA_BUSY);

// Joystick & Switch Pins
const int JSX = 36;          
const int JSY = 39;          
const int JSZ = 34; 
const int MODE_SWITCH = 26; 

// Packet Type Identifiers (Must match Slave)
#define PKT_HANDSHAKE_REQ  0xA1
#define PKT_HANDSHAKE_ACK  0xA2
#define PKT_DATA_CONTROL   0xB1
#define PKT_TELEMETRY      0xB2

// AES-128 ENCRYPTION KEY FOR LORA
const uint8_t aesKey[16] = { 'G', 'Y', 'e', 'o', 'k', 'k', 'L', 'e', 'i', 'a', 'W', 'n', 'a', 'g', 'n', 0 };

// Connection States for Formal Handshake
enum ConnectionState {
  STATE_DISCONNECTED,
  STATE_HANDSHAKING,
  STATE_CONNECTED
};
ConnectionState linkState = STATE_DISCONNECTED;

unsigned long lastHandshakeAttempt = 0;
const unsigned long HANDSHAKE_INTERVAL_MS = 1000; // Retry every 1s when disconnected

// Compact Control Packet to fit safely inside 16-byte AES block with 1-byte header
typedef struct __attribute__((packed)) lora_control_packet {
  int16_t x;       // 2 bytes
  int16_t y;       // 2 bytes
  int8_t mode;     // 1 byte
  uint32_t counter;// 4 bytes
} lora_control_packet;

// Optimized GPS Struct matching Slave layout (12 bytes total)
typedef struct __attribute__((packed)) struct_gps {
  float latitude;      // 4 bytes
  float longitude;     // 4 bytes
  uint8_t satellites;  // 1 byte
  bool fix;            // 1 byte
  int16_t heading;     // 2 bytes
} struct_gps;

lora_control_packet controlData = {0, 0, 10, 0};
struct_gps incomingGPS = {0.0, 0.0, 0, false, 0};

unsigned long lastLoRaRx = 0;
const unsigned long LORA_TIMEOUT_MS = 5000; 
volatile bool masterRxFlag = false;

void setMasterFlag(void) {
  masterRxFlag = true;
}
// -------------------------------------------------------------
// AES ENCRYPTION HELPERS
// -------------------------------------------------------------
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

  radio.transmit(cipherText, 16);
  
  delayMicroseconds(2000); // Increased to 2ms to give SX1262 enough turnaround time
  radio.getPacketLength(); // Clear stale local flags
  radio.startReceive();    // Open listening window safely
}

void decryptLoRaPayload(uint8_t *cipherText, uint8_t *outputData, uint16_t outputSize) {
  uint8_t decrypted[16] = {0};

  mbedtls_aes_context aes;
  mbedtls_aes_init(&aes);
  mbedtls_aes_setkey_dec(&aes, aesKey, 128);
  mbedtls_aes_crypt_ecb(&aes, MBEDTLS_AES_DECRYPT, cipherText, decrypted);
  mbedtls_aes_free(&aes);

  memcpy(outputData, decrypted, outputSize);
}

// -------------------------------------------------------------
// SETUP SECTION
// -------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  analogReadResolution(10);
  pinMode(JSX, INPUT);
  pinMode(JSY, INPUT);
  pinMode(JSZ, INPUT_PULLUP);
  pinMode(MODE_SWITCH, INPUT_PULLUP);

  pinMode(LORA_CS, OUTPUT);
  digitalWrite(LORA_CS, HIGH);

  SPI.begin(18, 19, 23, 5); 

  Serial.println(F("Init LoRa Master with Handshake Protocol..."));

  int state = radio.begin(923.0, 125.0, 7, 5, RADIOLIB_SX126X_SYNC_WORD_PRIVATE, 14);
  radio.setPacketReceivedAction(setMasterFlag);
  radio.startReceive();
  if (state == RADIOLIB_ERR_NONE) {
    Serial.println(F("LoRa Master Ready!"));
    radio.startReceive();
  } else {
    Serial.printf("LoRa Radio Init Fail, code %d\n", state);
  }
}

// -------------------------------------------------------------
// MAIN LOOP
// -------------------------------------------------------------
void loop() {
  // 1. Check for incoming LoRa packets (Interrupt-driven)
  if (masterRxFlag) {
    masterRxFlag = false;
    
    uint8_t rxBuffer[16] = {0};
    int state = radio.readData(rxBuffer, 16);
    
    if (state == RADIOLIB_ERR_NONE) {
      uint8_t decryptedPayload[16] = {0};
      decryptLoRaPayload(rxBuffer, decryptedPayload, 16);

      uint8_t pktType = decryptedPayload[0];

      // Ignore local self-echoes
      if (pktType == PKT_HANDSHAKE_REQ || pktType == PKT_DATA_CONTROL) {
        // Do nothing, drop self-echoes
      }
      // Handshake ACK received from Robot
      else if (pktType == PKT_HANDSHAKE_ACK) {
        linkState = STATE_CONNECTED;
        lastLoRaRx = millis();
        Serial.println(F("[Status] HANDSHAKE SUCCESSFUL: LORA LINK ACTIVE"));
      } 
      // Regular Telemetry received from Robot
      else if (pktType == PKT_TELEMETRY && linkState == STATE_CONNECTED) {
        memcpy(&incomingGPS, &decryptedPayload[1], sizeof(struct_gps));
        lastLoRaRx = millis();
        
        Serial.printf("[LoRa RX] Telemetry -> Lat: %.5f, Lon: %.5f, Sats: %d, Hdg: %d\n",
                      incomingGPS.latitude, incomingGPS.longitude, incomingGPS.satellites, incomingGPS.heading);
      } else {
        Serial.printf("[LoRa RX] Unknown Packet Type: 0x%02X\n", pktType);
      }
    } else {
      Serial.printf("[Master Radio Read Error]: %d\n", state);
    }
    radio.startReceive();
  }

  // 2. State Machine Handling
  switch (linkState) {
    case STATE_DISCONNECTED:
    case STATE_HANDSHAKING:
      // Periodically send a handshake request packet until acknowledged
      if (millis() - lastHandshakeAttempt > HANDSHAKE_INTERVAL_MS) {
        lastHandshakeAttempt = millis();
        Serial.println(F("[LoRa] Sending Handshake Request (SYN)..."));
        uint8_t dummy = 0;
        encryptAndSendLoRa(PKT_HANDSHAKE_REQ, &dummy, 1);
      }
      break;

    case STATE_CONNECTED:
      // Check connection timeout watchdog
      if (millis() - lastLoRaRx > LORA_TIMEOUT_MS) {
        linkState = STATE_DISCONNECTED;
        Serial.println(F("[Status] LORA LINK LOST: Connection Timeout. Returning to Handshake Mode."));
        break;
      }

      // Read Joystick and send regular control data
      controlData.x = analogRead(JSX);
      controlData.y = analogRead(JSY);
      controlData.mode = (digitalRead(MODE_SWITCH) == LOW) ? 10 : 11;

      static unsigned long lastTxTime = 0;
      if (millis() - lastTxTime >= 250) {
        lastTxTime = millis();
        controlData.counter++;
        encryptAndSendLoRa(PKT_DATA_CONTROL, (uint8_t*)&controlData, sizeof(lora_control_packet));
      }
      break;
  }

  vTaskDelay(10 / portTICK_PERIOD_MS);
}