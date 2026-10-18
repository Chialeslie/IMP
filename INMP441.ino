#include <driver/i2s.h>

const int I2S_SCK = 4;
const int I2S_WS = 5;
const int I2S_SD = 6;

#define I2S_PORT I2S_NUM_0
const int sampleRate = 16000;

#define AUDIO_BUFFER_SIZE 256
int16_t audio_chunk[AUDIO_BUFFER_SIZE];
size_t chunk_index = 0;

void setup() {
  Serial.begin(921600);
  delay(1000);
  Serial.println("[INFO] ESP32C3 Voice Module Initializing ...");

  // configure I2S peripheral
  const i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = sampleRate,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 64,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };

  // configure I2S pins
  const i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD
  };

  // install and start I2S driver
  if (i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL) != ESP_OK) {
    Serial.println("[ERROR] Failed to install I2S driver!");
    while (1);
  }

  if (i2s_set_pin(I2S_PORT, &pin_config) != ESP_OK) {
    Serial.println("[ERROR] Failed to set I2S pin configuration!");
    while(1);
  }

  i2s_start(I2S_PORT);
  Serial.println("[INFO] INMP 441 I2S Microphone Ready and Listening.");
}

void loop() {
  int32_t i2s_buffer[64];
  size_t bytes_read = 0;

  // Read 32-bit audio data block from I2S
  esp_err_t result = i2s_read(I2S_PORT, &i2s_buffer, sizeof(i2s_buffer), &bytes_read, portMAX_DELAY);

  if (result == ESP_OK && bytes_read > 0) {
    int samples = bytes_read / sizeof(int32_t);
    int16_t pcm_buffer[64];
    for (int i = 0; i < samples; i++) {
      pcm_buffer[i] = (int16_t)(i2s_buffer[i] >> 15);
    }
    Serial.write((uint8_t*)pcm_buffer, samples * sizeof(int16_t));
  }
}
