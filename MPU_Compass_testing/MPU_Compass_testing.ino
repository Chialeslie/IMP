#include <Wire.h>
#include <Adafruit_QMC5883P.h>

Adafruit_QMC5883P qmc = Adafruit_QMC5883P();
const int MPU_addr = 0x68;

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);
  
  delay(1000);
  Wire.begin();
  Wire.setClock(100000);
  delay(500);

  // Initialize QMC5883P Compass at Custom Address 0x2C
  if (!qmc.begin(0x2C, &Wire)) {
    Serial.println("QMC5883P not detected at 0x2C! Check wiring.");
    while (1) delay(10);
  }
  
  // Set mode to continuous measurement
  qmc.setMode(QMC5883P_MODE_CONTINUOUS);
  qmc.setRange(QMC5883P_RANGE_2G);

  // Wake up MPU6050
  Wire.beginTransmission(MPU_addr);
  Wire.write(0x6B); // PWR_MGMT_1
  Wire.write(0x00); // Wake up
  Wire.endTransmission(true);
  delay(100);

  Serial.println("All Sensors Initialized Successfully!");
}

void loop() {
  // --- 1. Read MPU6050 Pitch & Roll ---
  Wire.beginTransmission(MPU_addr);
  Wire.write(0x3B);
  byte writeErr = Wire.endTransmission(false);

  float roll = 0;
  float pitch = 0;

  if (writeErr == 0) {
    byte bytesReceived = Wire.requestFrom(MPU_addr, 6, true);
    if (bytesReceived == 6) {
      int16_t ax = Wire.read() << 8 | Wire.read();
      int16_t ay = Wire.read() << 8 | Wire.read();
      int16_t az = Wire.read() << 8 | Wire.read();

      roll  = atan2(-ay, -az) * 180.0 / PI;
      pitch = atan2(-ax, sqrt((long)ay * ay + (long)az * az)) * 180.0 / PI;
    }
  }

  // --- 2. Read QMC5883P Compass ---
  float mx, my, mz;
  int heading = 0;
  
  if (qmc.getGaussField(&mx, &my, &mz)) {
    float calcHeading = atan2(my, mx) * 180.0 / PI;
    if (calcHeading < 0) {
      calcHeading += 360.0;
    }
    heading = (int)calcHeading;
  }

  // Output Combined Data
  Serial.print("Roll: "); Serial.print(roll, 2);
  Serial.print(" | Pitch: "); Serial.print(pitch, 2);
  Serial.print(" | Compass Heading: "); Serial.println(heading);

  delay(200);
}