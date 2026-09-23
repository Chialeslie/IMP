#include <Wire.h>

const int COMPASS_ADDR = 0x2C; // QMC5883P I2C address
const int MPU_addr = 0x68;     // MPU6050 I2C address

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);
  
  Wire.begin();
  Wire.setClock(100000);
  delay(500);

  // --- QMC5883P EXACT INITIALIZATION SEQUENCE ---
  // The 'P' variant requires this specific boot-up routine
  Wire.beginTransmission(COMPASS_ADDR);
  Wire.write(0x29); 
  Wire.write(0x06); 
  Wire.endTransmission();
  delay(10);

  Wire.beginTransmission(COMPASS_ADDR);
  Wire.write(0x0B); 
  Wire.write(0x08); 
  Wire.endTransmission();
  delay(10);

  Wire.beginTransmission(COMPASS_ADDR);
  Wire.write(0x0A); 
  Wire.write(0xCD); // Continuous measurement mode
  Wire.endTransmission();
  delay(50);

  // --- Wake up MPU6050 ---
  Wire.beginTransmission(MPU_addr);
  Wire.write(0x6B); 
  Wire.write(0x00); 
  Wire.endTransmission(true);
  delay(100);

  Serial.println("QMC5883P & MPU6050 Online!");
}

void loop() {
  // --- Read MPU6050 Roll & Pitch ---
  Wire.beginTransmission(MPU_addr);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_addr, 6, true);
  
  float roll = 0, pitch = 0;
  if (Wire.available() >= 6) {
    int16_t ax = Wire.read() << 8 | Wire.read();
    int16_t ay = Wire.read() << 8 | Wire.read();
    int16_t az = Wire.read() << 8 | Wire.read();
    roll  = atan2(-ay, -az) * 180.0 / PI;
    pitch = atan2(-ax, sqrt((long)ay * ay + (long)az * az)) * 180.0 / PI;
  }

  // --- Read QMC5883P Data Registers ---
  // CRITICAL FIX: Data starts at 0x01 on the 'P' variant, NOT 0x00!
  Wire.beginTransmission(COMPASS_ADDR);
  Wire.write(0x01); 
  Wire.endTransmission(false);
  
  byte bytesRequested = Wire.requestFrom(COMPASS_ADDR, 6, true);
  
  float mx = 0, my = 0;
  int heading = 0;

  if (bytesRequested >= 6) {
    int16_t x = (int16_t)(Wire.read() | (Wire.read() << 8));
    int16_t y = (int16_t)(Wire.read() | (Wire.read() << 8));
    int16_t z = (int16_t)(Wire.read() | (Wire.read() << 8));

    mx = (float)x;
    my = (float)y;

    if (mx != 0 || my != 0) {
      float calcHeading = atan2(my, mx) * 180.0 / PI;
      if (calcHeading < 0) {
        calcHeading += 360.0;
      }
      heading = (int)calcHeading;
    }
  }

  // Output Combined Data
  Serial.print("Roll: "); Serial.print(roll, 2);
  Serial.print(" | Pitch: "); Serial.print(pitch, 2);
  Serial.print(" | MX: "); Serial.print(mx, 1);
  Serial.print(" MY: "); Serial.print(my, 1);
  Serial.print(" | Compass Heading: "); Serial.println(heading);

  delay(250);
}