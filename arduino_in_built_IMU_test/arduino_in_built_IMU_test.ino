#include "Arduino_BMI270_BMM150.h"

// Tell the library to look on Wire1 for the shield's sensor
BoschSensorClass imu(Wire1);

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!imu.begin()) {
    Serial.println("Failed to initialize IMU on GIGA Display Shield!");
    while (1);
  }

  Serial.println("IMU initialized successfully!");
  Serial.print("Accelerometer sample rate = ");
  Serial.print(imu.accelerationSampleRate());
  Serial.println(" Hz");
}

void loop() {
  float x, y, z;

  // Check and read accelerometer data (in G's)
  if (imu.accelerationAvailable()) {
    imu.readAcceleration(x, y, z);
    
    Serial.print("Accel X: "); Serial.print(x);
    Serial.print("\tY: "); Serial.print(y);
    Serial.print("\tZ: "); Serial.println(z);
  }
}