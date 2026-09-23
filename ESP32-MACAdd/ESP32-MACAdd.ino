#include "WiFi.h"

void setup() {
  Serial.begin(115200);
  delay(100);

  // 1. Initialize the specific interfaces so their drivers wake up
  WiFi.STA.begin();
  WiFi.AP.begin();

  // 2. Now print the addresses safely
  Serial.print("MAC Address: ");
  Serial.println(WiFi.macAddress());
  
  Serial.print("SoftAP MAC address: ");
  Serial.println(WiFi.softAPmacAddress());
}

void loop() {
  // Main loop
}
