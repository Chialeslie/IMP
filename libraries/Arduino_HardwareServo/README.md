# Arduino_HardwareServo

This library uses **hardware PWM** to control servo motors, enabling more precise and efficient signal generation compared to software-based approaches.

## 🔌 Compatibility
This library is currently available **for ArduinoCore-Zephyr boards only**.

## 🔩 Tested on following boards:

* Arduino UNO Q
* Arduino Nano BLE Sense

## ⚠️ Recommendations

Because it directly configures timer parameters (such as pulse period and duty cycle), the use of this library should not be mixed with direct `analogWrite` calls.