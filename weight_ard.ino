#include "HX711.h"

const int DT_PIN = 4;
const int SCK_PIN = 5;

const float CALIBRATION_FACTOR = 444.3;
const int TARE_SAMPLES = 20;
const float ZERO_CLAMP_GRAMS = 1.0;

HX711 scale;
long zero_offset = 0;

void tare_scale() {
  Serial.print("Taring with ");
  Serial.print(TARE_SAMPLES);
  Serial.println(" samples. Keep the scale empty.");

  zero_offset = scale.read_average(TARE_SAMPLES);

  Serial.print("Zero offset: ");
  Serial.println(zero_offset);
}

void setup() {
  Serial.begin(9600);
  scale.begin(DT_PIN, SCK_PIN);
  delay(500);
  tare_scale();
  Serial.println("System Ready");
}

void loop() {
  if (scale.is_ready()) {
    long reading = scale.read();
    Serial.print("Raw: ");
    Serial.println(reading);


    float grams = (reading - zero_offset) / CALIBRATION_FACTOR;
    if (fabs(grams) <= ZERO_CLAMP_GRAMS) {
      grams = 0.0;
    }

    Serial.print("Weight: ");
    Serial.print(grams, 2);
    Serial.println(" g");
  }
  delay(500);
}
