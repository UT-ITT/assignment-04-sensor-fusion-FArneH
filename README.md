[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/AktWbCri)
# assignment-04-CV-Sensor-Fusion

## AR Bubble Catcher

Ein AR-Spiel das die Webcam und ein Board mit 4 ArUco-Markern (DICT_6X6_250) in den Ecken verwendet.

Rote Bälle fliegen von rechts nach links über das Board. Mit dem Zeigefinger kann man die Bälle berühren und zum Platzen bringen, bevor sie die linke Seite erreichen. Der grüne Punkt zeigt wo der Finger erkannt wird.

**Starten:**
```
python ar_game/AR_game.py
```
Kamera auf das Board richten bis die Bälle erscheinen. Nur den Zeigefinger ausstrecken für das beste Tracking-Ergebnis.

## Sensor Fusion

ArUco-Marker (ID 23, angezeigt von der DIPPID-App) wird im gewarpten Board getrackt (roter Punkt). Per DIPPID werden Beschleunigungsdaten empfangen und mit einem Komplementärfilter mit der Kameraposition fusioniert (grüner Punkt). Alpha mit Pfeiltasten anpassen, Button 1 setzt die Vorhersage zurück.

**Starten:**
```
python sensor_fusion/sensor_fusion.py
```

## Reflexion Sensor Fusion

Der Komplementärfilter kombiniert die Kameraposition (absolut, aber verzögert) mit dem Beschleunigungssensor (schnell, aber driftet). Mit `alpha = 1.0` folgt der grüne Punkt exakt dem roten – der Beschleunigungssensor hat keinen Einfluss. Mit `alpha = 0.0` wird nur der Beschleunigungssensor integriert, was zunächst flüssig wirkt, aber schnell abdriftet weil sich Messfehler aufaddieren. Bei mittleren Werten (z.B. `alpha = 0.5`) ist ein guter Kompromiss: die Kamera korrigiert regelmäßig den Drift und der Beschleunigungssensor macht die Bewegung zwischen den Frames flüssiger.