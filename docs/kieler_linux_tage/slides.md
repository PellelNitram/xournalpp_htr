---
theme: default
title: Open Source ML Features und ihre Herausforderungen
info: |
  24. Kieler Linux Tage 2026
  Martin Lellep — lellep.xyz
author: Martin Lellep
keywords: open-source,machine-learning,linux,handschrifterkennung
exportFilename: open-source-ml-features
transition: slide-left
---

# Open Source ML Features und ihre Herausforderungen

24\. Kieler Linux Tage 2026

<br>

**Martin Lellep**

[lellep.xyz](https://lellep.xyz) · [LinkedIn](https://www.linkedin.com/in/martin-lellep-858600152/) · [Programm](https://www.kieler-linuxtage.de/p/KOLT26#840)

---

# TODOs

- Story Line
- Slide Wording
- Füge vorherige Gedanken hinzu
- Liste max N Challenges und Learnings, offloade Rest in Appendix
- Füge YT-Channel hinzu
- Beautify Design

---

# Was ihr mitnehmen werdet

- Wie ein ML-Feature von der Idee bis ins fertige Produkt entsteht
- Welche Hürden euch erwarten — und wie ihr sie umgeht
- Praktische Tipps, um selbst loszulegen

---

# Agenda

1. Vorstellung & Motivation
2. Was ist Machine Learning? — Kurzer Überblick
3. Fallstudie: Handschrifterkennung für digitale Notizen
4. Herausforderungen bei der Entwicklung
5. Lernerfahrungen, Tipps & Tricks
6. Fazit & Ausblick

---

# Vorstellung

- Martin Lellep
- Seit vier Jahren: Open-Source-Feature zur Handschrifterkennung
- Für digital geschriebene Notizen
- TODO: Weiterer Hintergrund

---

# Was ist Machine Learning?

- Software, die aus Beispielen lernt statt fest programmiert zu werden
- TODO: Einfache Illustration / Analogie

---

# ML-Features begegnen euch überall

- 📷 **Google Photos / Immich** — Gesichtserkennung zum automatischen Sortieren von Fotos
- ✉️ **Thunderbird** — Spam-Erkennung filtert unerwünschte E-Mails
- 🔧 **GIMP** _(hypothetisch)_ — Automatisches Freistellen von Objekten per Klick

<!--
Anfängerfreundlich halten — keine Formeln, kein Jargon.
Ziel: Das Publikum soll verstehen, warum man Daten braucht und was "Training" bedeutet.
-->

---

# Warum ML in Open Source?

- ML-Features machen Software deutlich nützlicher
  - Beispiel: Handschrift in Text umwandeln, Autovervollständigung, Bildbearbeitung
- Bisher vor allem in kommerzieller Software zu finden
- **Warum?** Hoher Aufwand und Trainingsdaten schwer zu beschaffen
- **Ziel:** Diese Features auch in Open Source ermöglichen

---

# Fallstudie: Handschrifterkennung

- TODO: Projekt vorstellen (Xournal++, HTR-Plugin)
- TODO: Demo / Screenshots — Was kann das Feature?
- TODO: Einfacher Architektur-Überblick (Eingabe → Modell → Ausgabe)

---

# Herausforderung 1: Woher kommen die Daten?

- Ein ML-Modell braucht viele Beispiele zum Lernen
- Bei Open Source: Woher bekommt man diese Daten?
- Lizenzfragen — darf man die Daten überhaupt nutzen?
- Qualität: Nicht alle Daten sind gleich gut
- Benchmark-Datasets: Wie misst man, ob das Modell gut genug ist?
  - Für Xournal++ existiert noch kein öffentliches Benchmark-Dataset
- TODO: Konkrete Erfahrungen

---

# Herausforderung 2: Training braucht Rechenpower

- ML-Modelle zu trainieren dauert lange und braucht spezielle Hardware (GPUs)
- Als Einzelperson oder kleines Team: Woher nehmen?
- Ergebnisse müssen nachvollziehbar sein (Reproduzierbarkeit)
- TODO: Konkrete Erfahrungen

---

# Herausforderung 3: Vom Modell zum fertigen Feature

- Das trainierte Modell muss in die Software integriert werden
- Modelle können groß sein — wie liefert man sie aus?
- Muss auf verschiedenen Betriebssystemen laufen
- TODO: Konkrete Erfahrungen

---

# Herausforderung 4: Community & langfristige Pflege

- Wer reviewt ML-Code in einem Open-Source-Projekt?
- Erwartungen der Nutzer vs. was realistisch möglich ist
- Langfristige Wartung und Weiterentwicklung
- TODO: Konkrete Erfahrungen

---

# Lernerfahrung: Online-Demo bauen

- Auch wenn das Feature komplett offline läuft — baut eine Online-Demo
- Senkt die Hürde, es auszuprobieren, enorm
- Kein Installieren, kein Einrichten — einfach im Browser testen
- TODO: Konkretes Beispiel / Link zur Demo

---

# Lernerfahrung: Video-Content erstellen

- Videos erreichen ein viel breiteres Publikum als Blogposts oder READMEs
- Zeigt das Feature in Aktion — Leute verstehen sofort, was es kann
- Hilft bei Community-Building und Feedback
- TODO: Konkretes Beispiel / Erfahrungen

---

# Tipps & Tricks

- TODO: Praktische Empfehlungen für alle, die eigene ML-Features entwickeln wollen

---

# Fazit & Ausblick

- ML-Features in Open Source sind machbar — der Weg sieht aber anders aus als bei kommerziellen Produkten
- TODO: Zusammenfassung der wichtigsten Punkte
- TODO: Nächste Schritte

---
layout: end
---

# Danke! — Und jetzt seid ihr dran

<br>

🚀 **Baut ML-Features für eure Open-Source-Projekte!**

✍️ **Ihr nutzt Xournal++?** Schickt mir eure handgeschriebenen Notizen — sie helfen beim Aufbau eines offenen Benchmark-Datasets.

<br>

Martin Lellep — [lellep.xyz](https://lellep.xyz) · [LinkedIn](https://www.linkedin.com/in/martin-lellep-858600152/)

Fragen?
