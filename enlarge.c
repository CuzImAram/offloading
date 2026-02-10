#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <omp.h>
#include "image.h"

// Macros
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define ABS(x) (((x) < 0) ? -(x) : (x))
#define CLAMP(val, min, max) ((val) < (min) ? (min) : ((val) > (max) ? (max) : (val)))

// Pre-compute Entropy LUT (CPU)
// Einmalige Berechnung auf CPU, da Tabelle sehr klein. Transfer lohnt nicht.
int *create_entropy_lut()
{
    int *lut = malloc(sizeof(int) * 82);
    lut[0] = 0;
    for (int i = 1; i <= 81; ++i)
    {
        double p = (double)i / 81.0;
        lut[i] = (int)(-1.0 * log2(p) * p * (CHAR_MAX / 5.0 * 3.2));
    }
    return lut;
}

// Comparison for qsort
int compare_ints(const void *a, const void *b)
{
    int arg1 = *(const int *)a;
    int arg2 = *(const int *)b;
    return (arg1 > arg2) - (arg1 < arg2);
}

// Step 1: Integer Grayscale Conversion
unsigned char *gray(struct imgRawImage *image)
{
    unsigned int width = image->width;
    unsigned int height = image->height;
    unsigned char *output = malloc(sizeof(unsigned char) * 3 * width * height);
    unsigned char *pixels = image->lpData;

// Strategie: Massive Parallelität.
// 'collapse(2)' verschmilzt Schleifen für maximale Auslastung der GPU-Kerne.
// Daten werden explizit gemappt (Pixel -> GPU, Output -> Host).
// Beispiel: rgb = (100,150,200) -> (100*0.299 + 150*0.587 + 200*0.114) = 141 für alle Pixelkanäle
#pragma omp target teams distribute parallel for collapse(2) map(to : pixels[0 : 3 * width * height]) map(from : output[0 : 3 * width * height])
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int idx = (y * width + x) * 3;
            unsigned char r = pixels[idx + 0];
            unsigned char g = pixels[idx + 1];
            unsigned char b = pixels[idx + 2];

            unsigned char luma = (r * 299 + g * 587 + b * 114) / 1000;

            output[idx + 0] = luma;
            output[idx + 1] = luma;
            output[idx + 2] = luma;
        }
    }
    return output;
}

// Step 3.1: blue map
unsigned int *calculateMinEnergySums(unsigned int *data, int width, int height)
{
    unsigned int *output = malloc(sizeof(unsigned int) * width * height);

// Initialisierung der ersten Zeile parallel auf GPU
#pragma omp target teams distribute parallel for map(to : data[0 : width * height]) map(from : output[0 : width * height])
    for (int x = 0; x < width; ++x)
        output[x] = data[x];

// Strategie: Sequentielle Zeilenabhängigkeit (Dynamic Programming).
// 'target data': Hält 'output' und 'data' persistent im GPU-Speicher.
// Verhindert teures Kopieren zwischen den Iterationen der äußeren Schleife.
#pragma omp target data map(to : data[0 : width * height]) map(tofrom : output[0 : width * height])
    {
        // Äußere Schleife (y) läuft auf CPU zur Steuerung (wegen Abhängigkeit y-1).
        for (int y = 1; y < height; ++y)
        {
// Innere Schleife (x) wird pro Zeile auf GPU offloaded.
#pragma omp target teams distribute parallel for
            for (int x = 0; x < width; ++x)
            {
                unsigned int min_val;
                int idx_prev = (y - 1) * width + x;

                // Fall 1: Rechter Bildrand. Es gibt keinen Nachbarn oben-rechts.
                if (x == width - 1)
                {
                    min_val = MIN(output[idx_prev - 1], output[idx_prev]);
                }

                // Fall 2: Linker Bildrand. Es gibt keinen Nachbarn oben-links.
                else if (x == 0)
                {
                    min_val = MIN(output[idx_prev], output[idx_prev + 1]);
                }

                // Fall 3: Standardfall (Pixel in der Mitte). Alle drei oberen Nachbarn existieren.
                else
                {
                    min_val = MIN(MIN(output[idx_prev - 1], output[idx_prev]), output[idx_prev + 1]);
                }

                // Die kumulierte Energie dieses Pixels ist seine eigene Energie (data)
                // plus der kleinste Pfadwert von oben (min_val).
                output[y * width + x] = data[y * width + x] + min_val;
            }
        }
    }
    return output;
}

// Step 2: red map
unsigned int *calculateEnergySobel(struct imgRawImage *image, int *entropy_lut)
{
    unsigned int width = image->width;
    unsigned int height = image->height;
    unsigned int *output = malloc(sizeof(unsigned int) * height * width);
    unsigned char *pixels = image->lpData;

// Strategie: Unabhängige Pixel-Operation.
// LUT wird auf GPU gemappt für schnellen Zugriff.
// 'collapse(2)' für maximale Thread-Anzahl.
#pragma omp target teams distribute parallel for collapse(2)      \
    map(to : pixels[0 : 3 * width * height], entropy_lut[0 : 82]) \
    map(from : output[0 : width * height])
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            // GRADIENT e1
            int gx, gy, e_1;
            // Randbehandlung: Pixel außerhalb des Bildes werden auf den Rand projiziert (CLAMP).
            int y_m1 = CLAMP(y - 1, 0, height - 1);
            int y_p1 = CLAMP(y + 1, 0, height - 1);
            int x_m1 = CLAMP(x - 1, 0, width - 1);
            int x_p1 = CLAMP(x + 1, 0, width - 1);

            // Gx-Kernel: Erkennt vertikale Kanten (Änderung in horizontaler Richtung).
            gx = -1 * pixels[(y_m1 * width + x_m1) * 3] + 1 * pixels[(y_m1 * width + x_p1) * 3] -
                 2 * pixels[(y * width + x_m1) * 3] + 2 * pixels[(y * width + x_p1) * 3] -
                 1 * pixels[(y_p1 * width + x_m1) * 3] + 1 * pixels[(y_p1 * width + x_p1) * 3];

            // Gy-Kernel: Erkennt horizontale Kanten (Änderung in vertikaler Richtung).
            gy = -1 * pixels[(y_m1 * width + x_m1) * 3] - 2 * pixels[(y_m1 * width + x) * 3] -
                 1 * pixels[(y_m1 * width + x_p1) * 3] + 1 * pixels[(y_p1 * width + x_m1) * 3] +
                 2 * pixels[(y_p1 * width + x) * 3] + 1 * pixels[(y_p1 * width + x_p1) * 3];

            // Stärke des Gradienten als Summe der Beträge.
            e_1 = ABS(gx) + ABS(gy);

            // ENTROPIE e_Ent
            int bins[9] = {0};                            // 9 Bins für Grauwertbereiche (Anzahl)
            int local_min = INT_MAX, local_max = INT_MIN; // kleinstes und größtes Grau in 9x9 Umgebung

            // 1. Durchlauf: Min/Max Grauwert in der 9x9 Umgebung finden
            for (int v = -4; v < 4; ++v)
            {
                for (int u = -4; u < 4; ++u)
                {
                    int val = pixels[(CLAMP(y + v, 0, height - 1) * width + CLAMP(x + u, 0, width - 1)) * 3];
                    local_min = MIN(local_min, val);
                    local_max = MAX(local_max, val);
                }
            }
            int hist_width = local_max - local_min + 1;

            // 2. Durchlauf: Häufigkeit der Grauwerte in 9 "Bins" zählen
            for (int v = -4; v < 4; ++v)
            {
                for (int u = -4; u < 4; ++u)
                {
                    int val = pixels[(CLAMP(y + v, 0, height - 1) * width + CLAMP(x + u, 0, width - 1)) * 3];
                    int i = (val - local_min) * 9 / hist_width;
                    bins[i]++;
                }
            }

            // Entropie berechnen unter Verwendung der Look-Up Table (LUT)
            int e_entropy = 0;
            for (int i = 0; i < 9; ++i)
            {
                e_entropy += entropy_lut[bins[i]];
            }

            // Gesamte Pixelenergie
            output[y * width + x] = e_1 + e_entropy; // Kombination aus Kantenstärke (e1) und Texturkomplexität (e_entropy)
        }
    }
    return output;
}

// Step 3.2: Startpunkte der k minimalen Pfade finden
// Strategie: Ausführung auf CPU.
// Algorithmus (finden der k kleinsten Werte in einer Reihe unter Berücksichtigung von Ausschlüssen)
// ist stark sequentiell und lohnt den GPU-Overhead nicht.
void findAllSeams(unsigned int *minEnergySums, int width, int height, int k, int *seamStartIndices)
{
    // Äußere Schleife: Wir suchen nacheinander k verschiedene Seams.
    for (int seam_id = 0; seam_id < k; ++seam_id)
    {
        int min_x = -1;
        unsigned int min_energy = UINT_MAX;

        // Suche den besten Startpunkt (x-Koordinate) in der untersten Zeile.
        for (int x = 0; x < width; ++x)
        {
            // Prüfen, ob dieser Index x bereits für einen vorherigen Seam gewählt wurde.
            int already_selected = 0;
            for (int prev = 0; prev < seam_id; ++prev)
            {
                if (seamStartIndices[prev] == x)
                {
                    already_selected = 1; // Index ist bereits vergeben
                    break;
                }
            }

            // Falls x schon belegt ist, überspringen wir diesen Pixel.
            if (already_selected)
                continue;

            // Hole die kumulierte Energie des Pixels aus der untersten Zeile (height - 1).
            unsigned int energy = minEnergySums[(height - 1) * width + x];

            // Prüfe, ob die aktuelle Energie kleiner als das bisherige Minimum ist.
            // Zusätzliche Bedingung (Tie-Breaking): Bei gleicher Energie wählen wir das kleinere x (weiter links).
            if (energy < min_energy || (energy == min_energy && (min_x == -1 || x < min_x)))
            {
                min_energy = energy;
                min_x = x;
            }
        }

        // Falls kein gültiger Index gefunden wurde (Sicherheitscheck), Fallback auf 0.
        if (min_x == -1)
            min_x = 0;

        // Speichere den gefundenen Start-Index für diesen Seam.
        seamStartIndices[seam_id] = min_x;
    }

    // Sortiere alle gefundenen Start-Indizes aufsteigend.
    // Dies ist notwendig, damit insertAllSeams die Pixel in der richtigen Reihenfolge einfügen kann.
    qsort(seamStartIndices, k, sizeof(int), compare_ints);
}

// Step 3.3: Pfade der k minimalen Seams verfolgen
void traceAllSeams(unsigned int *minEnergySums, int width, int height, int k, int *seamStartIndices, int *seamPaths)
{
// Strategie: Parallelisierung über die Anzahl der Seams (k).
// Jeder Thread berechnet einen kompletten Pfad (Bottom-Up).
// Read-Only Zugriff auf minEnergySums ermöglicht konfliktfreie Parallelisierung.
#pragma omp target teams distribute parallel for map(to : minEnergySums[0 : width * height], seamStartIndices[0 : k]) map(from : seamPaths[0 : k * height])
    for (int seam_id = 0; seam_id < k; ++seam_id)
    {
        int x = seamStartIndices[seam_id];
        seamPaths[seam_id * height + (height - 1)] = x;

        // Laufe das Bild von der vorletzten Zeile (height - 2) bis zur ersten Zeile (0) hoch.
        for (int y = height - 2; y >= 0; --y)
        {
            unsigned int min_val;
            int next_x = x;
            // Wert direkt über dem aktuellen Pixel (v0).
            unsigned int v0 = minEnergySums[y * width + x];

            // Fall 1: Der aktuelle Pixel befindet sich am linken Bildrand.
            if (x == 0)
            {
                // Nur Nachbarn oben (v0) und oben-rechts (v1) prüfen.
                unsigned int v1 = minEnergySums[y * width + x + 1];
                min_val = MIN(v0, v1);
                // Wenn der rechte Nachbar kleiner ist, wechsle die x-Position.
                if (v1 == min_val)
                    next_x = x + 1;
            }
            // Fall 2: Der aktuelle Pixel befindet sich am rechten Bildrand.
            else if (x == width - 1)
            {
                // Nur Nachbarn oben (v0) und oben-links (vm1) prüfen.
                unsigned int vm1 = minEnergySums[y * width + x - 1];
                min_val = MIN(vm1, v0);
                // Wenn der linke Nachbar kleiner oder gleich ist, wechsle die x-Position.
                if (vm1 == min_val)
                    next_x = x - 1;
            }
            // Fall 3: Der Pixel hat drei Nachbarn in der Zeile darüber.
            else
            {
                unsigned int vm1 = minEnergySums[y * width + x - 1]; // Oben-Links
                unsigned int v1 = minEnergySums[y * width + x + 1];  // Oben-Rechts

                // Finde das Minimum aus den drei Nachbarn (v0, vm1, v1).
                min_val = MIN(MIN(vm1, v0), v1);

                // Priorisierung: Wenn links kleiner oder gleich ist, gehe nach links.
                if (vm1 == min_val)
                    next_x = x - 1;
                // Ansonsten, wenn rechts kleiner ist, gehe nach rechts.
                else if (v1 == min_val)
                    next_x = x + 1;
                // Bleibe sonst in der Mitte (next_x = x).
            }

            // Aktualisiere die x-Koordinate für die nächste Zeile.
            x = next_x;

            // Speichere die gewählte Koordinate für diese Zeile y im Gesamtpfad des Seams.
            seamPaths[seam_id * height + y] = x;
        }
    }
}

// Step 4
struct imgRawImage *insertAllSeams(struct imgRawImage *image, int k, int *seamPaths)
{
    int height = image->height;
    int width = image->width;
    int new_width = width + k; // Das Bild wird um k Pixel breiter
    unsigned char *newData = malloc(sizeof(unsigned char) * 3 * new_width * height);
    unsigned char *oldData = image->lpData;

// Strategie: Parallelisierung über Zeilen (y). Jede Zeile wird eigenständig verbreitert.
#pragma omp target teams distribute parallel for map(to : oldData[0 : width * height * 3], seamPaths[0 : k * height]) map(from : newData[0 : new_width * height * 3])
    for (int y = 0; y < height; ++y)
    {
        int seam_positions[4096];           // Lokaler Speicher auf der GPU für die Seam-Positionen dieser Zeile
        int load_k = (k < 4096) ? k : 4096; // Begrenzung durch GPU-Stack-Größe

        // 1. Schritt: Alle k Seam-Positionen für die aktuelle Zeile y laden
        for (int s = 0; s < load_k; ++s)
            seam_positions[s] = seamPaths[s * height + y];

        // 2. Schritt: Seam-Positionen sortieren (Bubble Sort)
        // Notwendig, damit wir die Zeile linear von links nach rechts füllen können.
        for (int i = 0; i < load_k - 1; ++i)
            for (int j = 0; j < load_k - i - 1; ++j)
                if (seam_positions[j] > seam_positions[j + 1])
                {
                    int t = seam_positions[j];
                    seam_positions[j] = seam_positions[j + 1];
                    seam_positions[j + 1] = t;
                }

        int write_idx = 0; // Schreib-Zeiger für das neue, breitere Bild
        int seam_idx = 0;  // Zeiger auf den nächsten einzufügenden Seam

        // 3. Schritt: Original-Pixel kopieren und bei Bedarf duplizieren
        for (int x = 0; x < width; ++x)
        {
            // Original-Pixel (RGB) in das neue Bild kopieren
            newData[(y * new_width + write_idx) * 3 + 0] = oldData[(y * width + x) * 3 + 0];
            newData[(y * new_width + write_idx) * 3 + 1] = oldData[(y * width + x) * 3 + 1];
            newData[(y * new_width + write_idx) * 3 + 2] = oldData[(y * width + x) * 3 + 2];
            write_idx++;

            // Falls der aktuelle Pixel x Teil eines (oder mehrerer) Seams ist:
            // Füge den gleichen Pixel erneut ein (Duplikation).
            while (seam_idx < load_k && seam_positions[seam_idx] == x)
            {
                newData[(y * new_width + write_idx) * 3 + 0] = oldData[(y * width + x) * 3 + 0];
                newData[(y * new_width + write_idx) * 3 + 1] = oldData[(y * width + x) * 3 + 1];
                newData[(y * new_width + write_idx) * 3 + 2] = oldData[(y * width + x) * 3 + 2];
                write_idx++;
                seam_idx++;
            }
        }
    }
    free(oldData); // Alten Bildspeicher freigeben
    image->lpData = newData;
    image->width = new_width; // Neue Breite im Header speichern
    return image;
}

struct imgRawImage *increaseWidth(struct imgRawImage *image, int seams)
{
    int *entropy_lut = create_entropy_lut();
    // Datenfluss: Ergebnisse werden zwischen Schritten zur CPU zurückgeholt,
    // um an die nächste Funktion übergeben zu werden.
    unsigned int *pixelEnergies = calculateEnergySobel(image, entropy_lut);
    free(entropy_lut);
    unsigned int *minEnergySums = calculateMinEnergySums(pixelEnergies, image->width, image->height);
    free(pixelEnergies);
    int *seamStartIndices = malloc(sizeof(int) * seams);

    findAllSeams(minEnergySums, image->width, image->height, seams, seamStartIndices);
    int *seamPaths = malloc(sizeof(int) * seams * image->height); // Memory Coalescing (1D Array für GPU)

    traceAllSeams(minEnergySums, image->width, image->height, seams, seamStartIndices, seamPaths);
    free(minEnergySums);
    free(seamStartIndices);

    insertAllSeams(image, seams, seamPaths);
    free(seamPaths);
    return image;
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        printf("Usage: %s inputJPEG outputJPEG numSeams\n", argv[0]);
        return 0;
    }
    struct imgRawImage *input = loadJpegImageFile(argv[1]);
    clock_t start = clock();
    input->lpData = gray(input);
    struct imgRawImage *output = increaseWidth(input, atoi(argv[3]));
    clock_t end = clock();
    printf("Execution time: %4.2f sec\n", (double)((double)(end - start) / CLOCKS_PER_SEC));
    storeJpegImageFile(output, argv[2]);
    return 0;
}