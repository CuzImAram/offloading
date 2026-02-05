#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <omp.h>
#include "image.h"

// Hilfsmacros angepasst für direkte Pointer-Nutzung (für GPU/CPU Kompatibilität)
#define IDX3(y, x, z, w) ((y)*(w)*3 + (x)*3 + (z))
#define IDX1(y, x, w) ((y)*(w) + (x))

#define MIN(a, b) (((a)<(b))?(a):(b))
#define MAX(a, b) (((a)>(b))?(a):(b))
#define ABS(x) (((x) < 0) ? -(x) : (x))

// Torus Macros
#define TORUS_Y(y, height) (((y) + (height)) % (height))
#define TORUS_X(x, width) (((x) + (width)) % (width))

#define ENTROP(p) (-1.0 * log2((p)) * (p) * (CHAR_MAX / 5.0 * 3.2))

unsigned char *gray(struct imgRawImage *image) {
    unsigned int width = image->width;
    unsigned int height = image->height;
    unsigned char *output = malloc(sizeof(unsigned char) * 3 * width * height);
    unsigned char *input_data = image->lpData;

    // Offload auf GPU (Beibehalten)
    #pragma omp target teams distribute parallel for map(to: input_data[0:3*width*height]) map(from: output[0:3*width*height])
    for (int i = 0; i < height * width; ++i) {
        int y = i / width;
        int x = i % width;

        float r = (float)input_data[IDX3(y, x, 0, width)];
        float g = (float)input_data[IDX3(y, x, 1, width)];
        float b = (float)input_data[IDX3(y, x, 2, width)];

        unsigned char luma = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);

        output[IDX3(y, x, 0, width)] = luma;
        output[IDX3(y, x, 1, width)] = luma;
        output[IDX3(y, x, 2, width)] = luma;
    }
    return output;
}

unsigned int *calculateMinEnergySums(unsigned int *data, int width, int height) {
    unsigned int *output = malloc(sizeof(unsigned int) * width * height);

    // Wavefront-Pattern auf der GPU (Beibehalten)
    #pragma omp target data map(to: data[0:width*height]) map(from: output[0:width*height])
    {
        // Erste Zeile initialisieren
        #pragma omp target teams distribute parallel for
        for (int x = 0; x < width; ++x) {
            output[IDX1(0, x, width)] = data[IDX1(0, x, width)];
        }

        // Iteriere über Zeilen (Sequenziell auf dem Device Controller, parallel in X)
        #pragma omp target teams num_teams(1) thread_limit(1024)
        {
            for (int y = 1; y < height; ++y) {
                #pragma omp parallel for
                for (int x = 0; x < width; ++x) {
                    unsigned int val = data[IDX1(y, x, width)];
                    unsigned int min_prev;

                    unsigned int up = output[IDX1(y - 1, x, width)];

                    if (x == 0) { // linker Rand
                        unsigned int up_right = output[IDX1(y - 1, x + 1, width)];
                        min_prev = MIN(up, up_right);
                    } else if (x == width - 1) { // rechter Rand
                        unsigned int up_left = output[IDX1(y - 1, x - 1, width)];
                        min_prev = MIN(up_left, up);
                    } else { // Mitte
                        unsigned int up_left = output[IDX1(y - 1, x - 1, width)];
                        unsigned int up_right = output[IDX1(y - 1, x + 1, width)];
                        min_prev = MIN(MIN(up_left, up), up_right);
                    }
                    output[IDX1(y, x, width)] = val + min_prev;
                }
            }
        }
    }
    return output;
}

void swap(unsigned int *xp, unsigned int *yp) {
    int temp = *xp;
    *xp = *yp;
    *yp = temp;
}

void selectionSort(unsigned int arr[], int n) {
    int i, j, max_idx;
    for (i = 0; i < n - 1; i++) {
        max_idx = i;
        for (j = i + 1; j < n; j++)
            if (arr[j] > arr[max_idx])
                max_idx = j;
        swap(&arr[max_idx], &arr[i]);
    }
}

unsigned int *calculateEnergySobel(struct imgRawImage *image) {
    unsigned int width = image->width;
    unsigned int height = image->height;
    unsigned int *output = malloc(sizeof(unsigned int) * height * width);
    unsigned char *lpData = image->lpData;

    // GPU Offloading für Sobel/Entropie (Beibehalten)
    #pragma omp target teams distribute parallel for collapse(2) \
    map(to: lpData[0:3*width*height]) \
    map(from: output[0:width*height])
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Indizes vorberechnen
            int y_m1 = TORUS_Y(y - 1, height);
            int y_p1 = TORUS_Y(y + 1, height);
            int x_m1 = TORUS_X(x - 1, width);
            int x_p1 = TORUS_X(x + 1, width);

            // Zugriff auf lpData im Device Memory
            // Sobel X
            int gx = -1 * lpData[IDX3(y_m1, x_m1, 0, width)]
            + 1 * lpData[IDX3(y_m1, x_p1, 0, width)]
            - 2 * lpData[IDX3(y, x_m1, 0, width)]
            + 2 * lpData[IDX3(y, x_p1, 0, width)]
            - 1 * lpData[IDX3(y_p1, x_m1, 0, width)]
            + 1 * lpData[IDX3(y_p1, x_p1, 0, width)];

            // Sobel Y
            int gy = -1 * lpData[IDX3(y_m1, x_m1, 0, width)]
            - 2 * lpData[IDX3(y_m1, x, 0, width)]
            - 1 * lpData[IDX3(y_m1, x_p1, 0, width)]
            + 1 * lpData[IDX3(y_p1, x_m1, 0, width)]
            + 2 * lpData[IDX3(y_p1, x, 0, width)]
            + 1 * lpData[IDX3(y_p1, x_p1, 0, width)];

            int e_1 = ABS(gx) + ABS(gy);

            // Entropie
            double bins[9] = {0};
            int local_min = INT_MAX;
            int local_max = INT_MIN;
            int e_entropy = 0;

            for (int v = -4; v < 4; ++v) {
                for (int u = -4; u < 4; ++u) {
                    int y_idx = TORUS_Y(y + v, height);
                    int x_idx = TORUS_X(x + u, width);
                    int val = lpData[IDX3(y_idx, x_idx, 0, width)];
                    if (val < local_min) local_min = val;
                    if (val > local_max) local_max = val;
                }
            }
            int hist_width = local_max - local_min + 1;

            for (int v = -4; v < 4; ++v) {
                for (int u = -4; u < 4; ++u) {
                    int y_idx = TORUS_Y(y + v, height);
                    int x_idx = TORUS_X(x + u, width);
                    int val = lpData[IDX3(y_idx, x_idx, 0, width)];
                    int i = (val - local_min) * 9 / hist_width;
                    if (i < 0) i = 0;
                    if (i > 8) i = 8;
                    bins[i] += 1.0;
                }
            }

            for (int i = 0; i < 9; ++i) {
                bins[i] /= 81.0;
                if (bins[i] > 0.0) {
                    e_entropy += (int) ENTROP(bins[i]);
                }
            }

            output[IDX1(y, x, width)] = e_1 + e_entropy;
        }
    }
    return output;
}

struct imgRawImage *increaseWidth(struct imgRawImage *image, int seams) {
    int height = image->height;
    unsigned int *newMinEnergySums;
    unsigned char *newData;

    // GPU Offloading bleibt hier erhalten
    unsigned int *pixelEnergies = calculateEnergySobel(image);
    unsigned int *minEnergySums = calculateMinEnergySums(pixelEnergies, image->width, image->height);
    free(pixelEnergies);

    // Seams finden (Sequenziell)
    unsigned int *mins = malloc(sizeof(unsigned int) * seams);
    int width = image->width;

    for (int k = 0; k < seams; ++k) {
        mins[k] = width;
        for (int j = 0; j < width; ++j) {
            int skip = 0;
            for (int l = 0; l < k; ++l) {
                if (mins[l] == j) {
                    skip = 1;
                    break;
                }
            }
            if (skip == 1) continue;

            unsigned int currentVal = minEnergySums[IDX1(height - 1, j, width)];
            unsigned int minVal = (mins[k] == width) ? UINT_MAX : minEnergySums[IDX1(height - 1, mins[k], width)];

            if (mins[k] == width || currentVal < minVal) {
                mins[k] = j;
            }
        }
    }

    for (int k = 0; k < seams; ++k) {
        if (mins[k] >= width) {
            mins[k] = width - 1;
        }
    }

    selectionSort(mins, seams);

    // Sequenzielle Verarbeitung der Datenstruktur-Updates auf der CPU
    for (int i = 0; i < seams; ++i) {
        unsigned int minIdx = mins[i];
        int width = image->width;
        unsigned char *oldData = image->lpData;

        newMinEnergySums = malloc(sizeof(unsigned int) * (width + 1) * height);
        newData = malloc(sizeof(unsigned char) * 3 * (width + 1) * height);

        // CPU-Parallelisierung entfernt (Overhead vermeiden)
        for (int j = 0; j <= minIdx; ++j) {
            newMinEnergySums[IDX1(height - 1, j, width + 1)] = minEnergySums[IDX1(height - 1, j, width)];
            newData[IDX3(height - 1, j, 0, width + 1)] = oldData[IDX3(height - 1, j, 0, width)];
            newData[IDX3(height - 1, j, 1, width + 1)] = oldData[IDX3(height - 1, j, 1, width)];
            newData[IDX3(height - 1, j, 2, width + 1)] = oldData[IDX3(height - 1, j, 2, width)];
        }

        newMinEnergySums[IDX1(height - 1, minIdx + 1, width + 1)] = minEnergySums[IDX1(height - 1, minIdx, width)];
        newData[IDX3(height - 1, minIdx + 1, 0, width + 1)] = oldData[IDX3(height - 1, minIdx, 0, width)];
        newData[IDX3(height - 1, minIdx + 1, 1, width + 1)] = oldData[IDX3(height - 1, minIdx, 1, width)];
        newData[IDX3(height - 1, minIdx + 1, 2, width + 1)] = oldData[IDX3(height - 1, minIdx, 2, width)];

        // CPU-Parallelisierung entfernt
        for (int j = minIdx + 1; j < width; ++j) {
            newMinEnergySums[IDX1(height - 1, j + 1, width + 1)] = minEnergySums[IDX1(height - 1, j, width)];
            newData[IDX3(height - 1, j + 1, 0, width + 1)] = oldData[IDX3(height - 1, j, 0, width)];
            newData[IDX3(height - 1, j + 1, 1, width + 1)] = oldData[IDX3(height - 1, j, 1, width)];
            newData[IDX3(height - 1, j + 1, 2, width + 1)] = oldData[IDX3(height - 1, j, 2, width)];
        }

        int x = minIdx;

        // Backtracking
        for (int y = height - 2; y >= 0; --y) {
            unsigned int min;
            unsigned int m_curr = minEnergySums[IDX1(y, x, width)];
            unsigned int m_left = (x > 0) ? minEnergySums[IDX1(y, x - 1, width)] : UINT_MAX;
            unsigned int m_right = (x < width - 1) ? minEnergySums[IDX1(y, x + 1, width)] : UINT_MAX;

            if (x == 0) {
                min = MIN(m_curr, m_right);
            } else if (x == width - 1) {
                min = MIN(m_left, m_curr);
            } else {
                min = MIN(m_left, MIN(m_curr, m_right));
            }

            if (x > 0 && m_left == min) {
                x = x - 1;
            } else if (x < width - 1 && m_right == min) {
                x = x + 1;
            }

            // CPU-Parallelisierung entfernt
            for (int j = 0; j <= x; ++j) {
                newMinEnergySums[IDX1(y, j, width + 1)] = minEnergySums[IDX1(y, j, width)];
                newData[IDX3(y, j, 0, width + 1)] = oldData[IDX3(y, j, 0, width)];
                newData[IDX3(y, j, 1, width + 1)] = oldData[IDX3(y, j, 1, width)];
                newData[IDX3(y, j, 2, width + 1)] = oldData[IDX3(y, j, 2, width)];
            }

            newMinEnergySums[IDX1(y, x + 1, width + 1)] = minEnergySums[IDX1(y, x, width)];
            newData[IDX3(y, x + 1, 0, width + 1)] = oldData[IDX3(y, x, 0, width)];
            newData[IDX3(y, x + 1, 1, width + 1)] = oldData[IDX3(y, x, 1, width)];
            newData[IDX3(y, x + 1, 2, width + 1)] = oldData[IDX3(y, x, 2, width)];

            // CPU-Parallelisierung entfernt
            for (int j = x + 1; j < width; ++j) {
                newMinEnergySums[IDX1(y, j + 1, width + 1)] = minEnergySums[IDX1(y, j, width)];
                newData[IDX3(y, j + 1, 0, width + 1)] = oldData[IDX3(y, j, 0, width)];
                newData[IDX3(y, j + 1, 1, width + 1)] = oldData[IDX3(y, j, 1, width)];
                newData[IDX3(y, j + 1, 2, width + 1)] = oldData[IDX3(y, j, 2, width)];
            }
        }

        free(image->lpData);
        image->lpData = newData;
        image->width = width + 1;
        free(minEnergySums);
        minEnergySums = newMinEnergySums;
    }

    free(mins);
    free(minEnergySums);
    return image;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Usage: %s inputJPEG outputJPEG numSeams\n", argv[0]);
        return 0;
    }
    char *inputFile = argv[1];
    char *outputFile = argv[2];
    int seams = atoi(argv[3]);

    struct imgRawImage *input = loadJpegImageFile(inputFile);

    double start = omp_get_wtime();

    input->lpData = gray(input);

    struct imgRawImage *output = increaseWidth(input, seams);

    double end = omp_get_wtime();
    printf("Execution time: %4.2f sec\n", end - start);
    storeJpegImageFile(output, outputFile);

    return 0;
}
