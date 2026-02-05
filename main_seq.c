#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include "image.h"

// Macros
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define ABS(x) (((x) < 0) ? -(x) : (x))
#define CLAMP(val, min, max) ((val) < (min) ? (min) : ((val) > (max) ? (max) : (val)))

// Pre-compute Entropy LUT (CPU)
int *create_entropy_lut() {
    int *lut = malloc(sizeof(int) * 82);
    lut[0] = 0;
    for (int i = 1; i <= 81; ++i) {
        double p = (double)i / 81.0;
        lut[i] = (int)(-1.0 * log2(p) * p * (CHAR_MAX / 5.0 * 3.2));
    }
    return lut;
}

// Comparison for qsort
int compare_ints(const void *a, const void *b) {
    int arg1 = *(const int *)a;
    int arg2 = *(const int *)b;
    return (arg1 > arg2) - (arg1 < arg2);
}

// Integer Grayscale Conversion
unsigned char *gray(struct imgRawImage *image) {
    unsigned int width = image->width;
    unsigned int height = image->height;
    unsigned char *output = malloc(sizeof(unsigned char) * 3 * width * height);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * 3;
            unsigned char r = image->lpData[idx + 0];
            unsigned char g = image->lpData[idx + 1];
            unsigned char b = image->lpData[idx + 2];

            // Integer math for deterministic results across all platforms
            unsigned char luma = (r * 299 + g * 587 + b * 114) / 1000;

            output[idx + 0] = luma;
            output[idx + 1] = luma;
            output[idx + 2] = luma;
        }
    }
    return output;
}

unsigned int *calculateMinEnergySums(unsigned int *data, int width, int height) {
    unsigned int *output = malloc(sizeof(unsigned int) * width * height);

    for (int x = 0; x < width; ++x) {
        output[x] = data[x];
    }

    for (int y = 1; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            unsigned int min_val;
            int idx_prev = (y - 1) * width + x;

            if (x == width - 1) {
                min_val = MIN(output[idx_prev - 1], output[idx_prev]);
            } else if (x == 0) {
                min_val = MIN(output[idx_prev], output[idx_prev + 1]);
            } else {
                min_val = MIN(MIN(output[idx_prev - 1], output[idx_prev]), output[idx_prev + 1]);
            }
            output[y * width + x] = data[y * width + x] + min_val;
        }
    }
    return output;
}

unsigned int *calculateEnergySobel(struct imgRawImage *image, int *entropy_lut) {
    unsigned int width = image->width;
    unsigned int height = image->height;
    unsigned int *output = malloc(sizeof(unsigned int) * height * width);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int gx, gy, e_1;
            int y_m1 = CLAMP(y - 1, 0, height - 1);
            int y_p1 = CLAMP(y + 1, 0, height - 1);
            int x_m1 = CLAMP(x - 1, 0, width - 1);
            int x_p1 = CLAMP(x + 1, 0, width - 1);

            gx = -1 * image->lpData[(y_m1 * width + x_m1) * 3] + 1 * image->lpData[(y_m1 * width + x_p1) * 3]
            - 2 * image->lpData[(y * width + x_m1) * 3] + 2 * image->lpData[(y * width + x_p1) * 3]
            - 1 * image->lpData[(y_p1 * width + x_m1) * 3] + 1 * image->lpData[(y_p1 * width + x_p1) * 3];

            gy = -1 * image->lpData[(y_m1 * width + x_m1) * 3] - 2 * image->lpData[(y_m1 * width + x) * 3] - 1 * image->lpData[(y_m1 * width + x_p1) * 3]
            + 1 * image->lpData[(y_p1 * width + x_m1) * 3] + 2 * image->lpData[(y_p1 * width + x) * 3] + 1 * image->lpData[(y_p1 * width + x_p1) * 3];

            e_1 = ABS(gx) + ABS(gy);

            int bins[9] = {0};
            int local_min = INT_MAX;
            int local_max = INT_MIN;

            for (int v = -4; v < 4; ++v) {
                for (int u = -4; u < 4; ++u) {
                    int val = image->lpData[(CLAMP(y + v, 0, height - 1) * width + CLAMP(x + u, 0, width - 1)) * 3];
                    local_min = MIN(local_min, val);
                    local_max = MAX(local_max, val);
                }
            }
            int hist_width = local_max - local_min + 1;

            for (int v = -4; v < 4; ++v) {
                for (int u = -4; u < 4; ++u) {
                    int val = image->lpData[(CLAMP(y + v, 0, height - 1) * width + CLAMP(x + u, 0, width - 1)) * 3];
                    int i = (val - local_min) * 9 / hist_width;
                    bins[i]++;
                }
            }

            int e_entropy = 0;
            for (int i = 0; i < 9; ++i) {
                e_entropy += entropy_lut[bins[i]];
            }

            output[y * width + x] = e_1 + e_entropy;
        }
    }
    return output;
}

void findAllSeams(unsigned int *minEnergySums, int width, int height, int k, int *seamStartIndices) {
    for (int seam_id = 0; seam_id < k; ++seam_id) {
        int min_x = -1;
        unsigned int min_energy = UINT_MAX;
        for (int x = 0; x < width; ++x) {
            int already_selected = 0;
            for (int prev = 0; prev < seam_id; ++prev) {
                if (seamStartIndices[prev] == x) { already_selected = 1; break; }
            }
            if (already_selected) continue;

            unsigned int energy = minEnergySums[(height - 1) * width + x];
            if (energy < min_energy || (energy == min_energy && (min_x == -1 || x < min_x))) {
                min_energy = energy;
                min_x = x;
            }
        }
        if (min_x == -1) min_x = 0;
        seamStartIndices[seam_id] = min_x;
    }
    qsort(seamStartIndices, k, sizeof(int), compare_ints);
}

void traceAllSeams(unsigned int *minEnergySums, int width, int height, int k, int *seamStartIndices, int *seamPaths) {
    for (int seam_id = 0; seam_id < k; ++seam_id) {
        int x = seamStartIndices[seam_id];
        seamPaths[seam_id * height + (height - 1)] = x;
        for (int y = height - 2; y >= 0; --y) {
            unsigned int min_val;
            int next_x = x;
            unsigned int v0 = minEnergySums[y * width + x];

            if (x == 0) {
                unsigned int v1 = minEnergySums[y * width + x + 1];
                min_val = MIN(v0, v1);
                if (v1 == min_val) next_x = x + 1;
            } else if (x == width - 1) {
                unsigned int vm1 = minEnergySums[y * width + x - 1];
                min_val = MIN(vm1, v0);
                if (vm1 == min_val) next_x = x - 1;
            } else {
                unsigned int vm1 = minEnergySums[y * width + x - 1];
                unsigned int v1 = minEnergySums[y * width + x + 1];
                min_val = MIN(MIN(vm1, v0), v1);
                if (vm1 == min_val) next_x = x - 1;
                else if (v1 == min_val) next_x = x + 1;
            }
            x = next_x;
            seamPaths[seam_id * height + y] = x;
        }
    }
}

struct imgRawImage *insertAllSeams(struct imgRawImage *image, int k, int *seamPaths) {
    int height = image->height;
    int width = image->width;
    int new_width = width + k;
    unsigned char *newData = malloc(sizeof(unsigned char) * 3 * new_width * height);
    unsigned char *oldData = image->lpData;

    for (int y = 0; y < height; ++y) {
        int seam_positions[4096];
        int load_k = (k < 4096) ? k : 4096;
        for (int s = 0; s < load_k; ++s) seam_positions[s] = seamPaths[s * height + y];

        for(int i=0; i<load_k-1; ++i)
            for(int j=0; j<load_k-i-1; ++j)
                if(seam_positions[j] > seam_positions[j+1]) {
                    int t = seam_positions[j]; seam_positions[j] = seam_positions[j+1]; seam_positions[j+1] = t;
                }

                int write_idx = 0;
            int seam_idx = 0;
        for (int x = 0; x < width; ++x) {
            newData[(y * new_width + write_idx) * 3 + 0] = oldData[(y * width + x) * 3 + 0];
            newData[(y * new_width + write_idx) * 3 + 1] = oldData[(y * width + x) * 3 + 1];
            newData[(y * new_width + write_idx) * 3 + 2] = oldData[(y * width + x) * 3 + 2];
            write_idx++;
            while (seam_idx < load_k && seam_positions[seam_idx] == x) {
                newData[(y * new_width + write_idx) * 3 + 0] = oldData[(y * width + x) * 3 + 0];
                newData[(y * new_width + write_idx) * 3 + 1] = oldData[(y * width + x) * 3 + 1];
                newData[(y * new_width + write_idx) * 3 + 2] = oldData[(y * width + x) * 3 + 2];
                write_idx++;
                seam_idx++;
            }
        }
    }
    free(oldData);
    image->lpData = newData;
    image->width = new_width;
    return image;
}

struct imgRawImage *increaseWidth(struct imgRawImage *image, int seams) {
    int *entropy_lut = create_entropy_lut();
    unsigned int *pixelEnergies = calculateEnergySobel(image, entropy_lut);
    free(entropy_lut);
    unsigned int *minEnergySums = calculateMinEnergySums(pixelEnergies, image->width, image->height);
    free(pixelEnergies);
    int *seamStartIndices = malloc(sizeof(int) * seams);
    findAllSeams(minEnergySums, image->width, image->height, seams, seamStartIndices);
    int *seamPaths = malloc(sizeof(int) * seams * image->height);
    traceAllSeams(minEnergySums, image->width, image->height, seams, seamStartIndices, seamPaths);
    free(minEnergySums);
    free(seamStartIndices);
    insertAllSeams(image, seams, seamPaths);
    free(seamPaths);
    return image;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
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
