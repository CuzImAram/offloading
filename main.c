#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <omp.h>
#include "image.h"

#define d3(y, x, z) image->lpData[(y)*image->width*3+(x)*3+(z)]
#define o3(y, x, z) output[(y)*width*3+(x)*3+(z)]
#define od3(y, x, z) oldData[(y)*width*3+(x)*3+(z)]
#define nd3(y, x, z) newData[(y)*(width+1)*3+(x)*3+(z)]
#define m1(y, x) minEnergySums[(y)*width+(x)]
#define d1(y, x) data[(y)*width+(x)]
#define o(y, x) output[(y)*width+(x)]
#define nw(y, x) newMinEnergySums[(y)*(width+1)+(x)]
#define MIN(a, b) (((a)<(b))?(a):(b))
#define MAX(a, b) (((a)>(b))?(a):(b))
#define ABS(x) (((x)<0)?-(x):(x))

// Note: Since the gradient isn't normalized,
// we rescale the summands in the entropy calculations slightly,
#define entrop(p) (-1.0 * log2((p)) * (p) * (CHAR_MAX / 5.0 * 3.2))

// Torus indexing for safe boundary access
#define TORUS_Y(y, height) (((y) + (height)) % (height))
#define TORUS_X(x, width) (((x) + (width)) % (width))


unsigned char *gray(struct imgRawImage *image) {
    unsigned int width = image->width;
    unsigned int height = image->height;
    unsigned char *output = malloc(sizeof(unsigned char) * 3 * width * height);
    unsigned char *input = image->lpData;

    // GPU offloading for gray conversion - fully parallel
    #pragma omp target teams distribute parallel for collapse(2) \
    map(to: input[0:width*height*3]) \
    map(from: output[0:width*height*3])
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width * 3 + x * 3;
            unsigned char luma = (unsigned char) (
                0.299f * (float) input[idx + 0]
                + 0.587f * (float) input[idx + 1]
                + 0.114f * (float) input[idx + 2]);
            output[idx + 0] = luma;
            output[idx + 1] = luma;
            output[idx + 2] = luma;
        }
    }
    return output;
}

unsigned int *calculateMinEnergySums(unsigned int *data, int width, int height) {
    unsigned int *output = malloc(sizeof(unsigned int) * width * height);

    // First row initialization - parallel
    #pragma omp target teams distribute parallel for \
    map(to: data[0:width*height]) \
    map(from: output[0:width*height])
    for (int x = 0; x < width; ++x) {
        output[x] = data[x];
    }

    // Subsequent rows - must be sequential in y, but parallel in x
    // Each row depends on previous row, so we do row-by-row on GPU
    #pragma omp target data map(to: data[0:width*height]) \
    map(tofrom: output[0:width*height])
    {
        for (int y = 1; y < height; ++y) {
            #pragma omp target teams distribute parallel for
            for (int x = 0; x < width; ++x) {
                unsigned int min_val;
                int idx = y * width + x;
                int idx_prev = (y - 1) * width + x;

                if (x == width - 1) { // rightmost pixel
                    min_val = MIN(output[idx_prev - 1], output[idx_prev]);
                } else if (x == 0) { // leftmost pixel
                    min_val = MIN(output[idx_prev], output[idx_prev + 1]);
                } else { // middle pixels
                    min_val = MIN(MIN(output[idx_prev - 1], output[idx_prev]),
                                  output[idx_prev + 1]);
                }
                output[idx] = data[idx] + min_val;
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
    unsigned char *input = image->lpData;

    // GPU offloading for energy calculation
    // This is the most computationally intensive part
    #pragma omp target teams distribute parallel for collapse(2) \
    map(to: input[0:width*height*3]) \
    map(from: output[0:width*height])
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int gx, gy, e_1, local_min, local_max, hist_width, e_entropy;
            double bins[9];

            // Step 1: Compute edge-component using Torus indexing
            // apply Sobel operator in X direction
            int y_m1 = TORUS_Y(y - 1, height);
            int y_p1 = TORUS_Y(y + 1, height);
            int x_m1 = TORUS_X(x - 1, width);
            int x_p1 = TORUS_X(x + 1, width);

            gx = -1 * input[(y_m1 * width + x_m1) * 3 + 0]
            + 1 * input[(y_m1 * width + x_p1) * 3 + 0]
            - 2 * input[(y * width + x_m1) * 3 + 0]
            + 2 * input[(y * width + x_p1) * 3 + 0]
            - 1 * input[(y_p1 * width + x_m1) * 3 + 0]
            + 1 * input[(y_p1 * width + x_p1) * 3 + 0];

            // apply Sobel operator in Y direction
            gy = -1 * input[(y_m1 * width + x_m1) * 3 + 0]
            - 2 * input[(y_m1 * width + x) * 3 + 0]
            - 1 * input[(y_m1 * width + x_p1) * 3 + 0]
            + 1 * input[(y_p1 * width + x_m1) * 3 + 0]
            + 2 * input[(y_p1 * width + x) * 3 + 0]
            + 1 * input[(y_p1 * width + x_p1) * 3 + 0];

            e_1 = ABS(gx) + ABS(gy);

            // Step 2: Compute entropy-component with Torus indexing
            for (int i = 0; i < 9; ++i) {
                bins[i] = 0;
            }
            local_min = INT_MAX;
            local_max = INT_MIN;
            e_entropy = 0;

            // find min/max for local histogram (8x8 neighborhood)
            for (int v = -4; v < 4; ++v) {
                for (int u = -4; u < 4; ++u) {
                    int y_idx = TORUS_Y(y + v, height);
                    int x_idx = TORUS_X(x + u, width);
                    int pixel_val = input[(y_idx * width + x_idx) * 3 + 0];
                    local_min = MIN(local_min, pixel_val);
                    local_max = MAX(local_max, pixel_val);
                }
            }
            hist_width = local_max - local_min + 1;

            // compute local histogram
            for (int v = -4; v < 4; ++v) {
                for (int u = -4; u < 4; ++u) {
                    int y_idx = TORUS_Y(y + v, height);
                    int x_idx = TORUS_X(x + u, width);
                    int pixel_val = input[(y_idx * width + x_idx) * 3 + 0];
                    int i = (pixel_val - local_min) * 9 / hist_width;
                    bins[i] += 1.0;
                }
            }

            // compute entropy
            for (int i = 0; i < 9; ++i) {
                bins[i] /= 81.0;
                if (bins[i] > 0.0) {
                    e_entropy += (int) entrop(bins[i]);
                }
            }

            // Step 3: assign energy value
            output[y * width + x] = e_1 + e_entropy;
        }
    }
    return output;
}

// increases the number of columns by seams
struct imgRawImage *increaseWidth(struct imgRawImage *image, int seams) {
    int height = image->height;
    unsigned int *newMinEnergySums;
    unsigned char *newData;

    unsigned int *pixelEnergies = calculateEnergySobel(image);
    unsigned int *minEnergySums = calculateMinEnergySums(pixelEnergies, image->width, image->height);
    free(pixelEnergies);

    // find seams by looking at the bottom row
    unsigned int mins[seams];
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
            if (skip == 1) {
                continue;
            }
            if (mins[k] == width || m1(height - 1, j) < m1(height - 1, mins[k])) {
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

    // Seam insertion remains on CPU - too complex for efficient GPU parallelization
    for (int i = 0; i < seams; ++i) {
        unsigned int minIdx = mins[i];
        int width = image->width;
        unsigned char *oldData = image->lpData;
        newMinEnergySums = malloc(sizeof(unsigned int) * (width + 1) * height);
        newData = malloc(sizeof(unsigned char) * 3 * (width + 1) * height);

        // copy the pixels on the left side of the seam
        for (int j = 0; j <= minIdx; ++j) {
            nw(height - 1, j) = m1(height - 1, j);
            nd3(height - 1, j, 0) = od3(height - 1, j, 0);
            nd3(height - 1, j, 1) = od3(height - 1, j, 1);
            nd3(height - 1, j, 2) = od3(height - 1, j, 2);
        }
        nw(height - 1, minIdx + 1) = m1(height - 1, minIdx);
        nd3(height - 1, minIdx + 1, 0) = od3(height - 1, minIdx, 0);
        nd3(height - 1, minIdx + 1, 1) = od3(height - 1, minIdx, 1);
        nd3(height - 1, minIdx + 1, 2) = od3(height - 1, minIdx, 2);

        for (int j = minIdx + 1; j < width; ++j) {
            nw(height - 1, j + 1) = m1(height - 1, j);
            nd3(height - 1, j + 1, 0) = od3(height - 1, j, 0);
            nd3(height - 1, j + 1, 1) = od3(height - 1, j, 1);
            nd3(height - 1, j + 1, 2) = od3(height - 1, j, 2);
        }

        int x = minIdx;
        for (int y = height - 2; y >= 0; --y) {
            unsigned int min;
            if (x == 0) {
                min = MIN(m1(y, x), m1(y, x + 1));
            } else if (x == width - 1) {
                min = MIN(m1(y, x - 1), m1(y, x));
            } else {
                min = MIN(m1(y, x - 1), MIN(m1(y, x), m1(y, x + 1)));
            }
            if (x > 0 && m1(y, x - 1) == min) {
                x = x - 1;
            } else if (x < width - 1 && m1(y, x + 1) == min) {
                x = x + 1;
            }
            for (int j = 0; j <= x; ++j) {
                nw(y, j) = m1(y, j);
                nd3(y, j, 0) = od3(y, j, 0);
                nd3(y, j, 1) = od3(y, j, 1);
                nd3(y, j, 2) = od3(y, j, 2);
            }
            nw(y, x + 1) = m1(y, x);
            nd3(y, x + 1, 0) = od3(y, x, 0);
            nd3(y, x + 1, 1) = od3(y, x, 1);
            nd3(y, x + 1, 2) = od3(y, x, 2);
            for (int j = x + 1; j < width; ++j) {
                nw(y, j + 1) = m1(y, j);
                nd3(y, j + 1, 0) = od3(y, j, 0);
                nd3(y, j + 1, 1) = od3(y, j, 1);
                nd3(y, j + 1, 2) = od3(y, j, 2);
            }
        }
        free(image->lpData);
        image->lpData = newData;
        image->width = width + 1;
        free(minEnergySums);
        minEnergySums = newMinEnergySums;
    }
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
    clock_t start = clock();

    input->lpData = gray(input);
    struct imgRawImage *output = increaseWidth(input, seams);

    clock_t end = clock();
    printf("Execution time: %4.2f sec\n", (double) ((double) (end - start) / CLOCKS_PER_SEC));
    storeJpegImageFile(output, outputFile);

    return 0;
}
