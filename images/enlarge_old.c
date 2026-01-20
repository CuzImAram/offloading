#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <string.h>
#include <omp.h>
#include "image.h"

#define MIN(a, b) (((a)<(b))?(a):(b))
#define MAX(a, b) (((a)>(b))?(a):(b))
#define CLAMP(val, min, max) MIN(MAX(val, min), max)

#define IDX3(y, x, z, w) ((y)*(w)*3 + (x)*3 + (z))
#define IDX1(y, x, w) ((y)*(w) + (x))

// Note: Since the gradient isn't normalized,
// we rescale the summands in the entropy calculations slightly,
// #define entrop(p) (-1.0 * log2((p)) * (p) * (CHAR_MAX / 5.0 * 3.2))
#pragma omp declare target
double entrop(double p) {
    if (p <= 0.0) return 0.0;
    return -1.0 * log2(p) * p * (CHAR_MAX / 5.0 * 3.2);
}
#pragma omp end declare target

unsigned char *gray(struct imgRawImage *image) {
    unsigned int width = image->width;
    unsigned int height = image->height;
    unsigned char *output = malloc(sizeof(unsigned char) * 3 * (width) * height);
    unsigned char *input_data = image->lpData;

    #pragma omp target map(to: input_data[0:3*width*height], width, height) map(from: output[0:3*width*height])
    #pragma omp teams distribute parallel for collapse(2)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            unsigned char luma = (unsigned char) (
                    0.299f * (float) input_data[IDX3(y, x, 0, width)]
                    + 0.587f * (float) input_data[IDX3(y, x, 1, width)]
                    + 0.114f * (float) input_data[IDX3(y, x, 2, width)]);
            output[IDX3(y, x, 0, width)] = luma;
            output[IDX3(y, x, 1, width)] = luma;
            output[IDX3(y, x, 2, width)] = luma;
        }
    }
    return output;
}

// Optimized swap/sort on host
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

// increases the number of columns by cols
struct imgRawImage *increaseWidth(struct imgRawImage *image, int seams) {
    int height = image->height;
    int initial_width = image->width;
    int final_width = initial_width + seams;

    // Use max width for allocation
    unsigned int max_pixels = final_width * height;
    
    // Allocate device buffers
    // We need ping-pong for image and minEnergySums
    // We treat strict pointer management manually
    unsigned char *d_img1 = (unsigned char *) omp_target_alloc(sizeof(unsigned char) * 3 * max_pixels, omp_get_default_device());
    unsigned char *d_img2 = (unsigned char *) omp_target_alloc(sizeof(unsigned char) * 3 * max_pixels, omp_get_default_device());
    
    unsigned int *d_energy = (unsigned int *) omp_target_alloc(sizeof(unsigned int) * max_pixels, omp_get_default_device());
    
    // Size changes, but we alloc max
    unsigned int *d_sums1 = (unsigned int *) omp_target_alloc(sizeof(unsigned int) * max_pixels, omp_get_default_device());
    unsigned int *d_sums2 = (unsigned int *) omp_target_alloc(sizeof(unsigned int) * max_pixels, omp_get_default_device());
    
    int *d_seamPath = (int *) omp_target_alloc(sizeof(int) * height, omp_get_default_device());

    // Copy initial image to d_img1
    unsigned char *srcData = image->lpData;
    // We need to copy row by row if we want to support padding? 
    // No, current buffer is compact 3*width*height.
    omp_target_memcpy(d_img1, srcData, sizeof(unsigned char) * 3 * initial_width * height, 
                      0, 0, omp_get_default_device(), omp_get_initial_device());

    // Current working pointers
    unsigned char *curr_img = d_img1;
    unsigned char *next_img = d_img2;
    unsigned int *curr_sums = d_sums1;
    unsigned int *next_sums = d_sums2;

    int current_width = initial_width;

    // --- Step 1: Calculate Energies (On Device) ---
    // Kernel: calculateEnergySobel
    #pragma omp target is_device_ptr(curr_img, d_energy)
    #pragma omp teams distribute parallel for collapse(2)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < current_width; ++x) {
            int gx, gy, e_1, local_min, local_max, hist_width, e_entropy;
            double bins[9];

            // Sobel X
            gx = -1 * curr_img[IDX3(CLAMP(y - 1, 0, height-1), CLAMP(x - 1, 0, current_width-1), 0, current_width)]
                 + 1 * curr_img[IDX3(CLAMP(y - 1, 0, height-1), CLAMP(x + 1, 0, current_width-1), 0, current_width)]
                 - 2 * curr_img[IDX3(y, CLAMP(x - 1, 0, current_width-1), 0, current_width)]
                 + 2 * curr_img[IDX3(y, CLAMP(x + 1, 0, current_width-1), 0, current_width)]
                 - 1 * curr_img[IDX3(CLAMP(y + 1, 0, height-1), CLAMP(x - 1, 0, current_width-1), 0, current_width)]
                 + 1 * curr_img[IDX3(CLAMP(y + 1, 0, height-1), CLAMP(x + 1, 0, current_width-1), 0, current_width)];

            // Sobel Y
            gy = -1 * curr_img[IDX3(CLAMP(y - 1, 0, height-1), CLAMP(x - 1, 0, current_width-1), 0, current_width)]
                 - 2 * curr_img[IDX3(CLAMP(y - 1, 0, height-1), x, 0, current_width)]
                 - 1 * curr_img[IDX3(CLAMP(y - 1, 0, height-1), CLAMP(x + 1, 0, current_width-1), 0, current_width)]
                 + 1 * curr_img[IDX3(CLAMP(y + 1, 0, height-1), CLAMP(x - 1, 0, current_width-1), 0, current_width)]
                 + 2 * curr_img[IDX3(CLAMP(y + 1, 0, height-1), x, 0, current_width)]
                 + 1 * curr_img[IDX3(CLAMP(y + 1, 0, height-1), CLAMP(x + 1, 0, current_width-1), 0, current_width)];

            e_1 = (int) (abs(gx) + abs(gy));

            // Entropy
            for (int i = 0; i < 9; ++i) bins[i] = 0;
            local_min = INT_MAX;
            local_max = INT_MIN;
            
            for (int v = -4; v < 4; ++v) {
                for (int u = -4; u < 4; ++u) {
                    unsigned char val = curr_img[IDX3(CLAMP(y + v, 0, height-1), CLAMP(x + u, 0, current_width-1), 0, current_width)];
                    local_min = MIN(local_min, val);
                    local_max = MAX(local_max, val);
                }
            }
            hist_width = local_max - local_min + 1;
            
            for (int v = -4; v < 4; ++v) {
                for (int u = -4; u < 4; ++u) {
                     unsigned char val = curr_img[IDX3(CLAMP(y + v, 0, height-1), CLAMP(x + u, 0, current_width-1), 0, current_width)];
                    int i = (val - local_min) * 9 / hist_width;
                    bins[i] += 1.0;
                }
            }

            e_entropy = 0;
            for (int i = 0; i < 9; ++i) {
                bins[i] /= 81.0;
                if (bins[i] > 0.0) {
                    e_entropy += (int) entrop(bins[i]);
                }
            }
            d_energy[IDX1(y, x, current_width)] = e_1 + e_entropy;
        }
    }

    // --- Step 2: Calculate MinEnergySums (On Device) ---
    // First row
    #pragma omp target is_device_ptr(d_energy, curr_sums)
    #pragma omp teams distribute parallel for
    for (int x = 0; x < current_width; ++x) {
        curr_sums[IDX1(0, x, current_width)] = d_energy[IDX1(0, x, current_width)];
    }

    // Other rows (Sequential in Y, Parallel in X)
    for (int y = 1; y < height; ++y) {
        #pragma omp target is_device_ptr(d_energy, curr_sums)
        #pragma omp teams distribute parallel for
        for (int x = 0; x < current_width; ++x) {
            unsigned int min_prev;
            if (x == current_width - 1) { 
                 min_prev = MIN(curr_sums[IDX1(y - 1, x - 1, current_width)], curr_sums[IDX1(y - 1, x, current_width)]);
            } else if (x == 0) { 
                 min_prev = MIN(curr_sums[IDX1(y - 1, x, current_width)], curr_sums[IDX1(y - 1, x + 1, current_width)]);
            } else {
                 min_prev = MIN(MIN(curr_sums[IDX1(y - 1, x - 1, current_width)], curr_sums[IDX1(y - 1, x, current_width)]), curr_sums[IDX1(y - 1, x + 1, current_width)]);
            }
            curr_sums[IDX1(y, x, current_width)] = d_energy[IDX1(y, x, current_width)] + min_prev;
        }
    }

    // --- Step 3: Find Seams (Initial Calculation on Host) ---
    // We need the last row of curr_sums to find start points
    unsigned int *last_row_sums = malloc(sizeof(unsigned int) * current_width);
    omp_target_memcpy(last_row_sums, 
                      &curr_sums[IDX1(height-1, 0, current_width)], 
                      sizeof(unsigned int) * current_width, 
                      0, 0, 
                      omp_get_initial_device(), omp_get_default_device());

    unsigned int mins[seams];
    // Find seams logic (CPU)
    for (int k = 0; k < seams; ++k) {
        mins[k] = current_width; 
        for (int j = 0; j < current_width; ++j) {
            int skip = 0;
            // Check if index j is already picked
            for (int l = 0; l < k; ++l) {
                if (mins[l] == j) {
                    skip = 1;
                    break;
                }
            }
            if (skip == 1) continue;

             if (mins[k] == current_width || last_row_sums[j] < last_row_sums[mins[k]]) {
                mins[k] = j;
            }
        }
    }
    // Handle bounds (unlikely but in original code)
    for (int k = 0; k < seams; ++k) {
        if (mins[k] >= current_width) mins[k] = current_width - 1;
    }
    selectionSort(mins, seams);
    free(last_row_sums);

    // --- Step 4: Expand Seams (On Device) ---
    for (int i = 0; i < seams; ++i) {
        int minIdx = mins[i];
        
        // Kernel A: Trace Path (1 thread)
        #pragma omp target is_device_ptr(curr_sums, d_seamPath)
        {
            // Replicate trace logic
            int x = minIdx;
            // Note: The loop goes Top -> Bottom? No, Bottom -> Top in original code logic for "x" update? 
            // Original: for (int y = height - 2; y >= 0; --y)
            // It starts at minIdx (at height-1 implicitly) and goes UP.
            
            d_seamPath[height-1] = minIdx; // Store start point

            for (int y = height - 2; y >= 0; --y) {
                 unsigned int min_val;
                 // Similar logic to find where min came from:
                 // "if (x > 0 && m1(y, x - 1) == min)" checks PREVIOUS row m1(y, ...)
                 // We look at row y neighbors to see which one was the parent of row y+1 pixel x.
                 
                if (x == 0) {
                    min_val = MIN(curr_sums[IDX1(y, x, current_width)], curr_sums[IDX1(y, x + 1, current_width)]);
                } else if (x == current_width - 1) {
                    min_val = MIN(curr_sums[IDX1(y, x - 1, current_width)], curr_sums[IDX1(y, x, current_width)]);
                } else {
                    min_val = MIN(MIN(curr_sums[IDX1(y, x - 1, current_width)], curr_sums[IDX1(y, x, current_width)]), curr_sums[IDX1(y, x + 1, current_width)]);
                }
                
                if (x > 0 && curr_sums[IDX1(y, x - 1, current_width)] == min_val) {
                    x = x - 1;
                } else if (x < current_width - 1 && curr_sums[IDX1(y, x + 1, current_width)] == min_val) {
                    x = x + 1;
                }
                d_seamPath[y] = x;
            }
        }

        int next_width = current_width + 1;

        // Kernel B: Expand (Parallel)
        #pragma omp target is_device_ptr(curr_img, next_img, curr_sums, next_sums, d_seamPath)
        #pragma omp teams distribute parallel for collapse(2)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < next_width; ++x) {
                int seamX = d_seamPath[y];
                
                // Logic for expanding at seamX
                // If x <= seamX: copy from x
                // If x == seamX + 1: duplicate seamX (copy from seamX again)
                // If x > seamX + 1: copy from x - 1
                
                int srcX;
                if (x <= seamX) srcX = x;
                else srcX = x - 1;
                
                // Copy Image Data
                next_img[IDX3(y, x, 0, next_width)] = curr_img[IDX3(y, srcX, 0, current_width)];
                next_img[IDX3(y, x, 1, next_width)] = curr_img[IDX3(y, srcX, 1, current_width)];
                next_img[IDX3(y, x, 2, next_width)] = curr_img[IDX3(y, srcX, 2, current_width)];
                
                // Copy Sums Data
                next_sums[IDX1(y, x, next_width)] = curr_sums[IDX1(y, srcX, current_width)];
            }
        }

        // Swap
        unsigned char *tmp_img = curr_img; curr_img = next_img; next_img = tmp_img;
        unsigned int *tmp_sums = curr_sums; curr_sums = next_sums; next_sums = tmp_sums;
        current_width++;
    }

    // Done. Download result.
    image->width = current_width;
    unsigned char *finalData = malloc(sizeof(unsigned char) * 3 * current_width * height);
    
    omp_target_memcpy(finalData, curr_img, sizeof(unsigned char) * 3 * current_width * height,
                      0, 0, omp_get_initial_device(), omp_get_default_device());

    // Free initial (old) data
    free(image->lpData);
    image->lpData = finalData;

    // Free device
    omp_target_free(d_img1, omp_get_default_device());
    omp_target_free(d_img2, omp_get_default_device());
    omp_target_free(d_energy, omp_get_default_device());
    omp_target_free(d_sums1, omp_get_default_device());
    omp_target_free(d_sums2, omp_get_default_device());
    omp_target_free(d_seamPath, omp_get_default_device());

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

    // Note: gray is now self-contained with device offloading
    input->lpData = gray(input);
    struct imgRawImage *output = increaseWidth(input, seams);

    clock_t end = clock();
    printf("Execution time: %4.2f sec\n", (double) ((double) (end - start) / CLOCKS_PER_SEC));
    storeJpegImageFile(output, outputFile);

    return 0;
}
