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
#define m1(y, x) minEnergySums[(y)*width+(x)]
#define d1(y, x) data[(y)*width+(x)]
#define o(y, x) output[(y)*width+(x)]
#define MIN(a, b) (((a)<(b))?(a):(b))
#define MAX(a, b) (((a)>(b))?(a):(b))
#define ABS(x) (((x)<0)?-(x):(x))

#define entrop(p) (-1.0 * log2((p)) * (p) * (CHAR_MAX / 5.0 * 3.2))

#define TORUS_Y(y, height) (((y) + (height)) % (height))
#define TORUS_X(x, width) (((x) + (width)) % (width))


unsigned char *gray(struct imgRawImage *image) {
    unsigned int width = image->width;
    unsigned int height = image->height;
    unsigned char *output = malloc(sizeof(unsigned char) * 3 * width * height);
    unsigned char *input = image->lpData;

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

    #pragma omp target teams distribute parallel for \
    map(to: data[0:width*height]) \
    map(from: output[0:width*height])
    for (int x = 0; x < width; ++x) {
        output[x] = data[x];
    }

    #pragma omp target data map(to: data[0:width*height]) \
    map(tofrom: output[0:width*height])
    {
        for (int y = 1; y < height; ++y) {
            #pragma omp target teams distribute parallel for
            for (int x = 0; x < width; ++x) {
                unsigned int min_val;
                int idx = y * width + x;
                int idx_prev = (y - 1) * width + x;

                if (x == width - 1) {
                    min_val = MIN(output[idx_prev - 1], output[idx_prev]);
                } else if (x == 0) {
                    min_val = MIN(output[idx_prev], output[idx_prev + 1]);
                } else {
                    min_val = MIN(MIN(output[idx_prev - 1], output[idx_prev]),
                                  output[idx_prev + 1]);
                }
                output[idx] = data[idx] + min_val;
            }
        }
    }

    return output;
}

unsigned int *calculateEnergySobel(struct imgRawImage *image) {
    unsigned int width = image->width;
    unsigned int height = image->height;
    unsigned int *output = malloc(sizeof(unsigned int) * height * width);
    unsigned char *input = image->lpData;

    #pragma omp target teams distribute parallel for collapse(2) \
    map(to: input[0:width*height*3]) \
    map(from: output[0:width*height])
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int gx, gy, e_1, local_min, local_max, hist_width, e_entropy;
            double bins[9];

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

            gy = -1 * input[(y_m1 * width + x_m1) * 3 + 0]
            - 2 * input[(y_m1 * width + x) * 3 + 0]
            - 1 * input[(y_m1 * width + x_p1) * 3 + 0]
            + 1 * input[(y_p1 * width + x_m1) * 3 + 0]
            + 2 * input[(y_p1 * width + x) * 3 + 0]
            + 1 * input[(y_p1 * width + x_p1) * 3 + 0];

            e_1 = ABS(gx) + ABS(gy);

            for (int i = 0; i < 9; ++i) {
                bins[i] = 0;
            }
            local_min = INT_MAX;
            local_max = INT_MIN;
            e_entropy = 0;

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

            for (int v = -4; v < 4; ++v) {
                for (int u = -4; u < 4; ++u) {
                    int y_idx = TORUS_Y(y + v, height);
                    int x_idx = TORUS_X(x + u, width);
                    int pixel_val = input[(y_idx * width + x_idx) * 3 + 0];
                    int i = (pixel_val - local_min) * 9 / hist_width;
                    bins[i] += 1.0;
                }
            }

            for (int i = 0; i < 9; ++i) {
                bins[i] /= 81.0;
                if (bins[i] > 0.0) {
                    e_entropy += (int) entrop(bins[i]);
                }
            }

            output[y * width + x] = e_1 + e_entropy;
        }
    }
    return output;
}

// OPTIMIZED: Find all k seams at once (deterministic)
void findAllSeams(unsigned int *minEnergySums, int width, int height,
                  int k, int *seamStartIndices) {
    // Find k minimum positions in bottom row (deterministic order)
    for (int seam_id = 0; seam_id < k; ++seam_id) {
        int min_x = -1;
        unsigned int min_energy = UINT_MAX;

        for (int x = 0; x < width; ++x) {
            // Check if already selected
            int already_selected = 0;
            for (int prev = 0; prev < seam_id; ++prev) {
                if (seamStartIndices[prev] == x) {
                    already_selected = 1;
                    break;
                }
            }
            if (already_selected) continue;

            unsigned int energy = minEnergySums[(height - 1) * width + x];
            // Deterministic tie-breaking: prefer lower x
            if (energy < min_energy || (energy == min_energy && (min_x == -1 || x < min_x))) {
                min_energy = energy;
                min_x = x;
            }
        }

        seamStartIndices[seam_id] = min_x;
    }

    // Sort for deterministic insertion order
    for (int i = 0; i < k - 1; ++i) {
        for (int j = i + 1; j < k; ++j) {
            if (seamStartIndices[i] > seamStartIndices[j]) {
                int temp = seamStartIndices[i];
                seamStartIndices[i] = seamStartIndices[j];
                seamStartIndices[j] = temp;
            }
        }
    }
                  }

                  // OPTIMIZED: Trace all seams in parallel on GPU
                  void traceAllSeams(unsigned int *minEnergySums, int width, int height,
                                     int k, int *seamStartIndices, int *seamPaths) {
                      // seamPaths[seam_id * height + y] = x position at row y

                      #pragma omp target teams distribute parallel for \
                      map(to: minEnergySums[0:width*height], seamStartIndices[0:k]) \
                      map(from: seamPaths[0:k*height])
                      for (int seam_id = 0; seam_id < k; ++seam_id) {
                          int x = seamStartIndices[seam_id];
                          seamPaths[seam_id * height + (height - 1)] = x;

                          // Trace upward
                          for (int y = height - 2; y >= 0; --y) {
                              unsigned int min_val;
                              int next_x = x;

                              if (x == 0) {
                                  unsigned int val_0 = minEnergySums[y * width + x];
                                  unsigned int val_1 = minEnergySums[y * width + x + 1];
                                  min_val = MIN(val_0, val_1);
                                  if (val_1 == min_val) next_x = x + 1;
                              } else if (x == width - 1) {
                                  unsigned int val_m1 = minEnergySums[y * width + x - 1];
                                  unsigned int val_0 = minEnergySums[y * width + x];
                                  min_val = MIN(val_m1, val_0);
                                  if (val_m1 == min_val) next_x = x - 1;
                              } else {
                                  unsigned int val_m1 = minEnergySums[y * width + x - 1];
                                  unsigned int val_0 = minEnergySums[y * width + x];
                                  unsigned int val_p1 = minEnergySums[y * width + x + 1];
                                  min_val = MIN(MIN(val_m1, val_0), val_p1);

                                  // Deterministic tie-breaking: prefer left, then center, then right
                                  if (val_m1 == min_val) next_x = x - 1;
                                  else if (val_p1 == min_val) next_x = x + 1;
                              }

                              x = next_x;
                              seamPaths[seam_id * height + y] = x;
                          }
                      }
                                     }

                                     // OPTIMIZED: Insert all seams at once (parallel per row)
                                     struct imgRawImage *insertAllSeams(struct imgRawImage *image, int k, int *seamPaths) {
                                         int height = image->height;
                                         int width = image->width;
                                         int new_width = width + k;

                                         unsigned char *oldData = image->lpData;
                                         unsigned char *newData = malloc(sizeof(unsigned char) * 3 * new_width * height);

                                         // Parallel insertion: each row independently
                                         #pragma omp target teams distribute parallel for \
                                         map(to: oldData[0:width*height*3], seamPaths[0:k*height]) \
                                         map(from: newData[0:new_width*height*3])
                                         for (int y = 0; y < height; ++y) {
                                             // Collect seam positions for this row (sorted)
                                             int seam_positions[k];
                                             for (int s = 0; s < k; ++s) {
                                                 seam_positions[s] = seamPaths[s * height + y];
                                             }

                                             // Sort seam positions for this row (bubble sort is fine for small k)
                                             for (int i = 0; i < k - 1; ++i) {
                                                 for (int j = 0; j < k - i - 1; ++j) {
                                                     if (seam_positions[j] > seam_positions[j + 1]) {
                                                         int temp = seam_positions[j];
                                                         seam_positions[j] = seam_positions[j + 1];
                                                         seam_positions[j + 1] = temp;
                                                     }
                                                 }
                                             }

                                             int write_idx = 0;
                                             int seam_idx = 0;

                                             for (int x = 0; x < width; ++x) {
                                                 // Copy original pixel
                                                 newData[(y * new_width + write_idx) * 3 + 0] = oldData[(y * width + x) * 3 + 0];
                                                 newData[(y * new_width + write_idx) * 3 + 1] = oldData[(y * width + x) * 3 + 1];
                                                 newData[(y * new_width + write_idx) * 3 + 2] = oldData[(y * width + x) * 3 + 2];
                                                 write_idx++;

                                                 // Check if this is a seam position - if so, duplicate
                                                 while (seam_idx < k && seam_positions[seam_idx] == x) {
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

                                     // OPTIMIZED: Single-pass seam carving
                                     struct imgRawImage *increaseWidth(struct imgRawImage *image, int seams) {
                                         int height = image->height;
                                         int width = image->width;

                                         // PHASE 1: Calculate energy ONCE (on GPU)
                                         printf("Phase 1: Calculating energy...\n");
                                         unsigned int *pixelEnergies = calculateEnergySobel(image);
                                         unsigned int *minEnergySums = calculateMinEnergySums(pixelEnergies, width, height);
                                         free(pixelEnergies);

                                         // PHASE 2: Find all seams at once (deterministic)
                                         printf("Phase 2: Finding %d seams...\n", seams);
                                         int *seamStartIndices = malloc(sizeof(int) * seams);
                                         findAllSeams(minEnergySums, width, height, seams, seamStartIndices);

                                         // PHASE 3: Trace all seams in parallel (on GPU)
                                         printf("Phase 3: Tracing seams...\n");
                                         int *seamPaths = malloc(sizeof(int) * seams * height);
                                         traceAllSeams(minEnergySums, width, height, seams, seamStartIndices, seamPaths);

                                         free(minEnergySums);
                                         free(seamStartIndices);

                                         // PHASE 4: Insert all seams at once (parallel per row)
                                         printf("Phase 4: Inserting seams...\n");
                                         insertAllSeams(image, seams, seamPaths);

                                         free(seamPaths);

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
