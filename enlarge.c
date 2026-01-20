#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <omp.h>
#include "image.h"

// Define macros for host/device code compatibility
// We will redefine these locally in functions where necessary or ensure variables match
#define d3_ptr(ptr, w, y, x, z) ptr[(y)*(w)*3+(x)*3+(z)]
#define o3_ptr(ptr, w, y, x, z) ptr[(y)*(w)*3+(x)*3+(z)]
// Old macros for compatibility with existing code structure (will be shadowed or replaced)
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

// Note: Since the gradient isn't normalized,
// we rescale the summands in the entropy calculations slightly,
#define entrop(p) (-1.0 * log2((p)) * (p) * (CHAR_MAX / 5.0 * 3.2)) 

unsigned char *gray(struct imgRawImage *image) {
    unsigned int width = image->width;
    unsigned int height = image->height;
    unsigned int size = 3 * width * height;
    
    // Allocate output on device
    unsigned char *output = (unsigned char *)omp_target_alloc(size * sizeof(unsigned char), omp_get_default_device());

    unsigned char *inputData = image->lpData;

    #pragma omp target teams distribute parallel for is_device_ptr(output) map(to: inputData[0:size])
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            unsigned char luma = (unsigned char) (
                    0.299f * (float) d3_ptr(inputData, width, y, x, 0)
                    + 0.587f * (float) d3_ptr(inputData, width, y, x, 1)
                    + 0.114f * (float) d3_ptr(inputData, width, y, x, 2));
            o3_ptr(output, width, y, x, 0) = luma;
            o3_ptr(output, width, y, x, 1) = luma;
            o3_ptr(output, width, y, x, 2) = luma;
        }
    }
    // We return the device pointer. 
    // Note: The input 'image->lpData' was Host memory, but now we return Device memory.
    // The calling function must handle this.
    return output;
}

unsigned int *calculateMinEnergySums(unsigned int *data, int width, int height) {
    unsigned int *output = (unsigned int *)omp_target_alloc(sizeof(unsigned int) * width * height, omp_get_default_device());
    
    // row 0
    #pragma omp target teams distribute parallel for is_device_ptr(output, data)
    for (int x = 0; x < width; ++x) {
        output[x] = data[x]; // d1(0, x)
    }

    // Wavefront
    for (int y = 1; y < height; ++y) {
        #pragma omp target teams distribute parallel for is_device_ptr(output, data)
        for (int x = 0; x < width; ++x) {
            unsigned int val = data[y*width+x];
            unsigned int min_prev;
            if (x == width - 1) { // rightmost pixel of a row
                unsigned int a = output[(y-1)*width + (x-1)];
                unsigned int b = output[(y-1)*width + x];
                min_prev = MIN(a, b);
            } else if (x == 0) { // leftmost pixel of a row
                unsigned int b = output[(y-1)*width + x];
                unsigned int c = output[(y-1)*width + (x+1)];
                min_prev = MIN(b, c);
            } else {
                unsigned int a = output[(y-1)*width + (x-1)];
                unsigned int b = output[(y-1)*width + x];
                unsigned int c = output[(y-1)*width + (x+1)];
                min_prev = MIN(MIN(a, b), c);
            }
            output[y*width+x] = val + min_prev;
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
    unsigned int *output = (unsigned int *)omp_target_alloc(sizeof(unsigned int) * height * width, omp_get_default_device());
    
    unsigned char *lpData = image->lpData; // This is a Device Pointer now

    #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(output, lpData)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int gx, gy, e_1, local_min, local_max, hist_width, e_entropy;
            double bins[9];

            // Access check/padding logic? Original code access y-1 directly.
            // Assuming image bounds are valid or we replicate original unsafe access.
            // Using d3_ptr macro
            
            // apply Sobel operator in X direction
            gx = -1 * d3_ptr(lpData, width, y - 1, x - 1, 0)
                 + 1 * d3_ptr(lpData, width, y - 1, x + 1, 0)
                 - 2 * d3_ptr(lpData, width, y, x - 1, 0)
                 + 2 * d3_ptr(lpData, width, y, x + 1, 0)
                 - 1 * d3_ptr(lpData, width, y + 1, x - 1, 0)
                 + 1 * d3_ptr(lpData, width, y + 1, x + 1, 0);

            // apply Sobel operator in Y direction
            gy = -1 * d3_ptr(lpData, width, y - 1, x - 1, 0)
                 - 2 * d3_ptr(lpData, width, y - 1, x, 0)
                 - 1 * d3_ptr(lpData, width, y - 1, x + 1, 0)
                 + 1 * d3_ptr(lpData, width, y + 1, x - 1, 0)
                 + 2 * d3_ptr(lpData, width, y + 1, x, 0)
                 + 1 * d3_ptr(lpData, width, y + 1, x + 1, 0);

            e_1 = (int) (abs(gx) + abs(gy));

            // Entropy
            for (int i = 0; i < 9; ++i) bins[i] = 0;
            local_min = INT_MAX;
            local_max = INT_MIN;
            
            for (int v = -4; v < 4; ++v) {
              for (int u = -4; u < 4; ++u) {
                unsigned char val = d3_ptr(lpData, width, y + v, x + u, 0);
                local_min = MIN(local_min, val);
                local_max = MAX(local_max, val);
              }
            }
            hist_width = local_max - local_min + 1;
            
            for (int v = -4; v < 4; ++v) {
              for (int u = -4; u < 4; ++u) {
                unsigned char val = d3_ptr(lpData, width, y + v, x + u, 0);
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

            output[y*width+x] = e_1 + e_entropy;
        }
    }
    return output;
}

// increases the number of columns by cols
struct imgRawImage *increaseWidth(struct imgRawImage *image, int seams) {
    int height = image->height;
    int width = image->width; // initial width
    
    // image->lpData is already Device pointer from gray()
    
    // Calculate Energies
    unsigned int *pixelEnergies = calculateEnergySobel(image);
    unsigned int *minEnergySums = calculateMinEnergySums(pixelEnergies, width, height);
    omp_target_free(pixelEnergies, omp_get_default_device());

    // START calculate MINS on HOST
    // We need the last row of minEnergySums on Host to find seams
    unsigned int *hostLastRow = malloc(sizeof(unsigned int) * width);
    omp_target_memcpy(hostLastRow, minEnergySums + (height-1)*width, sizeof(unsigned int) * width, 0, 0, omp_get_initial_device(), omp_get_default_device());

    unsigned int mins[seams];
    // Copy the implementation of 'm1(height-1, j)' access using hostLastRow
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
            
            if (mins[k] == width || hostLastRow[j] < hostLastRow[mins[k]]) {
                mins[k] = j;
            }
        }
    }
    
    // Adjust for out of bounds? Original code: 
    for (int k = 0; k < seams; ++k) {
        if (mins[k] >= width) {
            mins[k] = width - 1;
        }
    }
    selectionSort(mins, seams);
    free(hostLastRow);
    // END calculate MINS

    // Path buffer on device
    int *d_path = (int *)omp_target_alloc(sizeof(int) * height, omp_get_default_device());

    for (int i = 0; i < seams; ++i) {
        unsigned int minIdx = mins[i];
        int current_width = image->width;
        
        unsigned int *newMinEnergySums = (unsigned int *)omp_target_alloc(sizeof(unsigned int) * (current_width + 1) * height, omp_get_default_device());
        unsigned char *newData = (unsigned char *)omp_target_alloc(sizeof(unsigned char) * 3 * (current_width + 1) * height, omp_get_default_device());
        unsigned char *oldData = image->lpData;
        
        // Kernel 1: Trace seam (Serial on GPU)
        // Access minEnergySums (m1).
        #pragma omp target is_device_ptr(minEnergySums, d_path)
        {
            // Trace from bottom to top
            int x = minIdx; 
            // Note: minIdx is from original mins. 
            // NOTE: In the original code, 'm1' during logic is accessed from 'minEnergySums'.
            // The logic:
            // x starts at minIdx.
            // Loop y down.
            // Use m1(y, x) etc to find min.
            
            for (int y = height - 1; y >= 0; --y) {
                 d_path[y] = x;
                 
                 if (y > 0) { // Calculate next x for y-1
                     // Look at row y-1?
                     // Original code loop: for (int y = height - 2; y >= 0; --y)
                     // And it decides x for the *next* iteration (going up) based on current x?
                     // No, finding min of neighbors at y.
                     // Original code:
                     // int x = minIdx;
                     // ... copy bottom row ...
                     // for (y = height - 2 ... ) {
                     //    min = MIN(m1(y, x-1), m1(y, x), m1(y, x+1))
                     //    update x
                     // }
                     
                     // We need to replicate this update for 'x'.
                     // Access row y-1? No, original accesses `m1(y, ...)` inside loop where y goes height-2 down.
                     // So it accesses row `y` to decide move from `y+1` to `y`?
                     // Wait, the Seam is a path.
                     // The pixel at (y+1, x_next) is connected to (y, x_current).
                     // We are at y. We want to find best parent at y.
                     // The loop is:
                     /*
                        for (int y = height - 2; y >= 0; --y) {
                            // check neighbors at y
                            if (x==0) min = MIN(m1(y,x), m1(y,x+1)) ...
                            // update x to new min pos
                        }
                     */
                    // Yes, so we look at row y (next row to be processed going up).
                    
                    int target_y = y - 1; 
                    unsigned int min_val;
                    unsigned int v_l, v_m, v_r;
                    // Access m1 at target_y.
                    // Pointer math: minEnergySums[target_y * current_width + ...]
                    
                    if (x == 0) {
                        v_m = minEnergySums[target_y * current_width + x];
                        v_r = minEnergySums[target_y * current_width + x + 1];
                        min_val = MIN(v_m, v_r);
                    } else if (x == current_width - 1) {
                         v_l = minEnergySums[target_y * current_width + x - 1];
                         v_m = minEnergySums[target_y * current_width + x];
                         min_val = MIN(v_l, v_m);
                    } else {
                         v_l = minEnergySums[target_y * current_width + x - 1];
                         v_m = minEnergySums[target_y * current_width + x];
                         v_r = minEnergySums[target_y * current_width + x + 1];
                         min_val = MIN(v_l, MIN(v_m, v_r));
                    }
                    
                    if (x > 0 && minEnergySums[target_y * current_width + x - 1] == min_val) {
                        x = x - 1;
                    } else if (x < current_width - 1 && minEnergySums[target_y * current_width + x + 1] == min_val) {
                        x = x + 1;
                    }
                    // x is now updated for the *next* iteration (which is target_y, i.e., y-1)
                    // We will store it in the next loop iter d_path[target_y] = x;
                 }
            }
        }

        // Kernel 2: Copy and Enlarge (Parallel)
        #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(oldData, minEnergySums, newData, newMinEnergySums, d_path)
        for (int y = 0; y < height; ++y) {
            for (int j = 0; j <= current_width; ++j) {
                // Determine seam x for this row
                int seam_x = d_path[y];
                
                // Pointers for this row
                // old row: y * current_width
                // new row: y * (current_width + 1)
                
                if (j <= seam_x) {
                    // Copy as is
                    unsigned int val = minEnergySums[y * current_width + j];
                    newMinEnergySums[y * (current_width+1) + j] = val;
                    newData[y * (current_width+1) * 3 + j*3 + 0] = oldData[y * current_width * 3 + j*3 + 0];
                    newData[y * (current_width+1) * 3 + j*3 + 1] = oldData[y * current_width * 3 + j*3 + 1];
                    newData[y * (current_width+1) * 3 + j*3 + 2] = oldData[y * current_width * 3 + j*3 + 2];
                } else if (j == seam_x + 1) {
                    // Inserted pixel at seam_x + 1
                    // Copy from seam_x (duplication)
                    unsigned int val = minEnergySums[y * current_width + seam_x];
                    newMinEnergySums[y * (current_width+1) + j] = val;
                    newData[y * (current_width+1) * 3 + j*3 + 0] = oldData[y * current_width * 3 + seam_x*3 + 0];
                    newData[y * (current_width+1) * 3 + j*3 + 1] = oldData[y * current_width * 3 + seam_x*3 + 1];
                    newData[y * (current_width+1) * 3 + j*3 + 2] = oldData[y * current_width * 3 + seam_x*3 + 2];
                } else {
                    // j > seam_x + 1. Original index was j - 1
                    unsigned int val = minEnergySums[y * current_width + (j-1)];
                    newMinEnergySums[y * (current_width+1) + j] = val;
                    newData[y * (current_width+1) * 3 + j*3 + 0] = oldData[y * current_width * 3 + (j-1)*3 + 0];
                    newData[y * (current_width+1) * 3 + j*3 + 1] = oldData[y * current_width * 3 + (j-1)*3 + 1];
                    newData[y * (current_width+1) * 3 + j*3 + 2] = oldData[y * current_width * 3 + (j-1)*3 + 2];
                }
            }
        }
        
        // Free old
        omp_target_free(oldData, omp_get_default_device());
        omp_target_free(minEnergySums, omp_get_default_device());
        
        image->lpData = newData; // Update wrapper (still device ptr)
        image->width = current_width + 1;
        minEnergySums = newMinEnergySums;
    }
    
    omp_target_free(d_path, omp_get_default_device());
    omp_target_free(minEnergySums, omp_get_default_device());

    // Final Copy Back to Host
    unsigned int final_size = image->width * image->height * 3;
    unsigned char *hostData = malloc(final_size);
    omp_target_memcpy(hostData, image->lpData, final_size, 0, 0, omp_get_initial_device(), omp_get_default_device());
    omp_target_free(image->lpData, omp_get_default_device());
    
    image->lpData = hostData;
    
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
    
    // gray now returns a Device Pointer!
    // We update input->lpData to point to Device Memory.
    // The previous Host Memory from loadJpegImageFile should be freed or ignored (leaked if not freed, but loadJpeg likely malloced it)
    // To be clean:
    // We can't easily free it without keeping the pointer before overwrite, but strictly speaking checking image->lpData
    // 'gray' function takes 'input' struct.
    
    unsigned char *host_input_ptr = input->lpData;
    input->lpData = gray(input);
    // free(host_input_ptr); // This is likely safe to free now as gray copied it to device.
    
    // Now input->lpData is Device Pointer.
    struct imgRawImage *output = increaseWidth(input, seams);
    // output->lpData is now Host Pointer again (converted back at end of increaseWidth).

    clock_t end = clock();
    printf("Execution time: %4.2f sec\n", (double) ((double) (end - start) / CLOCKS_PER_SEC));
    storeJpegImageFile(output, outputFile);

    return 0;
}
