#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <time.h>
#include "image.h"

// Define min/max macros if not available
#ifndef min
#define min(a,b) (((a)<(b))?(a):(b))
#endif
#ifndef max
#define max(a,b) (((a)>(b))?(a):(b))
#endif

// --- Helper Functions ---

// Get pixel pointer from raw image data (RGB layout)
unsigned char* get_pixel(struct imgRawImage *img, int x, int y) {
    if (x < 0) x = 0;
    if (x >= img->width) x = img->width - 1;
    if (y < 0) y = 0;
    if (y >= img->height) y = img->height - 1;
    return &img->lpData[(y * img->width + x) * 3];
}

// Calculate Scharr gradient energy for the whole image
// Python reference: b_energy = abs(Scharr(b, -1, 1, 0)) + abs(Scharr(b, -1, 0, 1)) ... sum(channels)
double* calc_energy_map(struct imgRawImage *img) {
    int w = img->width;
    int h = img->height;
    double *energy = (double*)malloc(w * h * sizeof(double));

    // Scharr Kernels
    // Dx:         Dy:
    // -3  0  3    -3 -10 -3
    // -10 0 10     0   0  0
    // -3  0  3     3  10  3

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            double pix_energy = 0.0;

            // Calculate gradient for each channel (R, G, B)
            for (int c = 0; c < 3; c++) {
                double dx = 0.0;
                double dy = 0.0;

                // 3x3 Convolution
                // We clamp borders using get_pixel logic

                // Top Row
                unsigned char *p_tl = get_pixel(img, x-1, y-1);
                unsigned char *p_tm = get_pixel(img, x,   y-1);
                unsigned char *p_tr = get_pixel(img, x+1, y-1);

                // Middle Row
                unsigned char *p_ml = get_pixel(img, x-1, y);
                unsigned char *p_mr = get_pixel(img, x+1, y);

                // Bottom Row
                unsigned char *p_bl = get_pixel(img, x-1, y+1);
                unsigned char *p_bm = get_pixel(img, x,   y+1);
                unsigned char *p_br = get_pixel(img, x+1, y+1);

                // Apply Scharr X
                dx += -3.0 * p_tl[c] + 3.0 * p_tr[c];
                dx += -10.0 * p_ml[c] + 10.0 * p_mr[c];
                dx += -3.0 * p_bl[c] + 3.0 * p_br[c];

                // Apply Scharr Y
                dy += -3.0 * p_tl[c] - 10.0 * p_tm[c] - 3.0 * p_tr[c];
                dy +=  3.0 * p_bl[c] + 10.0 * p_bm[c] + 3.0 * p_br[c];

                pix_energy += fabs(dx) + fabs(dy);
            }
            energy[y * w + x] = pix_energy;
        }
    }
    return energy;
}

// Calculate Cumulative Map (Backward)
// M[y, x] = E[y, x] + min(M[y-1, x-1], M[y-1, x], M[y-1, x+1])
double* cumulative_map_backward(double *energy_map, int w, int h) {
    double *output = (double*)malloc(w * h * sizeof(double));

    // Copy first row
    memcpy(output, energy_map, w * sizeof(double));

    for (int row = 1; row < h; row++) {
        for (int col = 0; col < w; col++) {
            double e_curr = energy_map[row * w + col];

            double up = output[(row - 1) * w + col];
            double up_left = (col > 0) ? output[(row - 1) * w + (col - 1)] : DBL_MAX;
            double up_right = (col < w - 1) ? output[(row - 1) * w + (col + 1)] : DBL_MAX;

            output[row * w + col] = e_curr + min(up, min(up_left, up_right));
        }
    }
    return output;
}

// Find the seam (path of indices)
int* find_seam(double *cumulative_map, int w, int h) {
    int *seam = (int*)malloc(h * sizeof(int));

    // Find min index in last row
    int min_col = 0;
    double min_val = cumulative_map[(h - 1) * w + 0];

    for (int col = 1; col < w; col++) {
        if (cumulative_map[(h - 1) * w + col] < min_val) {
            min_val = cumulative_map[(h - 1) * w + col];
            min_col = col;
        }
    }
    seam[h - 1] = min_col;

    // Backtrack
    for (int row = h - 2; row >= 0; row--) {
        int prev_x = seam[row + 1];

        // Check 3 neighbors above
        int best_x = prev_x; // default to up
        double min_neighbor = cumulative_map[row * w + prev_x];

        // Check left parent
        if (prev_x > 0) {
            double val = cumulative_map[row * w + (prev_x - 1)];
            if (val < min_neighbor) {
                min_neighbor = val;
                best_x = prev_x - 1;
            }
        }

        // Check right parent
        if (prev_x < w - 1) {
            double val = cumulative_map[row * w + (prev_x + 1)];
            if (val < min_neighbor) {
                min_neighbor = val;
                best_x = prev_x + 1;
            }
        }

        seam[row] = best_x;
    }
    return seam;
}

// Remove vertical seam from image (modifies img struct)
void delete_seam(struct imgRawImage *img, int *seam) {
    int h = img->height;
    int w = img->width;
    int new_w = w - 1;

    unsigned char *new_data = (unsigned char*)malloc(new_w * h * 3);

    for (int row = 0; row < h; row++) {
        int s_col = seam[row];
        unsigned char *row_ptr = &img->lpData[row * w * 3];
        unsigned char *new_row_ptr = &new_data[row * new_w * 3];

        // Copy before seam
        if (s_col > 0) {
            memcpy(new_row_ptr, row_ptr, s_col * 3);
        }

        // Copy after seam
        if (s_col < w - 1) {
            memcpy(&new_row_ptr[s_col * 3], &row_ptr[(s_col + 1) * 3], (w - 1 - s_col) * 3);
        }
    }

    free(img->lpData);
    img->lpData = new_data;
    img->width = new_w;
}

// Add vertical seam to image (modifies img struct)
void add_seam(struct imgRawImage *img, int *seam) {
    int h = img->height;
    int w = img->width;
    int new_w = w + 1;

    unsigned char *new_data = (unsigned char*)malloc(new_w * h * 3);

    for (int row = 0; row < h; row++) {
        int col = seam[row];
        unsigned char *old_row = &img->lpData[row * w * 3];
        unsigned char *new_row = &new_data[row * new_w * 3];

        if (col == 0) {
            // Logic based on python: if col == 0, average current and next
            // p = average(col, col+1)
            // out[col] = old[col]
            // out[col+1] = p
            // out[col+1:] = old[col:]

            // 1. Copy old[col] to new[col]
            for(int c=0; c<3; c++) new_row[0*3+c] = old_row[0*3+c];

            // 2. Insert Average at new[col+1]
            for(int c=0; c<3; c++) {
                double val = (double)old_row[0*3+c];
                // Access safe check for next pixel
                if (w > 1) val += (double)old_row[1*3+c];
                else val += (double)old_row[0*3+c]; // fallback if width=1
                new_row[1*3+c] = (unsigned char)(val / 2.0);
            }

            // 3. Copy rest
            if (w > 1) {
                memcpy(&new_row[2 * 3], &old_row[1 * 3], (w - 1) * 3);
            }
        } else {
            // Logic based on python:
            // p = average(col-1, col+1) -- Wait, python slice is [col-1 : col+1], so it averages indices (col-1) and (col).
            // output[:col] = old[:col]
            // output[col] = p
            // output[col+1:] = old[col:]

            // 1. Copy up to col (exclusive) i.e. 0 to col-1
            memcpy(new_row, old_row, col * 3);

            // 2. Insert Average at new[col]
            // Average of old[col-1] and old[col]
            for (int c = 0; c < 3; c++) {
                double p1 = (double)old_row[(col - 1) * 3 + c];
                double p2 = (double)old_row[col * 3 + c];
                new_row[col * 3 + c] = (unsigned char)((p1 + p2) / 2.0);
            }

            // 3. Copy rest from old[col] to new[col+1]
            memcpy(&new_row[(col + 1) * 3], &old_row[col * 3], (w - col) * 3);
        }
    }

    free(img->lpData);
    img->lpData = new_data;
    img->width = new_w;
}

// Update remaining seams indices after an insertion
void update_seams(int **seams_record, int remaining_count, int *current_seam, int h) {
    for (int i = 0; i < remaining_count; i++) {
        int *s = seams_record[i];
        for (int row = 0; row < h; row++) {
            // Python: seam[np.where(seam >= current_seam)] += 2
            if (s[row] >= current_seam[row]) {
                s[row] += 2;
            }
        }
    }
}

// Deep copy of image struct
struct imgRawImage* copy_image(struct imgRawImage *src) {
    struct imgRawImage *dst = (struct imgRawImage*)malloc(sizeof(struct imgRawImage));
    dst->width = src->width;
    dst->height = src->height;
    dst->numComponents = src->numComponents;
    unsigned long bytes = src->width * src->height * 3;
    dst->lpData = (unsigned char*)malloc(bytes);
    memcpy(dst->lpData, src->lpData, bytes);
    return dst;
}

// Free image struct
void free_image(struct imgRawImage *img) {
    if (img) {
        if (img->lpData) free(img->lpData);
        free(img);
    }
}

// --- Main Seam Insertion Logic ---
void enlarge_image(struct imgRawImage *img, int k) {
    int h = img->height;

    // 1. Create temp image for deletion phase
    struct imgRawImage *temp_img = copy_image(img);

    // Array to store the k seams found
    int **seams_record = (int**)malloc(k * sizeof(int*));

    printf("Finding %d seams to duplicate...\n", k);

    // 2. Find k seams
    for (int i = 0; i < k; i++) {
        double *energy = calc_energy_map(temp_img);
        double *cumulative = cumulative_map_backward(energy, temp_img->width, temp_img->height);
        int *seam = find_seam(cumulative, temp_img->width, temp_img->height);

        seams_record[i] = seam; // Store seam

        // Remove from temp to find next distinct seam
        delete_seam(temp_img, seam);

        free(energy);
        free(cumulative);
    }

    // Done with temp image
    free_image(temp_img);

    printf("Inserting seams...\n");

    // 3. Insert seams into original image
    // Iterate through recorded seams
    // Note: In python, they use `pop(0)`. So we iterate i from 0 to k-1.
    for (int i = 0; i < k; i++) {
        int *current_seam = seams_record[i];

        add_seam(img, current_seam);

        // Update remaining seams in the record
        // The remaining seams are from i+1 to k-1
        int remaining_count = k - 1 - i;
        if (remaining_count > 0) {
            update_seams(&seams_record[i+1], remaining_count, current_seam, h);
        }

        free(current_seam);
    }

    free(seams_record);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <input.jpg> <output.jpg> <k>\n", argv[0]);
        return 1;
    }

    char *inputFile = argv[1];
    char *outputFile = argv[2];
    int k = atoi(argv[3]);

    if (k <= 0) {
        printf("Error: k must be a positive integer.\n");
        return 1;
    }

    // Load Image
    struct imgRawImage *img = loadJpegImageFile(inputFile);
    if (!img) {
        fprintf(stderr, "Error loading image %s\n", inputFile);
        return 1;
    }

    // Process
    clock_t start = clock();
    enlarge_image(img, k);
    clock_t end = clock();

    double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Processing time: %f seconds\n", cpu_time_used);

    // Save Image
    if (storeJpegImageFile(img, outputFile) != 0) {
        fprintf(stderr, "Error saving image %s\n", outputFile);
        free_image(img);
        return 1;
    }

    printf("Image saved to %s (expanded by %d pixels)\n", outputFile, k);

    free_image(img);
    return 0;
}
