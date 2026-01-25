#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <string.h>
#include "image.h"

// Define macros for host/device code compatibility
#define d3_ptr(ptr, w, y, x, z) ptr[(y) * (w) * 3 + (x) * 3 + (z)]
#define o3_ptr(ptr, w, y, x, z) ptr[(y) * (w) * 3 + (x) * 3 + (z)]

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

// Note: Since the gradient isn't normalized,
// we rescale the summands in the entropy calculations slightly.
// CHANGED: Using log2f and float constants for GPU efficiency
#define entrop(p) (-1.0f * log2f((p)) * (p) * (127.0f / 5.0f * 3.2f))

unsigned char *gray(struct imgRawImage *image)
{
    unsigned int width = image->width;
    unsigned int height = image->height;
    unsigned int size = 3 * width * height;

    // Allocate output
    unsigned char *output = (unsigned char *)malloc(size * sizeof(unsigned char));

    unsigned char *inputData = image->lpData;

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            unsigned char luma = (unsigned char)(0.299f * (float)d3_ptr(inputData, width, y, x, 0) + 0.587f * (float)d3_ptr(inputData, width, y, x, 1) + 0.114f * (float)d3_ptr(inputData, width, y, x, 2));
            o3_ptr(output, width, y, x, 0) = luma;
            o3_ptr(output, width, y, x, 1) = luma;
            o3_ptr(output, width, y, x, 2) = luma;
        }
    }
    return output;
}

unsigned int *calculateMinEnergySums(unsigned int *data, int width, int height)
{
    unsigned int *output = (unsigned int *)malloc(sizeof(unsigned int) * width * height);

    // row 0
    for (int x = 0; x < width; ++x)
    {
        output[x] = data[x]; // d1(0, x)
    }

    // Wavefront
    for (int y = 1; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            unsigned int val = data[y * width + x];
            unsigned int min_prev;
            if (x == width - 1)
            { // rightmost pixel of a row
                unsigned int a = output[(y - 1) * width + (x - 1)];
                unsigned int b = output[(y - 1) * width + x];
                min_prev = MIN(a, b);
            }
            else if (x == 0)
            { // leftmost pixel of a row
                unsigned int b = output[(y - 1) * width + x];
                unsigned int c = output[(y - 1) * width + (x + 1)];
                min_prev = MIN(b, c);
            }
            else
            {
                unsigned int a = output[(y - 1) * width + (x - 1)];
                unsigned int b = output[(y - 1) * width + x];
                unsigned int c = output[(y - 1) * width + (x + 1)];
                min_prev = MIN(MIN(a, b), c);
            }
            output[y * width + x] = val + min_prev;
        }
    }
    return output;
}

void swap(unsigned int *xp, unsigned int *yp)
{
    unsigned int temp = *xp;
    *xp = *yp;
    *yp = temp;
}

void selectionSort(unsigned int arr[], int n)
{
    int i, j, max_idx;
    for (i = 0; i < n - 1; i++)
    {
        max_idx = i;
        for (j = i + 1; j < n; j++)
            if (arr[j] > arr[max_idx])
                max_idx = j;
        swap(&arr[max_idx], &arr[i]);
    }
}

unsigned int *calculateEnergySobel(struct imgRawImage *image)
{
    unsigned int width = image->width;
    unsigned int height = image->height;
    unsigned int *output = (unsigned int *)malloc(sizeof(unsigned int) * height * width);

    unsigned char *lpData = image->lpData;

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            // Guard against edge cases for Sobel kernel
            if (y <= 0 || y >= height - 1 || x <= 0 || x >= width - 1)
            {
                output[y * width + x] = 0;
                continue;
            }

            int gx, gy, e_1, local_min, local_max, hist_width, e_entropy;
            // CHANGED: Float is preferred on GPU
            float bins[9];

            gx = -1 * d3_ptr(lpData, width, y - 1, x - 1, 0) + 1 * d3_ptr(lpData, width, y - 1, x + 1, 0) - 2 * d3_ptr(lpData, width, y, x - 1, 0) + 2 * d3_ptr(lpData, width, y, x + 1, 0) - 1 * d3_ptr(lpData, width, y + 1, x - 1, 0) + 1 * d3_ptr(lpData, width, y + 1, x + 1, 0);

            gy = -1 * d3_ptr(lpData, width, y - 1, x - 1, 0) - 2 * d3_ptr(lpData, width, y - 1, x, 0) - 1 * d3_ptr(lpData, width, y - 1, x + 1, 0) + 1 * d3_ptr(lpData, width, y + 1, x - 1, 0) + 2 * d3_ptr(lpData, width, y + 1, x, 0) + 1 * d3_ptr(lpData, width, y + 1, x + 1, 0);

            // CHANGED: Use __builtin_abs to solve linker error
            e_1 = (int)(__builtin_abs(gx) + __builtin_abs(gy));

            for (int i = 0; i < 9; ++i)
                bins[i] = 0.0f;
            local_min = 255;
            local_max = 0;

            // Entropy calculation window
            for (int v = -4; v <= 4; ++v)
            {
                if (y + v < 0 || y + v >= height)
                    continue;
                for (int u = -4; u <= 4; ++u)
                {
                    if (x + u < 0 || x + u >= width)
                        continue;
                    unsigned char val = d3_ptr(lpData, width, y + v, x + u, 0);
                    local_min = MIN(local_min, val);
                    local_max = MAX(local_max, val);
                }
            }
            hist_width = local_max - local_min + 1;

            for (int v = -4; v <= 4; ++v)
            {
                if (y + v < 0 || y + v >= height)
                    continue;
                for (int u = -4; u <= 4; ++u)
                {
                    if (x + u < 0 || x + u >= width)
                        continue;
                    unsigned char val = d3_ptr(lpData, width, y + v, x + u, 0);
                    int i = (val - local_min) * 8 / (hist_width > 0 ? hist_width : 1);
                    bins[i] += 1.0f;
                }
            }

            e_entropy = 0;
            for (int i = 0; i < 9; ++i)
            {
                bins[i] /= 81.0f;
                if (bins[i] > 0.0f)
                {
                    e_entropy += (int)entrop(bins[i]);
                }
            }

            output[y * width + x] = e_1 + e_entropy;
        }
    }
    return output;
}

// increases the number of columns by cols
struct imgRawImage *increaseWidth(struct imgRawImage *image, int seams)
{
    int height = image->height;
    int width = image->width; // initial width

    // image->lpData is already pointer from gray()

    // Calculate Energies
    unsigned int *pixelEnergies = calculateEnergySobel(image);
    unsigned int *minEnergySums = calculateMinEnergySums(pixelEnergies, width, height);
    free(pixelEnergies);

    // START calculate MINS on HOST
    // We need the last row of minEnergySums to find seams
    unsigned int *hostLastRow = (unsigned int *)malloc(sizeof(unsigned int) * width);
    memcpy(hostLastRow, minEnergySums + (height - 1) * width, sizeof(unsigned int) * width);

    unsigned int *mins = (unsigned int *)malloc(sizeof(unsigned int) * seams);
    // Copy the implementation of 'm1(height-1, j)' access using hostLastRow
    for (int k = 0; k < seams; ++k)
    {
        mins[k] = width;
        for (int j = 0; j < width; ++j)
        {
            int skip = 0;
            for (int l = 0; l < k; ++l)
            {
                if (mins[l] == j)
                {
                    skip = 1;
                    break;
                }
            }
            if (skip == 1)
                continue;

            if (mins[k] == width || hostLastRow[j] < hostLastRow[mins[k]])
            {
                mins[k] = j;
            }
        }
    }

    // Adjust for out of bounds
    for (int k = 0; k < seams; ++k)
    {
        if (mins[k] >= width)
        {
            mins[k] = width - 1;
        }
    }
    selectionSort(mins, seams);
    free(hostLastRow);
    // END calculate MINS

    // Path buffer
    int *d_path = (int *)malloc(sizeof(int) * height);

    for (int i = 0; i < seams; ++i)
    {
        unsigned int minIdx = mins[i];
        int current_width = image->width;

        unsigned int *newMinEnergySums = (unsigned int *)malloc(sizeof(unsigned int) * (current_width + 1) * height);
        unsigned char *newData = (unsigned char *)malloc(sizeof(unsigned char) * 3 * (current_width + 1) * height);
        unsigned char *oldData = image->lpData;

        // Kernel 1: Trace seam (Serial)
        {
            // Trace from bottom to top
            int x = minIdx;

            for (int y = height - 1; y >= 0; --y)
            {
                d_path[y] = x;

                if (y > 0)
                { // Calculate next x for y-1
                    int target_y = y - 1;
                    unsigned int min_val;
                    unsigned int v_l, v_m, v_r;

                    if (x == 0)
                    {
                        v_m = minEnergySums[target_y * current_width + x];
                        v_r = minEnergySums[target_y * current_width + x + 1];
                        min_val = MIN(v_m, v_r);
                    }
                    else if (x == current_width - 1)
                    {
                        v_l = minEnergySums[target_y * current_width + x - 1];
                        v_m = minEnergySums[target_y * current_width + x];
                        min_val = MIN(v_l, v_m);
                    }
                    else
                    {
                        v_l = minEnergySums[target_y * current_width + x - 1];
                        v_m = minEnergySums[target_y * current_width + x];
                        v_r = minEnergySums[target_y * current_width + x + 1];
                        min_val = MIN(v_l, MIN(v_m, v_r));
                    }

                    if (x > 0 && minEnergySums[target_y * current_width + x - 1] == min_val)
                    {
                        x = x - 1;
                    }
                    else if (x < current_width - 1 && minEnergySums[target_y * current_width + x + 1] == min_val)
                    {
                        x = x + 1;
                    }
                }
            }
        }

        // Kernel 2: Copy and Enlarge (Parallel)
        for (int y = 0; y < height; ++y)
        {
            for (int j = 0; j <= current_width; ++j)
            {
                // Determine seam x for this row
                int seam_x = d_path[y];

                if (j <= seam_x)
                {
                    // Copy as is
                    unsigned int val = minEnergySums[y * current_width + j];
                    newMinEnergySums[y * (current_width + 1) + j] = val;
                    newData[y * (current_width + 1) * 3 + j * 3 + 0] = oldData[y * current_width * 3 + j * 3 + 0];
                    newData[y * (current_width + 1) * 3 + j * 3 + 1] = oldData[y * current_width * 3 + j * 3 + 1];
                    newData[y * (current_width + 1) * 3 + j * 3 + 2] = oldData[y * current_width * 3 + j * 3 + 2];
                }
                else if (j == seam_x + 1)
                {
                    // Inserted pixel at seam_x + 1
                    // Copy from seam_x (duplication)
                    unsigned int val = minEnergySums[y * current_width + seam_x];
                    newMinEnergySums[y * (current_width + 1) + j] = val;
                    newData[y * (current_width + 1) * 3 + j * 3 + 0] = oldData[y * current_width * 3 + seam_x * 3 + 0];
                    newData[y * (current_width + 1) * 3 + j * 3 + 1] = oldData[y * current_width * 3 + seam_x * 3 + 1];
                    newData[y * (current_width + 1) * 3 + j * 3 + 2] = oldData[y * current_width * 3 + seam_x * 3 + 2];
                }
                else
                {
                    // j > seam_x + 1. Original index was j - 1
                    unsigned int val = minEnergySums[y * current_width + (j - 1)];
                    newMinEnergySums[y * (current_width + 1) + j] = val;
                    newData[y * (current_width + 1) * 3 + j * 3 + 0] = oldData[y * current_width * 3 + (j - 1) * 3 + 0];
                    newData[y * (current_width + 1) * 3 + j * 3 + 1] = oldData[y * current_width * 3 + (j - 1) * 3 + 1];
                    newData[y * (current_width + 1) * 3 + j * 3 + 2] = oldData[y * current_width * 3 + (j - 1) * 3 + 2];
                }
            }
        }

        // Free old
        free(oldData);
        free(minEnergySums);

        image->lpData = newData; // Update wrapper
        image->width = current_width + 1;
        minEnergySums = newMinEnergySums;
    }

    free(d_path);
    free(minEnergySums);
    free(mins);

    return image;
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
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
    printf("Execution time: %4.2f sec\n", (double)((double)(end - start) / CLOCKS_PER_SEC));
    storeJpegImageFile(output, outputFile);

    return 0;
}
