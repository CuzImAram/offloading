#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <omp.h>
#include "image.h"

#define d3(y, x, z) lpData[(y) * width * 3 + (x) * 3 + (z)]
#define o3(y, x, z) output[(y) * width * 3 + (x) * 3 + (z)]
#define od3(y, x, z) oldData[(y) * width * 3 + (x) * 3 + (z)]
#define nd3(y, x, z) newData[(y) * (width + 1) * 3 + (x) * 3 + (z)]
#define m1(y, x) minEnergySums[(y) * width + (x)]
#define d1(y, x) data[(y) * width + (x)]
#define o(y, x) output[(y) * width + (x)]
#define nw(y, x) newMinEnergySums[(y) * (width + 1) + (x)]
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

// Makro für gepaddetes Bild (Padding = 4)
#define PAD 4
#define pd3(y, x, z) paddedData[((y) + PAD) * paddedWidth * 3 + ((x) + PAD) * 3 + (z)]

// Note: Since the gradient isn't normalized,
// we rescale the summands in the entropy calculations slightly,
#define entrop(p) (-1.0 * log2((p)) * (p) * (CHAR_MAX / 5.0 * 3.2))

unsigned char *gray(struct imgRawImage *image)
{
    unsigned int width = image->width;
    unsigned int height = image->height;
    unsigned char *lpData = image->lpData;
    unsigned char *output = malloc(sizeof(unsigned char) * 3 * width * height);
    long size = (long)width * height * 3;

    #pragma omp target teams distribute parallel for collapse(2) \
        map(to: lpData[0:size]) map(from: output[0:size])
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            unsigned char luma = (unsigned char)(0.299f * (float)d3(y, x, 0) + 0.587f * (float)d3(y, x, 1) + 0.114f * (float)d3(y, x, 2));
            o3(y, x, 0) = luma;
            o3(y, x, 1) = luma;
            o3(y, x, 2) = luma;
        }
    }
    return output;
}

unsigned int *calculateMinEnergySums(unsigned int *data, int width, int height)
{
    unsigned int *output = malloc(sizeof(unsigned int) * width * height);
    long size = (long)width * height;

    #pragma omp target data map(to: data[0:size]) map(tofrom: output[0:size])
    {
        // Erste Zeile kopieren
        #pragma omp target teams distribute parallel for
        for (int x = 0; x < width; ++x)
        {
            o(0, x) = d1(0, x);
        }

        // Dynamische Programmierung - zeilenweise mit Abhängigkeit
        for (int y = 1; y < height; ++y)
        {
            #pragma omp target teams distribute parallel for
            for (int x = 0; x < width; ++x)
            {
                if (x == width - 1)
                { // rightmost pixel of a row
                    o(y, x) = d1(y, x) + MIN(o(y - 1, x - 1), o(y - 1, x));
                }
                else if (x == 0)
                { // leftmost pixel of a row
                    o(y, x) = d1(y, x) + MIN(o(y - 1, x), o(y - 1, x + 1));
                }
                else
                {
                    o(y, x) = d1(y, x) + MIN(MIN(o(y - 1, x - 1), o(y - 1, x)), o(y - 1, x + 1));
                }
            }
        }
    }
    return output;
}

void swap(unsigned int *xp, unsigned int *yp)
{
    int temp = *xp;
    *xp = *yp;
    *yp = temp;
}

void selectionSort(unsigned int arr[], int n)
{
    int i, j, max_idx;

    // One by one move boundary of unsorted subarray
    for (i = 0; i < n - 1; i++)
    {
        // Find the maximum element in unsorted array
        max_idx = i;
        for (j = i + 1; j < n; j++)
            if (arr[j] > arr[max_idx])
                max_idx = j;

        // Swap the found minimum element with the first element
        swap(&arr[max_idx], &arr[i]);
    }
}

unsigned int *calculateEnergySobel(struct imgRawImage *image)
{
    unsigned int width = image->width;
    unsigned int height = image->height;
    unsigned char *lpData = image->lpData;
    unsigned int *output = malloc(sizeof(unsigned int) * height * width);
    
    // Erstelle gepaddetes Bild für korrekte Randzugriffe (wie im Original)
    int paddedWidth = width + 2 * PAD;
    int paddedHeight = height + 2 * PAD;
    unsigned char *paddedData = malloc(sizeof(unsigned char) * 3 * paddedWidth * paddedHeight);
    
    // Kopiere Originalbild in die Mitte des gepaddeten Bildes
    // und fülle Ränder mit 0 (wie uninitialisierter Speicher im Original)
    for (int y = 0; y < paddedHeight; ++y)
    {
        for (int x = 0; x < paddedWidth; ++x)
        {
            int srcY = y - PAD;
            int srcX = x - PAD;
            if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
            {
                paddedData[y * paddedWidth * 3 + x * 3 + 0] = lpData[srcY * width * 3 + srcX * 3 + 0];
                paddedData[y * paddedWidth * 3 + x * 3 + 1] = lpData[srcY * width * 3 + srcX * 3 + 1];
                paddedData[y * paddedWidth * 3 + x * 3 + 2] = lpData[srcY * width * 3 + srcX * 3 + 2];
            }
            else
            {
                // Außerhalb: mit 0 füllen (simuliert undefinierten Speicher)
                paddedData[y * paddedWidth * 3 + x * 3 + 0] = 0;
                paddedData[y * paddedWidth * 3 + x * 3 + 1] = 0;
                paddedData[y * paddedWidth * 3 + x * 3 + 2] = 0;
            }
        }
    }

    long paddedSize = (long)paddedWidth * paddedHeight * 3;
    long outSize = (long)width * height;

    #pragma omp target teams distribute parallel for collapse(2) \
        map(to: paddedData[0:paddedSize], width, height, paddedWidth) \
        map(from: output[0:outSize])
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int gx, gy, e_1, local_min, local_max, hist_width, e_entropy;
            double bins[9];

            // Step 1: Compute edge-component
            // apply Sobel operator in X direction
            gx = -1 * pd3(y - 1, x - 1, 0) + 1 * pd3(y - 1, x + 1, 0) 
               - 2 * pd3(y, x - 1, 0) + 2 * pd3(y, x + 1, 0) 
               - 1 * pd3(y + 1, x - 1, 0) + 1 * pd3(y + 1, x + 1, 0);

            // apply Sobel operator in Y direction
            gy = -1 * pd3(y - 1, x - 1, 0) - 2 * pd3(y - 1, x, 0) - 1 * pd3(y - 1, x + 1, 0) 
               + 1 * pd3(y + 1, x - 1, 0) + 2 * pd3(y + 1, x, 0) + 1 * pd3(y + 1, x + 1, 0);

            e_1 = (gx >= 0 ? gx : -gx) + (gy >= 0 ? gy : -gy);

            // Step 2: Compute entropy-component
            // clear out bins and reset variables
            for (int i = 0; i < 9; ++i)
            {
                bins[i] = 0;
            }
            local_min = INT_MAX;
            local_max = INT_MIN;
            e_entropy = 0;

            // find min/max for local histogram
            for (int v = -4; v < 4; ++v)
            {
                for (int u = -4; u < 4; ++u)
                {
                    int val = pd3(y + v, x + u, 0);
                    local_min = MIN(local_min, val);
                    local_max = MAX(local_max, val);
                }
            }
            hist_width = local_max - local_min + 1;

            // compute local histogram
            for (int v = -4; v < 4; ++v)
            {
                for (int u = -4; u < 4; ++u)
                {
                    int i = (pd3(y + v, x + u, 0) - local_min) * 9 / hist_width;
                    bins[i] += 1.0;
                }
            }

            // compute entropy
            for (int i = 0; i < 9; ++i)
            {
                bins[i] /= 81.0; // Normalize the counter to turn it into a probability
                if (bins[i] > 0.0)
                {
                    e_entropy += (int)entrop(bins[i]);
                }
            }

            // Step 3: assign energy value
            o(y, x) = e_1 + e_entropy;
        }
    }
    
    free(paddedData);
    return output;
}

// increases the number of columns by cols
struct imgRawImage *increaseWidth(struct imgRawImage *image, int seams)
{
    int height = image->height;
    unsigned int *newMinEnergySums;
    unsigned char *newData;

    unsigned int *pixelEnergies = calculateEnergySobel(image);
    unsigned int *minEnergySums = calculateMinEnergySums(pixelEnergies, image->width, image->height);
    free(pixelEnergies);

    // find seams by looking at the bottom row
    unsigned int mins[seams];
    int width = image->width;

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
            { // index is already a minimum
                continue;
            }
            if (mins[k] == width || m1(height - 1, j) < m1(height - 1, mins[k]))
            {
                mins[k] = j;
            }
        }
    }

    for (int k = 0; k < seams; ++k)
    {
        if (mins[k] >= width)
        {
            mins[k] = width - 1;
        }
    }

    selectionSort(mins, seams);

    for (int i = 0; i < seams; ++i)
    {
        unsigned int minIdx = mins[i];
        // each iteration increases the width by 1
        int width = image->width;
        unsigned char *oldData = image->lpData;
        // printf("iteration %i with width=%i and minIdx=%d\n", i, width, minIdx);
        newMinEnergySums = malloc(sizeof(unsigned int) * (width + 1) * height);
        newData = malloc(sizeof(unsigned char) * 3 * (width + 1) * height);

        // copy the pixels on the left side of the seam
        for (int j = 0; j <= minIdx; ++j)
        {
            nw(height - 1, j) = m1(height - 1, j);
            nd3(height - 1, j, 0) = od3(height - 1, j, 0);
            nd3(height - 1, j, 1) = od3(height - 1, j, 1);
            nd3(height - 1, j, 2) = od3(height - 1, j, 2);
        }
        nw(height - 1, minIdx + 1) = m1(height - 1, minIdx);
        nd3(height - 1, minIdx + 1, 0) = od3(height - 1, minIdx, 0);
        nd3(height - 1, minIdx + 1, 1) = od3(height - 1, minIdx, 1);
        nd3(height - 1, minIdx + 1, 2) = od3(height - 1, minIdx, 2);
        // move all pixels right of the seam 1 to the right
        for (int j = minIdx + 1; j < width; ++j)
        {
            nw(height - 1, j + 1) = m1(height - 1, j);
            nd3(height - 1, j + 1, 0) = od3(height - 1, j, 0);
            nd3(height - 1, j + 1, 1) = od3(height - 1, j, 1);
            nd3(height - 1, j + 1, 2) = od3(height - 1, j, 2);
        }
        int x = minIdx;
        for (int y = height - 2; y >= 0; --y)
        {
            unsigned int min;
            if (x == 0)
            {
                min = MIN(m1(y, x), m1(y, x + 1));
            }
            else if (x == width - 1)
            {
                min = MIN(m1(y, x - 1), m1(y, x));
            }
            else
            {
                min = MIN(m1(y, x - 1), MIN(m1(y, x), m1(y, x + 1)));
            }
            if (x > 0 && m1(y, x - 1) == min)
            {
                x = x - 1;
            }
            else if (x < width - 1 && m1(y, x + 1) == min)
            {
                x = x + 1;
            }
            for (int j = 0; j <= x; ++j)
            {
                nw(y, j) = m1(y, j);
                nd3(y, j, 0) = od3(y, j, 0);
                nd3(y, j, 1) = od3(y, j, 1);
                nd3(y, j, 2) = od3(y, j, 2);
            }
            nw(y, x + 1) = m1(y, x);
            nd3(y, x + 1, 0) = od3(y, x, 0);
            nd3(y, x + 1, 1) = od3(y, x, 1);
            nd3(y, x + 1, 2) = od3(y, x, 2);
            for (int j = x + 1; j < width; ++j)
            {
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
    free(minEnergySums);
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
    double start = omp_get_wtime();

    input->lpData = gray(input);
    struct imgRawImage *output = increaseWidth(input, seams);

    double end = omp_get_wtime();
    printf("Execution time: %4.2f sec\n", end - start);
    storeJpegImageFile(output, outputFile);

    return 0;
}
