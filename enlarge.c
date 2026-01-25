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
    map(to : lpData[0 : size]) map(from : output[0 : size])
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

// Erste Zeile kopieren (parallelisierbar)
#pragma omp target teams distribute parallel for map(to : data[0 : size]) map(tofrom : output[0 : size])
    for (int x = 0; x < width; ++x)
    {
        o(0, x) = d1(0, x);
    }

// Dynamische Programmierung - zeilenweise mit Abhängigkeit
// Jede Zeile kann parallel berechnet werden, aber Zeilen müssen sequentiell sein
#pragma omp target data map(to : data[0 : size]) map(tofrom : output[0 : size])
    {
        for (int y = 1; y < height; ++y)
        {
#pragma omp target teams distribute parallel for
            for (int x = 0; x < width; ++x)
            {
                if (x == width - 1)
                { // rightmost pixel of a row
                    o(y, x) = d1(y, x) + MIN(output[(y - 1) * width + (x - 1)], output[(y - 1) * width + x]);
                }
                else if (x == 0)
                { // leftmost pixel of a row
                    o(y, x) = d1(y, x) + MIN(output[(y - 1) * width + x], output[(y - 1) * width + (x + 1)]);
                }
                else
                {
                    o(y, x) = d1(y, x) + MIN(MIN(output[(y - 1) * width + (x - 1)], output[(y - 1) * width + x]), output[(y - 1) * width + (x + 1)]);
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
    long imgSize = (long)width * height * 3;
    long outSize = (long)width * height;

#pragma omp target teams distribute parallel for collapse(2) \
    map(to : lpData[0 : imgSize]) map(from : output[0 : outSize])
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int gx, gy, e_1, local_min, local_max, hist_width, e_entropy;
            double bins[9];

            // Step 1: Compute edge-component
            // apply Sobel operator in X direction - mit Boundary-Check
            int ym1 = (y > 0) ? y - 1 : 0;
            int yp1 = (y < height - 1) ? y + 1 : height - 1;
            int xm1 = (x > 0) ? x - 1 : 0;
            int xp1 = (x < width - 1) ? x + 1 : width - 1;

            gx = -1 * d3(ym1, xm1, 0) + 1 * d3(ym1, xp1, 0) - 2 * d3(y, xm1, 0) + 2 * d3(y, xp1, 0) - 1 * d3(yp1, xm1, 0) + 1 * d3(yp1, xp1, 0);

            // apply Sobel operator in Y direction
            gy = -1 * d3(ym1, xm1, 0) - 2 * d3(ym1, x, 0) - 1 * d3(ym1, xp1, 0) + 1 * d3(yp1, xm1, 0) + 2 * d3(yp1, x, 0) + 1 * d3(yp1, xp1, 0);

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

            // find min/max for local histogram mit Boundary-Check
            for (int v = -4; v < 4; ++v)
            {
                for (int u = -4; u < 4; ++u)
                {
                    int yy = y + v;
                    int xx = x + u;
                    if (yy < 0)
                        yy = 0;
                    if (yy >= height)
                        yy = height - 1;
                    if (xx < 0)
                        xx = 0;
                    if (xx >= width)
                        xx = width - 1;
                    int val = d3(yy, xx, 0);
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
                    int yy = y + v;
                    int xx = x + u;
                    if (yy < 0)
                        yy = 0;
                    if (yy >= height)
                        yy = height - 1;
                    if (xx < 0)
                        xx = 0;
                    if (xx >= width)
                        xx = width - 1;
                    int i = (d3(yy, xx, 0) - local_min) * 9 / hist_width;
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
    return output;
}

// increases the number of columns by cols
struct imgRawImage *increaseWidth(struct imgRawImage *image, int seams)
{
    int height = image->height;
    int width = image->width;

    unsigned int *pixelEnergies = calculateEnergySobel(image);
    unsigned int *minEnergySums = calculateMinEnergySums(pixelEnergies, width, height);
    free(pixelEnergies);

    // find seams by looking at the bottom row
    unsigned int *mins = malloc(sizeof(unsigned int) * seams);

// Parallele Suche nach den k kleinsten Werten
#pragma omp parallel for
    for (int k = 0; k < seams; ++k)
    {
        mins[k] = width;
    }

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

    // Vorberechnung aller Seam-Pfade
    int *seamPaths = malloc(sizeof(int) * seams * height);

#pragma omp parallel for
    for (int i = 0; i < seams; ++i)
    {
        int x = mins[i];
        seamPaths[i * height + (height - 1)] = x;

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
            seamPaths[i * height + y] = x;
        }
    }

    // Alle Seams auf einmal einfügen
    int newWidth = width + seams;
    unsigned char *newData = malloc(sizeof(unsigned char) * 3 * newWidth * height);
    unsigned char *oldData = image->lpData;

    long oldSize = (long)width * height * 3;
    long newSize = (long)newWidth * height * 3;
    long pathSize = (long)seams * height;

#pragma omp target teams distribute parallel for collapse(2)                                \
    map(to : oldData[0 : oldSize], seamPaths[0 : pathSize], width, height, seams, newWidth) \
    map(from : newData[0 : newSize])
    for (int y = 0; y < height; ++y)
    {
        for (int newX = 0; newX < newWidth; ++newX)
        {
            // Zähle wie viele Seams links von oder bei dieser Position liegen
            int seamsBefore = 0;
            int isSeam = 0;
            int seamIdx = -1;

            for (int s = 0; s < seams; ++s)
            {
                int seamX = seamPaths[s * height + y];
                if (seamX < newX - seamsBefore)
                {
                    seamsBefore++;
                }
                else if (seamX == newX - seamsBefore && !isSeam)
                {
                    isSeam = 1;
                    seamIdx = s;
                    seamsBefore++;
                }
            }

            int oldX = newX - seamsBefore;

            if (isSeam && seamIdx >= 0)
            {
                // Dupliziere den Pixel des Seams
                int srcX = seamPaths[seamIdx * height + y];
                if (srcX >= 0 && srcX < width)
                {
                    newData[y * newWidth * 3 + newX * 3 + 0] = oldData[y * width * 3 + srcX * 3 + 0];
                    newData[y * newWidth * 3 + newX * 3 + 1] = oldData[y * width * 3 + srcX * 3 + 1];
                    newData[y * newWidth * 3 + newX * 3 + 2] = oldData[y * width * 3 + srcX * 3 + 2];
                }
            }
            else if (oldX >= 0 && oldX < width)
            {
                // Kopiere normalen Pixel
                newData[y * newWidth * 3 + newX * 3 + 0] = oldData[y * width * 3 + oldX * 3 + 0];
                newData[y * newWidth * 3 + newX * 3 + 1] = oldData[y * width * 3 + oldX * 3 + 1];
                newData[y * newWidth * 3 + newX * 3 + 2] = oldData[y * width * 3 + oldX * 3 + 2];
            }
        }
    }

    free(image->lpData);
    free(minEnergySums);
    free(mins);
    free(seamPaths);

    image->lpData = newData;
    image->width = newWidth;

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
