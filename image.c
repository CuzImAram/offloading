#include <stdlib.h>
#include <stdio.h>
#include <jpeglib.h>
#include <jerror.h>
#include "image.h"

// load and save functions from https://www.tspi.at/2020/03/20/libjpegexample.html
struct imgRawImage *loadJpegImageFile(char *lpFilename) {
    struct jpeg_decompress_struct info;
    struct jpeg_error_mgr err;

    struct imgRawImage *lpNewImage;

    unsigned long int imgWidth, imgHeight;
    int numComponents;

    unsigned long int dwBufferBytes;
    unsigned char *lpData;

    unsigned char *lpRowBuffer[1];

    FILE *fHandle;

    fHandle = fopen(lpFilename, "rb");
    if (fHandle == NULL) {
#ifdef DEBUG
        fprintf(stderr, "%s:%u: Failed to read file %s\n", __FILE__, __LINE__, lpFilename);
#endif
        return NULL; /* ToDo */
    }

    info.err = jpeg_std_error(&err);
    jpeg_create_decompress(&info);

    jpeg_stdio_src(&info, fHandle);
    jpeg_read_header(&info, TRUE);

    jpeg_start_decompress(&info);
    imgWidth = info.output_width;
    imgHeight = info.output_height;
    numComponents = info.num_components;

#ifdef DEBUG
    fprintf(
      stderr,
      "%s:%u: Reading JPEG with dimensions %lu x %lu and %u components\n",
      __FILE__, __LINE__,
      imgWidth, imgHeight, numComponents
    );
#endif

    dwBufferBytes = imgWidth * imgHeight * 3; /* We only read RGB, not A */
    lpData = (unsigned char *) malloc(sizeof(unsigned char) * dwBufferBytes);

    lpNewImage = (struct imgRawImage *) malloc(sizeof(struct imgRawImage));
    lpNewImage->numComponents = numComponents;
    lpNewImage->width = imgWidth;
    lpNewImage->height = imgHeight;
    lpNewImage->lpData = lpData;

    /* Read scanline by scanline */
    while (info.output_scanline < info.output_height) {
        lpRowBuffer[0] = (unsigned char *) (&lpData[3 * info.output_width * info.output_scanline]);
        jpeg_read_scanlines(&info, lpRowBuffer, 1);
    }

    jpeg_finish_decompress(&info);
    jpeg_destroy_decompress(&info);
    fclose(fHandle);

    return lpNewImage;
}

int storeJpegImageFile(struct imgRawImage *lpImage, char *lpFilename) {
    struct jpeg_compress_struct info;
    struct jpeg_error_mgr err;

    unsigned char *lpRowBuffer[1];

    FILE *fHandle;

    fHandle = fopen(lpFilename, "wb");
    if (fHandle == NULL) {
#ifdef DEBUG
        fprintf(stderr, "%s:%u Failed to open output file %s\n", __FILE__, __LINE__, lpFilename);
#endif
        return 1;
    }

    info.err = jpeg_std_error(&err);
    jpeg_create_compress(&info);

    jpeg_stdio_dest(&info, fHandle);

    info.image_width = lpImage->width;
    info.image_height = lpImage->height;
    info.input_components = 3;
    info.in_color_space = JCS_RGB;

    jpeg_set_defaults(&info);
    jpeg_set_quality(&info, 100, TRUE);

    jpeg_start_compress(&info, TRUE);

    /* Write every scanline ... */
    while (info.next_scanline < info.image_height) {
        lpRowBuffer[0] = &(lpImage->lpData[info.next_scanline * (lpImage->width * 3)]);
        jpeg_write_scanlines(&info, lpRowBuffer, 1);
    }

    jpeg_finish_compress(&info);
    fclose(fHandle);

    jpeg_destroy_compress(&info);
    return 0;
}

