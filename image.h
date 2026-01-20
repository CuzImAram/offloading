struct imgRawImage {
    unsigned int numComponents;
    unsigned long int width, height;

    unsigned char *lpData;
};

struct imgRawImage *loadJpegImageFile(char *lpFilename);

int storeJpegImageFile(struct imgRawImage *lpImage, char *lpFilename);
