#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include "tiffio.h"
#include <thread>
#include <cassert>
#include <mm_malloc.h>
#include <immintrin.h>

//  For passing parameters to the threads
typedef	struct {
    uint32 *raster;
    uint32 w;
    uint32 h;
    const float *filter;
    int f_len;
    int n_threads;
    uint32 *outRaster;
    int ID;
    int f_dim;
    int f_halfdim;
    int SEQ_START;
    int SEQ_FINISH;
    int SEQ_LEN;
    int regLen;
    int maxVecIndex;
    float*** filterArr;
} param;

// Prototypes
void filter_image_seq(uint32 *raster, uint32 w, uint32 h, const float *filter, int f_len);
int filterPixel(int row, int col, uint32 *raster, int w, int h, const float *filter, int f_dim);
int filterPixel_AVX(int row, int col, uint32 *raster, int w, int h, float*** filter, int f_len, uint32 *outRaster);
void filter_image_par(uint32 *raster, uint32 w, uint32 h, const float *filter, int f_len, int n_threads);
void* helper_filter_image_seq(void *args);



// saves TIFF file from data in `raster`
void save_tiff(const char *fname, uint32 *raster, uint32 w, uint32 h) {
    TIFF *tif = TIFFOpen(fname, "w");
    if (! raster) {
        throw std::runtime_error("Could not open output file");
    }
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, w);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, h);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 4);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8);
    TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_DEFLATE);
    TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFWriteEncodedStrip(tif, 0, raster, w*h*4);
    TIFFClose(tif);
}

// loads image data from `fname` (allocating dynamic memory)
// *w and *h are updated with the image dimensions
// raster is a matrix flattened into an array using row-major order
// every uint32 in the array is 4 bytes, enconding 8-bit packed ABGR
// A: transparency attribute (can be ignored)
// B: blue pixel
// G: green pixel
// R: red pixel


uint32 *load_tiff(const char *fname, uint32 *w, uint32 *h) {
    TIFF *tif = TIFFOpen(fname, "r");
    if (! tif) {
        throw std::runtime_error("Could not open input file");
    }

    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, w);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, h);
    uint32 *raster = (uint32 *) _TIFFmalloc(*w * *h * sizeof (uint32));
    if (! raster) {
        TIFFClose(tif);
        throw std::runtime_error("Memory allocation error");
    }
    if (! TIFFReadRGBAImageOriented(tif, *w, *h, raster, ORIENTATION_TOPLEFT, 0)) {
        TIFFClose(tif);
        throw std::runtime_error("Could not read raster from TIFF image");
    }
    TIFFClose(tif);
    return raster;
}



// Edits the passed raster image 8 pixels at a time using the provided filter.
void filter_image_seq(uint32 *raster, uint32 w, uint32 h, const float *filter, int f_len) {

    // to get RGB values from a pixel, you can either use bitwise masks
    // or rely on the following macros:
    // TIFFGetR(raster[i]) red
    // TIFFGetG(raster[i]) green
    // TIFFGetB(raster[i]) blue
    // TIFFGetA(raster[i]) this value should be ignored
    //
    // to modify RGB values from a pixel, you can use bitwise shifts or masks
    // each pixel stores values in the order ABGR
    // raster[i] = 0xFF0000FF;  // assigns Red to a pixel i
    // raster[i] = 0xFF00FF00;  // assigns Green to a pixel i
    // raster[i] = 0xFFFF0000;  // assigns Blue to a pixel i
    // note that the first byte is always FF (alpha = 255)
    //
    // TODO: here you will filter the image in raster
    //

    // outRaster holds the new values of raster so they can be copied over all at once at the end
    // This prevents dependencies (updating pixels as they are calculated would be all dependencies)
    uint32 *outRaster = (uint32 *) malloc(w * h * sizeof (uint32));

    // Computing te dimensions of the filter
    int f_dim = sqrt(f_len);
    int f_halfdim = f_dim/2;

    // Computes how many floats a 256-bit AVX register can hold
    unsigned int regLen = (256/8) / sizeof(float);

    // Compute which parts of the image can be handled with vectorized instructions
    // And which parts must be handled sequentially
    int SEQ_START = f_halfdim;
    int SEQ_FINISH = w-f_halfdim;
    int SEQ_LEN = SEQ_FINISH - SEQ_START;
    int maxVecIndex = SEQ_FINISH - (SEQ_FINISH % regLen);

    // Creates an array that represents the filter but has an array of 8 of the same
    // value in each location for use in AVX registers
    float*** avxFilterArr = (float***) malloc(f_dim * sizeof(float**));
    for (int i = 0; i < f_dim; i++) {
        avxFilterArr[i] = (float**) malloc(f_dim * sizeof(float*));
        for (int k = 0; k < f_dim; k++) {
            avxFilterArr[i][k] = (float*) _mm_malloc(regLen * sizeof(float),32);
            float filterVal = filter[i*f_dim + k];
            for (int h = 0; h < regLen; h++) {
                avxFilterArr[i][k][h] = filterVal;
            }
        }
    }

    // If the image work area evenly divides into the register length, do it all with SIMD
    if(SEQ_LEN % regLen == 0) {
        // Process only with AVX
        for (int j = f_halfdim; j < h - f_halfdim; j++) {
            // Process 8 pixels at a time to get through rows
            for (int k = f_halfdim; k < w - f_halfdim; k+=8) {
                filterPixel_AVX(j, k, raster, w, h, avxFilterArr, f_len, outRaster);
            }
        }
    }
    // Otherwise, we must process the unevenly dividing pixels sequentially
    else {
        for (int j = f_halfdim; j < h - f_halfdim; j++) {
            // Process 8 pixels at a time to get through rows
            for (int k = f_halfdim; k < maxVecIndex; k+=8) {
                filterPixel_AVX(j, k, raster, w, h, avxFilterArr, f_len, outRaster);
            }
            // Process 1 pixel at a time for the ones that don't divide evenly
            for (int k = maxVecIndex;k < SEQ_FINISH; k++) {
                outRaster[j*w + k] = filterPixel(j, k, raster, w, h, filter, f_len);
            }
        }
    }
    // After we're done computing pixels, copy them to the raster image from outRaster
    for (int j = f_halfdim; j < h - f_halfdim; j++) {
        for (int k = f_halfdim; k < w - f_halfdim; k++) {
            raster[j*w + k] = outRaster[j*w + k];
        }
    }
}

// Filters 8 pixels at a time, outputting them into outRaster
int filterPixel_AVX(int row, int col, uint32 *raster, int w, int h, float*** filter, int f_len, uint32 *outRaster) {

    // Computes the number of floats that can fit into the
    unsigned int regLen = (256/8) / sizeof(float);

    // Computes the dimensiosn of the filter
    int f_dim = sqrt(f_len);
    int f_halfdim = f_dim/2;

    // Create and pack memory-aligned arrays to store RGB values to perform AVX instructions on
    // NOTE: creates a filter-sized 3d array of the next 8 pixels' RGB values (f_dim x f_dim x 8)
    float*** regArrR = (float***) _mm_malloc(f_dim * sizeof(float**),32);
    float*** regArrG = (float***) _mm_malloc(f_dim * sizeof(float**),32);
    float*** regArrB = (float***) _mm_malloc(f_dim * sizeof(float**),32);
    for (int i = 0; i < f_dim; i++) {
        regArrR[i] = (float**) _mm_malloc(f_dim * sizeof(float*),32);
        regArrG[i] = (float**) _mm_malloc(f_dim * sizeof(float*),32);
        regArrB[i] = (float**) _mm_malloc(f_dim * sizeof(float*),32);
        for (int k = 0; k < f_dim; k++) {
            // Memory-aligned malloc
            regArrR[i][k] = (float*) _mm_malloc(regLen * sizeof(float),32);
            regArrG[i][k] = (float*) _mm_malloc(regLen * sizeof(float),32);
            regArrB[i][k] = (float*) _mm_malloc(regLen * sizeof(float),32);
            for (int h = 0; h < regLen; h++) {
                // Store the 8 pixel RGB values into the memory-aligned arrays
                regArrR[i][k][h] = (float) TIFFGetR(raster[(row+i-f_halfdim)*w + (col+k-f_halfdim + h)]);
                regArrG[i][k][h] = (float) TIFFGetG(raster[(row+i-f_halfdim)*w + (col+k-f_halfdim + h)]);
                regArrB[i][k][h] = (float) TIFFGetB(raster[(row+i-f_halfdim)*w + (col+k-f_halfdim + h)]);
            }
        }
    }

    // Four 256 bit registers
    __m256 reg0, reg1, reg2, reg3;

    // Use SIMD instructions to multiply 8 values RGB values against the filter at once
    for (int i = 0; i < f_dim; i++) {
        for (int k = 0; k <f_dim; k++) {
            // Load up the RGB data into the 256 bit registers
            reg0 = _mm256_load_ps(regArrR[i][k]);
            reg1 = _mm256_load_ps(regArrG[i][k]);
            reg2 = _mm256_load_ps(regArrB[i][k]);
            reg3 = _mm256_load_ps(filter[i][k]);

            // Multiply each RGB register by the filter register
            reg0 = _mm256_mul_ps(reg0, reg3);
            reg1 = _mm256_mul_ps(reg1, reg3);
            reg2 = _mm256_mul_ps(reg2, reg3);

            // Store the filtered values back into the original arrays for easy use
            _mm256_store_ps(regArrR[i][k], reg0);
            _mm256_store_ps(regArrG[i][k], reg1);
            _mm256_store_ps(regArrB[i][k], reg2);

        }
    }
    // Accumulate the R G and B values for each of the 8 pixels
    for (int h = 0; h < regLen; h++) {
        float countR = 0.0f;
        float countG = 0.0f;
        float countB = 0.0f;
        // Accumulate the values for one pixel
        for (int i = 0; i < f_dim; i++) {
            for (int k = 0; k <f_dim; k++) {
                countR += regArrR[i][k][h];
                countG += regArrG[i][k][h];
                countB += regArrB[i][k][h];

            }
        }
        // Fix one pixels values if they are invalid
        if (countR > 255.0f)
            countR = 255.0f;
        else if (countR < 0.0f)
            countR = 0.0f;

        if (countG > 255.0f)
            countG = 255.0f;
        else if (countG < 0.0f)
            countG = 0.0f;

        if (countB > 255.0f)
            countB = 255.0f;
        else if (countB < 0.0f)
            countB = 0.0f;

        // Turn one pixel into ARGB hex and store it in the outRaster
        int out = 0;
        out = 255;
        out <<= 8;
        out |= (int) countR;
        out <<= 8;
        out |= (int) countG;
        out <<= 8;
        out |= (int) countB;
        outRaster[row*w + col + h] = out;
    }
}


// Uses the raster and filter to compute and return new pixel for the given index
int filterPixel(int row, int col, uint32 *raster, int w, int h, const float *filter, int f_len) {

    // Compute filter dimensions
    int f_dim = sqrt(f_len);
    int f_halfdim = f_dim/2;

    // Accumulators for RGB values
    float countR = 0.0f;
    float countG = 0.0f;
    float countB = 0.0f;

    // For every pixel in a 3x3 (or other filter dimensions) centered on the current pixel...
    for (int i = -f_halfdim; i <= f_halfdim; i++) {
        for (int n = -f_halfdim; n <= f_halfdim; n++) {
            // Accumulate the RGB values multiplied by the corresponding filter value
            countR += filter[(i + f_halfdim)*f_dim + (n+f_halfdim)] * (float) TIFFGetR(raster[(row+i)*w + (col+n)]);
            countG += filter[(i + f_halfdim)*f_dim + (n+f_halfdim)] * (float) TIFFGetG(raster[(row+i)*w + (col+n)]);
            countB += filter[(i + f_halfdim)*f_dim + (n+f_halfdim)] * (float) TIFFGetB(raster[(row+i)*w + (col+n)]);
        }
    }

    // Check to make sure we don't have any invalid RGB values
    if (countR > 255.0f)
        countR = 255.0f;
    else if (countR < 0.0f)
        countR = 0.0f;

    if (countG > 255.0f)
        countG = 255.0f;
    else if (countG < 0.0f)
        countG = 0.0f;

    if (countB > 255.0f)
        countB = 255.0f;
    else if (countB < 0.0f)
        countB = 0.0f;



    // Rebuild the pixel hex ARGB using shift and or operators on the accumulators
    int out = 0;
    out = 255; // A value is always 255
    out <<= 8;
    out |= (int) countR;
    out <<= 8;
    out |= (int) countG;
    out <<= 8;
    out |= (int) countB;
    return out;

}

// Edits the passed raster image in parallel 8 pixels at a time using the provided filter. Divides the work among n_threads threads
void filter_image_par(uint32 *raster, uint32 w, uint32 h, const float *filter, int f_len, int n_threads) {
    //
    // TODO: here you will filter the image in raster using threads
    //

    // Array of pthreads
    pthread_t *threads = new pthread_t[n_threads];

    // outRaster holds the new values of raster so they can be copied over all at once at the end
    // This prevents dependencies (updating pixels as they are calculated would be all dependencies)
    uint32 **outRaster = (uint32 **) malloc(n_threads * sizeof(uint32 *));

    // Computes the dimensions of the filter
    int f_dim = sqrt(f_len);
    int f_halfdim = f_dim / 2;

    // Computes how many floats a 256-bit AVX register can hold
    unsigned int regLen = (256/8) / sizeof(float);

    // Compute which parts of the image can be handled with vectorized instructions
    // And which parts must be handled sequentially
    int SEQ_START = f_halfdim;
    int SEQ_FINISH = w-f_halfdim;
    int SEQ_LEN = SEQ_FINISH - SEQ_START;
    int maxVecIndex = SEQ_FINISH - (SEQ_FINISH % regLen);

    // Creates an array that represents the filter but has an array of 8 of the same
    // value in each location for use in AVX registers
    float*** avxFilterArr = (float***) malloc(f_dim * sizeof(float**));
    for (int i = 0; i < f_dim; i++) {
        avxFilterArr[i] = (float**) malloc(f_dim * sizeof(float*));
        for (int k = 0; k < f_dim; k++) {
            avxFilterArr[i][k] = (float*) _mm_malloc(regLen * sizeof(float),32);
            float filterVal = filter[i*f_dim + k];
            for (int h = 0; h < regLen; h++) {
                avxFilterArr[i][k][h] = filterVal;
            }
        }
    }

    // Send the required parameters and start the threads
    for (int i = 0; i < n_threads; i++) {
        outRaster[i] = (uint32 *) malloc(w * h * sizeof(uint32));
        param *threadVals = new param;
        threadVals->raster = raster;
        threadVals->w = w;
        threadVals->h = h;
        threadVals->filter = filter;
        threadVals->f_len = f_len;
        threadVals->n_threads = n_threads;
        threadVals->outRaster = outRaster[i];
        threadVals->ID = i;
        threadVals->f_dim = f_dim;
        threadVals->f_halfdim = f_halfdim;
        threadVals->SEQ_START = SEQ_START;
        threadVals->SEQ_FINISH = SEQ_FINISH;
        threadVals->SEQ_LEN = SEQ_LEN;
        threadVals->regLen = regLen;
        threadVals->maxVecIndex = maxVecIndex;
        threadVals->filterArr = avxFilterArr;
        // Create the thread and make sure it started
        int ret = pthread_create(&threads[i], NULL, helper_filter_image_seq, (void *) threadVals);
        assert(!ret);
    }

    // Wait for the threads to finish
    for (int i = 0; i < n_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    // Collect all the thread data and write it to the raster image
    for (int i = 0; i < n_threads; i++) {

        // Compute where each thread was working
        int ID = i;
        int START = f_halfdim;
        int FINISH = h - f_halfdim;
        const unsigned int bandHeight = (unsigned int) ceil((1.0 * (FINISH - START)) / n_threads);
        int THREAD_START = START + ID * bandHeight;
        int THREAD_FINISH = THREAD_START + bandHeight - 1;
        if (THREAD_FINISH > FINISH - 1) {
            THREAD_FINISH = FINISH - 1;
        }
        // Get each thread's work and write to raster
        for (int j = THREAD_START; j <= THREAD_FINISH; j++) {
            for (int k = f_halfdim; k < w - f_halfdim; k++) {
                raster[j * w + k] = outRaster[i][j * w + k];
            }
        }
    }
}

// This function gets called by the threads
// Computes the new pixels 8 at a time for the assigned area and stores them back into the struct's outRaster
void* helper_filter_image_seq(void *args) {

    // TODO: here you will filter the image in raster

    // Retrieving the data from the struct the thread was created with
    param *params = (param *) args;
    uint32 *raster = params->raster;
    uint32 w = params->w;
    uint32 h = params->h;
    const float *filter = params->filter;
    int f_len = params->f_len;
    int n_threads = params->n_threads;
    uint32 *outRaster = params->outRaster;
    int ID = params->ID;
    int f_dim = params->f_dim;
    int f_halfdim = params->f_halfdim;
    int SEQ_START = params->SEQ_START;
    int SEQ_FINISH = params->SEQ_FINISH;
    int SEQ_LEN = params->SEQ_LEN;
    int regLen = params->regLen;
    int maxVecIndex = params->maxVecIndex;
    float*** filterArr = params->filterArr;

    // Compute the rows this thread should work on
    int START = f_halfdim;
    int FINISH = h-f_halfdim;
    const unsigned int bandHeight = (unsigned int) ceil((1.0*(FINISH - START))/n_threads);
    int THREAD_START = START + ID*bandHeight;
    int THREAD_FINISH = THREAD_START + bandHeight - 1;
    if (THREAD_FINISH > FINISH - 1) {
        THREAD_FINISH = FINISH - 1;
    }

    // If the image work area evenly divides into the register length, do it all with SIMD
    if(SEQ_LEN % regLen == 0) {
        // Process only with AVX
        for (int j = THREAD_START; j <= THREAD_FINISH; j++) {
            // Process pixels 8 at a time
            for (int k = f_halfdim; k < w - f_halfdim; k+=8) {
                filterPixel_AVX(j, k, raster, w, h, filterArr, f_len, outRaster);
            }
        }
    }
    // Otherwise, we must process the unevenly dividing pixels sequentially
    else {
        for (int j = THREAD_START; j <= THREAD_FINISH; j++) {
            // Process pixels 8 at a time as we can
            for (int k = f_halfdim; k < maxVecIndex; k+=8) {
                filterPixel_AVX(j, k, raster, w, h, filterArr, f_len, outRaster);
            }
            // Process unevenly dividing pixels sequentially
            for (int k = maxVecIndex;k < SEQ_FINISH; k++) {
                outRaster[j*w + k] = filterPixel(j, k, raster, w, h, filter, f_len);
            }
        }
    }
    // Send the updated outRaster back to the struct for retrieval
    params->outRaster = outRaster;
}



float *load_filter(const char *fname, int *n) {
    std::ifstream myfile(fname);
    if (! myfile) {
        throw std::runtime_error("Could not open filter file");
    }
    myfile >> *n;
    float *filter = new float[*n];
    for (int i = 0 ; i < *n ; i++) myfile >> filter[i];
    myfile.close();
    return filter;
}


int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cout << "Usage:\t./filter <in_fname> <out_fname> <filter_fname> <algo> <n_threads>" << std::endl;
        std::cout << "<in_fname> path to the input image" << std::endl;
        std::cout << "<out_fname> path to the output image" << std::endl;
        std::cout << "<filter_fname> path to the filter file" << std::endl;
        std::cout << "<algo> whether to use the sequential (seq) or parallel algorithm (par)" << std::endl;
        std::cout << "<n_threads> number of threads to use" << std::endl;
        return 0;
    }

    uint32 width, height;
    int n_threads = std::stoi(argv[5]);

    // loads the filter
    int f_len;
    float *filter = load_filter(argv[3], &f_len);

    // loads image bytes from file name supplied as a command line argument
    // this function allocates memory dynamically
    uint32 *image = load_tiff(argv[1], &width, &height);

    // measure time of the algorithm
    auto start = std::chrono::high_resolution_clock::now();
    if (! std::strcmp(argv[4], "seq")) {
        // call the sequential implementation
        printf("Image = %x\n", image);
        filter_image_seq(image, width, height, filter, f_len);
        printf("Image = %x\n", image);
    } else if (! std::strcmp(argv[4], "par")) {
        // TODO: call the parallel implementation
        filter_image_par(image, width, height, filter, f_len, n_threads);

    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << diff.count() << std::endl;

    // save new file with filtered image
    save_tiff(argv[2], image, width, height);

    // frees memory allocated by load_filter and load_tiff
    delete [] filter;
    _TIFFfree(image);

    return 0;
    /*
    uint32 countR = 4;
    uint32 countG = 77;
    uint32 countB = 250;
    */
    /*
    char aStr[3];
    char rStr[3]; // Maybe need actual char*? Or maybe need & in sprintf?
    char gStr[3];
    char bStr[3];

    printf("\ncountR = %d\n", countR);
    printf("\ncountG = %d\n", countG);
    printf("\ncountB = %d\n", countB);

    printf("\ncountR (hex) = %x\n", countR);
    printf("\ncountG (hex) = %x\n", countG);
    printf("\ncountB (hex) = %x\n", countB);

    int hexR;

    sprintf(aStr, "%x", 255);
    sprintf(rStr, "%x", countR);
    // Will this add the extra 0 at the beginning?
    sprintf(gStr, "%x", countG);
    sprintf(bStr, "%x", countB);



    printf("\naStr = %s\n", aStr);
    printf("\nrStr = %s\n", rStr);
    printf("\ngStr = %s\n", gStr);
    printf("\nbStr = %s\n", bStr);
    */


    /*
    printf("\nTotal = %x\n", total);

    char totalStr[7] = "";
    sprintf(totalStr, "%x", total);

    char hexStr[9] = "";

    if (strlen(totalStr) != 6) {
        //strCpy(hexStr, "0"(;
    }
    strcat(hexStr, totalStr);

    printf("\nhexStr = %s", hexStr);


    // hex type or uint32 type?
    int newPixel;

    sscanf(hexStr, "%x", newPixel);

    printf("%x", newPixel);
    */


}
