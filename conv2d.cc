// Matthew Silva
// Professor Alvarez
// CSC 415
// 19 April 2018

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include "tiffio.h"
#include <thread>
#include <cassert>

//  For passing parameters to the threads
typedef	struct {
    uint32 *raster;
    uint32 w;
    uint32 h;
    const float *filter;
    int f_dim;
    int THREAD_START;
    int THREAD_FINISH;
    int f_halfdim;
    uint32* copyRaster;
} param;

// Prototypes
void filter_image_seq(uint32 *raster, uint32 w, uint32 h, const float *filter, int f_len);
int filteredPixel(int row, int col, uint32 *raster, int w, int h, const float *filter, int f_dim, int f_halfdim);
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

// Edits the passed raster image using the provided filter
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

    // Computing the dimensions of the filter
    int f_dim = sqrt(f_len);
    int f_halfdim = f_dim/2;

    // Computes a new pixel for every valid pixel in the grid (valid = all but the edges)
    for (int j = f_halfdim; j < h - f_halfdim; j++) {
        for (int k = f_halfdim; k < w - f_halfdim; k++) {
            // Store computed pixels in the outRaster
            outRaster[j*w + k] = filteredPixel(j, k, raster, w, h, filter, f_dim, f_halfdim);
        }
    }

    // Update the real raster image array using the new values we calculated and stored in outRaster
    for (int j = f_halfdim; j < h - f_halfdim; j++) {
        for (int k = f_halfdim; k < w - f_halfdim; k++) {
            // Copy from outRaster to raster
            raster[j*w + k] = outRaster[j*w + k];
        }
    }
}

// Uses the raster and filter to compute and return a new pixel for the given index
int filteredPixel(int row, int col, uint32 *raster, int w, int h, const float *filter, int f_dim, int f_halfdim) {

    // Accumulators for RGB values
    float countR = 0.0f;
    float countG = 0.0f;
    float countB = 0.0f;

    // Code motion / common subexpressions
    int halfdimdim = f_halfdim*f_dim;
    int roww = row*w;
    int ihalfdimdim;
    int rowiw;
    // For every pixel in a 3x3 (or other filter dimensions) centered on the current pixel...
    for (int i = -f_halfdim; i <= f_halfdim; i++) {
        // Code motion / Common subexpressions
        ihalfdimdim = (i)*f_dim + halfdimdim;
        rowiw = (i)*w + roww;
        for (int n = -f_halfdim; n <= f_halfdim; n++) {
            // Accumulate the RGB values multiplied by the corresponding filter value
            countR += filter[ihalfdimdim + (n+f_halfdim)] * (float) TIFFGetR(raster[rowiw + (col+n)]);
            countG += filter[ihalfdimdim + (n+f_halfdim)] * (float) TIFFGetG(raster[rowiw + (col+n)]);
            countB += filter[ihalfdimdim + (n+f_halfdim)] * (float) TIFFGetB(raster[rowiw + (col+n)]);
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
    uint32 out = 0;
    out = 255; // A value is always 255
    out <<= 8;
    out |= (int) countB;
    out <<= 8;
    out |= (int) countG;
    out <<= 8;
    out |= (int) countR;
    return out;

}

// Edits the passed raster image in parallel using the provided filter. Divides the work among n_threads threads
void filter_image_par(uint32 *raster, uint32 w, uint32 h, const float *filter, int f_len, int n_threads) {
    //
    // TODO: here you will filter the image in raster using threads
    //

    // Array of pthreads
    pthread_t *threads = new pthread_t[n_threads];

    // Copying the raster to a copyRaster to avoid dependencies (will always calculate pixels based on copy)
    uint32 *copyRaster = (uint32 *) _TIFFmalloc(w * h * sizeof(uint32));
    _TIFFmemcpy(copyRaster, raster, w * h * sizeof(uint32));

    // Computes the dimensions of the filter
    int f_dim = sqrt(f_len);
    int f_halfdim = f_dim / 2;

    // Compute valid rows
    int START = f_halfdim;
    int FINISH = h - f_halfdim;

    // Compute the number of rows each thread will be assigned
    const unsigned int bandHeight = (unsigned int) ceil((1.0 * (FINISH - START)) / n_threads);

    // Builds a struct for each thread, packing in its required parameters, and then launches the thread
    for (int i = 0; i < n_threads; i++) {

        // Compute the rows the thread is responsible for
        int ID = i;
        int THREAD_START = START + ID*bandHeight;
        int THREAD_FINISH = THREAD_START + bandHeight - 1;
        if (THREAD_FINISH > FINISH - 1) {
            THREAD_FINISH = FINISH - 1;
        }

        // Get the data the thread will need for computations
        param *threadVals = new param;
        threadVals->raster = raster;
        threadVals->w = w;
        threadVals->h = h;
        threadVals->filter = filter;
        threadVals->f_dim = f_dim;
        threadVals->THREAD_START = THREAD_START;
        threadVals->THREAD_FINISH = THREAD_FINISH;
        threadVals->f_halfdim = f_halfdim;
        threadVals->copyRaster = copyRaster;

        // Create the thread
        int ret = pthread_create(&threads[i], NULL, helper_filter_image_seq, (void *) threadVals);
        assert(!ret); // Makes sure the thread launched properly
    }

    // Join the threads before returning to main to preserve timing
    for (int i = 0; i < n_threads; i++) {
        pthread_join(threads[i], NULL);
    }

}

// This function gets called by the threads.
// Computes the new pixels for the assigned area and writes them to the raster
void* helper_filter_image_seq(void *args) {

    // TODO: here you will filter the image in raster

    // Retrieving the data from the struct the thread was created with
    param *params = (param *) args;
    uint32 *raster = params->raster;
    uint32 w = params->w;
    uint32 h = params->h;
    const float *filter = params->filter;
    int f_dim = params->f_dim;
    int f_halfdim = params->f_halfdim;
    int THREAD_START = params->THREAD_START;
    int THREAD_FINISH = params->THREAD_FINISH;
    uint32 *copyRaster = params->copyRaster;


    // Filter the pixels the thread is responsible for and write them to the raster
    int jw;
    int whalfdim = w-f_halfdim;
    for (int j = THREAD_START; j <= THREAD_FINISH; j++) {

        // Code motion / common subexpressions
        jw = j*w;
        for (int k = f_halfdim; k < whalfdim; k++) {
            // Compute a pixel and copy it to the raster (Filter based on a copy of raster to prevent dependencies)
            raster[jw + k] = filteredPixel(j, k, copyRaster, w, h, filter, f_dim, f_halfdim);
        }
    }
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
        filter_image_seq(image, width, height, filter, f_len);
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
}
