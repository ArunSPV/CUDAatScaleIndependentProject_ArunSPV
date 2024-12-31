#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_cuda.h>
#include <helper_string.h>

#include <FreeImage.h>

// some parts of the code are from 
// boxFilterNPP lab in CUDAatScale coursera course

bool printfNPPinfo(int argc, char *argv[])
{
    const NppLibraryVersion *libVer = nppGetLibVersion();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor,
           libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
           (driverVersion % 100) / 10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
           (runtimeVersion % 100) / 10);

    // Min spec is SM 1.0 devices
    bool bVal = checkCudaCapabilities(1, 0);
    return bVal;
}

// load or open png image
FIBITMAP* loadImage(const std::string& filename) 
{
    printf("Opening png image\n");
    FIBITMAP *loadedimage = FreeImage_Load(FIF_PNG, filename.c_str(), PNG_DEFAULT);

    if (!loadedimage)
    {
        std::cerr << "Cannot load image error noticed" << std::endl;
        FreeImage_DeInitialise();
        exit(-1);
    }

    return loadedimage;
}

// Save png image
void saveImage(const std::string &filename, FIBITMAP *imagetosave)
{
    printf("Saving png image \n");
    if (!FreeImage_Save(FIF_PNG, imagetosave, filename.c_str(), PNG_DEFAULT))
    {
        std::cerr << "Error saving image" << std::endl;
        FreeImage_Unload(imagetosave);
        FreeImage_DeInitialise();
        exit(-1);
    }
}

FIBITMAP *flipPngImage(FIBITMAP* hstSrcPngImg)
{
    printf("Flipping png image \n");
    // get image width, height, bits per pixel, pitch 
    int imgWidth = FreeImage_GetWidth(hstSrcPngImg);
    int imgHeight = FreeImage_GetHeight(hstSrcPngImg);
    int imgBpp = FreeImage_GetBPP(hstSrcPngImg);
    int imgPitch = FreeImage_GetPitch(hstSrcPngImg);
    int imgChannels = imgBpp / 8;

    // create struct with ROI size
    NppiSize oSizeROI = {imgWidth, imgHeight};

    // get data from host image
    BYTE *imgData = FreeImage_GetBits(hstSrcPngImg);

    // allocate image sizes on device
    npp::ImageNPP_8u_C4 oDeviceSrc(oSizeROI.width, oSizeROI.height);
    npp::ImageNPP_8u_C4 oDeviceDst(oSizeROI.width, oSizeROI.height);

    // copy image data from host to gpu
    cudaMemcpy2D(oDeviceSrc.data(), imgPitch, imgData, imgPitch, 
        imgWidth * imgChannels * sizeof(Npp8u), imgHeight, cudaMemcpyHostToDevice);

    // do image flip using nppimirror method
    NPP_CHECK_NPP(nppiMirror_8u_C4R(oDeviceSrc.data(), 
        imgWidth * imgChannels * sizeof(Npp8u), oDeviceDst.data(),
        imgWidth * imgChannels * sizeof(Npp8u), oSizeROI, NPP_VERTICAL_AXIS));

    //  Copy the processed image data back from device to host
    cudaMemcpy2D(imgData, imgPitch, oDeviceDst.data(), imgPitch, 
        imgWidth * imgChannels * sizeof(Npp8u), imgHeight, cudaMemcpyDeviceToHost);

    // free device memory
    nppiFree(oDeviceSrc.data());
    nppiFree(oDeviceDst.data());

    // assign data back to host png format
    FIBITMAP *hstDstPngImg = FreeImage_ConvertFromRawBits(
        imgData,
        imgWidth,
        imgHeight,
        imgPitch,
        imgBpp,
        0, 0, 0,
        FALSE);

    return hstDstPngImg;
}

int main(int argc, char *argv[])
{
    printf("%s Starting...\n", argv[0]);

    try
    {

        // initial part of code to get device and command line args
        // adopted from boxFilterNPP lab from CUDAatScale coursera course
        std::string sFilename;
        char *filePath;

        findCudaDevice(argc, (const char **)argv);

        if (printfNPPinfo(argc, argv) == false)
        {
            exit(EXIT_SUCCESS);
        }

        if (checkCmdLineFlag(argc, (const char **)argv, "input"))
        {
            getCmdLineArgumentString(argc, (const char **)argv, "input", &filePath);
        }
        else
        {
            filePath = sdkFindFilePath("sloth.png", argv[0]);
        }

        if (filePath)
        {
            sFilename = filePath;
        }
        else
        {
            sFilename = "sloth.png";
        }

        std::string sResultFilename = sFilename;
        std::string::size_type dot = sResultFilename.rfind('.');

        if (dot != std::string::npos)
        {
            sResultFilename = sResultFilename.substr(0, dot);
        }

        sResultFilename += "_flipped.png";

        if (checkCmdLineFlag(argc, (const char **)argv, "output"))
        {
            char *outputFilePath;
            getCmdLineArgumentString(argc, (const char **)argv, "output",
                                     &outputFilePath);
            sResultFilename = outputFilePath;
        }

        // free image initialisation
        FreeImage_Initialise();

        // open png image
        FIBITMAP *hstSrcPngImg = loadImage(sFilename);

        // do image flipping using NPP mirror method
        FIBITMAP *hstDstPngImg = flipPngImage(hstSrcPngImg);

        // save the output png image
        saveImage(sResultFilename, hstDstPngImg);

        // free up
        FreeImage_Unload(hstSrcPngImg);
        FreeImage_DeInitialise();

        printf("Done \n");
        exit(EXIT_SUCCESS);
    }

    // catch npp exceptions
    catch (npp::Exception &rException)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rException << std::endl;
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
    }
    catch (...)
    {
        std::cerr << "Program error! An unknown type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
        return -1;
    }

    return 0;
}
