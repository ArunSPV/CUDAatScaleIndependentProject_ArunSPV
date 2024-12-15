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
// boxFilterNPP lab in CUDAAtScale coursera course

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

int main(int argc, char *argv[]) {

    printf("%s Starting...\n\n", argv[0]);

    try
    {

        std::string sFilename;

        char *filePath;

        findCudaDevice(argc, (const char **)argv);

        if (printfNPPinfo(argc, argv) == false) {

            exit(EXIT_SUCCESS);
        }

        if (checkCmdLineFlag(argc, (const char **)argv, "input")) {

            getCmdLineArgumentString(argc, (const char **)argv, "input", &filePath);
        }

        else {
            
            filePath = sdkFindFilePath("sloth.png", argv[0]);
        }

        if (filePath) {
            
            sFilename = filePath;
        }

        else {
            
            sFilename = "sloth.png";
        }

        std::string sResultFilename = sFilename;

        std::string::size_type dot = sResultFilename.rfind('.');

        if (dot != std::string::npos) {
            
            sResultFilename = sResultFilename.substr(0, dot);
        }

        sResultFilename += "_flipped.png";

        if (checkCmdLineFlag(argc, (const char **)argv, "output")) {

            char *outputFilePath;
            
            getCmdLineArgumentString(argc, (const char **)argv, "output",
                                     &outputFilePath);
            
            sResultFilename = outputFilePath;
        }

        FreeImage_Initialise();

        FIBITMAP *hstSrcPngImg = FreeImage_Load(FIF_PNG, sFilename.c_str(), PNG_DEFAULT);

        if (!hstSrcPngImg) {

            std::cerr << "Cannot load image error noticed" << std::endl;
            
            FreeImage_DeInitialise();
            
            return -1;
        }

        // Check color type of image
        FREE_IMAGE_COLOR_TYPE colorType = FreeImage_GetColorType(hstSrcPngImg);

        // Convert the image to 8-bit grayscale if colortype is not FIC_MINISBLACK
        if (colorType != FIC_MINISBLACK) {
            
            FIBITMAP *pngGrayScaleImg = FreeImage_ConvertToGreyscale(hstSrcPngImg);
            
            FreeImage_Unload(hstSrcPngImg);
            
            if (!pngGrayScaleImg) {

                std::cerr << "Cannot convert image to graysacale" << std::endl;
                
                FreeImage_DeInitialise();
                
                return -1;
            }

            hstSrcPngImg = pngGrayScaleImg;
        }

        // init image width and height 
        int imgWidth = FreeImage_GetWidth(hstSrcPngImg);
        int imgHeight = FreeImage_GetHeight(hstSrcPngImg);

        // create struct with ROI size
        NppiSize oSizeROI = {imgWidth, imgHeight};

        // allocate image size on device
        npp::ImageNPP_8u_C1 oDeviceSrc(oSizeROI.width, oSizeROI.height);
        npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);
        
        // get data from host image
        BYTE *imgData = FreeImage_GetBits(hstSrcPngImg);

        // copy image from host to device
        cudaMemcpy(oDeviceSrc.data(), imgData, imgWidth * imgHeight * sizeof(Npp8u), cudaMemcpyHostToDevice);

        // do image flip
        NPP_CHECK_NPP(nppiMirror_8u_C1R(
            oDeviceSrc.data(), oSizeROI.width, oDeviceDst.data(),
            oSizeROI.width, oSizeROI, NPP_VERTICAL_AXIS));

        // declare a host image for the result
        npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());

        // copy data from device to host
        cudaMemcpy(oHostDst.data(), oDeviceDst.data(), imgWidth * imgHeight * sizeof(Npp8u), cudaMemcpyDeviceToHost);

        nppiFree(oDeviceDst.data());
        nppiFree(oDeviceSrc.data());

        // assign data for an empty host image
        FIBITMAP *hstDstPngImg = FreeImage_ConvertFromRawBits(
            oHostDst.data(),
            imgWidth,
            imgHeight,
            imgWidth * sizeof(Npp8u),
            8,
            0, 0, 0,
            false);

        // Save image 
        if (!FreeImage_Save(FIF_PNG, hstDstPngImg, sResultFilename.c_str(), PNG_DEFAULT)) {

            std::cerr << "Error saving image" << std::endl;
            
            FreeImage_Unload(hstDstPngImg);
            
            FreeImage_DeInitialise();
            
            return -1;
        }

        // free up
        FreeImage_Unload(hstSrcPngImg);

        FreeImage_DeInitialise();

        exit(EXIT_SUCCESS);
    }

    catch (npp::Exception &rException) {
        
        std::cerr << "Program error! The following exception occurred: \n";
        
        std::cerr << rException << std::endl;
        
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
    }
    catch (...) {
        
        std::cerr << "Program error! An unknown type of exception occurred. \n";
        
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
        
        return -1;
    }

    return 0;
}
