#include <limits>


#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/utility.hpp>

#define __OPENCV_BUILD 1
#include <opencv2/core/private.cuda.hpp>

#include <cublas.h>
#include <cufft.h>

using namespace cv;
using namespace cv::cuda;

#define error_entry(entry)  { entry, #entry }

struct ErrorEntry
{
    int code;
    const char* str;
};

const ErrorEntry cufft_errors[] =
{
    error_entry( CUFFT_INVALID_PLAN ),
    error_entry( CUFFT_ALLOC_FAILED ),
    error_entry( CUFFT_INVALID_TYPE ),
    error_entry( CUFFT_INVALID_VALUE ),
    error_entry( CUFFT_INTERNAL_ERROR ),
    error_entry( CUFFT_EXEC_FAILED ),
    error_entry( CUFFT_SETUP_FAILED ),
    error_entry( CUFFT_INVALID_SIZE ),
    error_entry( CUFFT_UNALIGNED_DATA )
};

struct ErrorEntryComparer
{
    int code;
    ErrorEntryComparer(int code_) : code(code_) {}
    bool operator()(const ErrorEntry& e) const { return e.code == code; }
};

cv::String getErrorString(int code, const ErrorEntry* errors, size_t n)
{
    size_t idx = std::find_if(errors, errors + n, ErrorEntryComparer(code)) - errors;

    const char* msg = (idx != n) ? errors[idx].str : "Unknown error code";
    cv::String str = cv::format("%s [Code = %d]", msg, code);

    return str;
}

const int cufft_error_num = sizeof(cufft_errors) / sizeof(cufft_errors[0]);

#define cufftSafeCall(expr)  ___cufftSafeCall(expr, __FILE__, __LINE__, CV_Func)

void ___cufftSafeCall(int err, const char* file, const int line, const char* func)
{
    if (CUFFT_SUCCESS != err)
    {
        String msg = getErrorString(err, cufft_errors, cufft_error_num);
        cv::error(cv::Error::GpuApiCallError, msg, func, file, line);
    }
}

enum DftFlags {
    DFT_COMPLEX_INPUT = 64,
    DFT_DOUBLE = 1024
};

class DFTImplCustom
{
    Size dft_size, dft_size_opt;
    bool is_1d_input, is_row_dft, is_scaled_dft, is_inverse, is_complex_input,
        is_complex_output, is_double_precision;

    cufftType dft_type;
    cufftHandle plan;

public:
    DFTImplCustom(Size dft_size, int flags)
        : dft_size(dft_size),
          is_1d_input((dft_size.height == 1) || (dft_size.width == 1)),
          is_row_dft((flags & DFT_ROWS) != 0),
          is_scaled_dft((flags & DFT_SCALE) != 0),
          is_inverse((flags & DFT_INVERSE) != 0),
          is_complex_input((flags & DFT_COMPLEX_INPUT) != 0),
          is_complex_output(!(flags & DFT_REAL_OUTPUT)),
          is_double_precision((flags & DFT_DOUBLE) != 0),
          dft_type(!is_complex_input ? (is_double_precision ? CUFFT_D2Z : CUFFT_R2C)
           : (is_complex_output ? (is_double_precision ? CUFFT_Z2Z : CUFFT_C2C)
            : (is_double_precision? CUFFT_Z2D : CUFFT_C2R)))
    {

        CV_Assert( !(flags & DFT_COMPLEX_OUTPUT) );

        CV_Assert( is_complex_input || is_complex_output );

        CV_Assert( !is_1d_input );

        cufftSafeCall( cufftPlan2d(&plan, dft_size.height, dft_size.width, dft_type) );
    }

    ~DFTImplCustom()
    {
        cufftSafeCall( cufftDestroy(plan) );
    }

    void compute(InputArray _src, OutputArray _dst, Stream& stream)
    {
        GpuMat src = getInputMat(_src, stream);

        CV_Assert( src.type() == CV_32FC1 || src.type() == CV_32FC2
            || src.type() == CV_64FC2 || src.type() == CV_64FC1);
        CV_Assert( is_complex_input == (src.channels() == 2) );

        // VERY IMPORTANT
        CV_Assert( src.isContinuous() );

        cufftSafeCall( cufftSetStream(plan, StreamAccessor::getStream(stream)) );

        if (is_complex_input)
        {
            if (is_complex_output)
            {
                if (is_double_precision)
                {
                    createContinuous(dft_size, CV_64FC2, _dst);
                    GpuMat dst = _dst.getGpuMat();

                    cufftSafeCall(cufftExecZ2Z(
                            plan, src.ptr<cufftDoubleComplex>(), dst.ptr<cufftDoubleComplex>(),
                            is_inverse ? CUFFT_INVERSE : CUFFT_FORWARD));
                } else
                {
                    GpuMat dst = _dst.getGpuMat();
                    CV_Assert( !dst.empty() );

                    cufftSafeCall(cufftExecC2C(
                            plan, src.ptr<cufftComplex>(), dst.ptr<cufftComplex>(),
                            is_inverse ? CUFFT_INVERSE : CUFFT_FORWARD));
                }
            }
            else
            {
                if (is_double_precision)
                {
                    createContinuous(dft_size, CV_64F, _dst);
                    GpuMat dst = _dst.getGpuMat();

                    cufftSafeCall(cufftExecZ2D(
                            plan, src.ptr<cufftDoubleComplex>(), dst.ptr<cufftDoubleReal>()));
                } else
                {
                    createContinuous(dft_size, CV_32F, _dst);
                    GpuMat dst = _dst.getGpuMat();

                    cufftSafeCall(cufftExecC2R(
                            plan, src.ptr<cufftComplex>(), dst.ptr<cufftReal>()));
                }
            }
        }
        else
        {
            if (is_double_precision)
            {
                GpuMat dst = _dst.getGpuMat();
                CV_Assert( !dst.empty() );

                cufftSafeCall(cufftExecD2Z(
                                  plan, src.ptr<cufftDoubleReal>(), dst.ptr<cufftDoubleComplex>()));
            } else
            {
                createContinuous(Size(dft_size.width / 2 + 1, dft_size.height), CV_32FC2, _dst);

                GpuMat dst = _dst.getGpuMat();

                cufftSafeCall(cufftExecR2C(
                                  plan, src.ptr<cufftReal>(), dst.ptr<cufftComplex>()));
            }
        }

        if (is_scaled_dft)
            cuda::divide(_dst, Scalar::all(dft_size.area()), _dst, 1, -1, stream);
    }
};



Ptr<DFTImplCustom> createDFTCustom(Size dft_size, int flags)
{
    return makePtr<DFTImplCustom>(dft_size, flags);
}

void cv::cuda::dft(InputArray _src, OutputArray _dst, Size dft_size, int flags, Stream& stream)
{
    if (getInputMat(_src, stream).channels() == 2)
        flags |= DFT_COMPLEX_INPUT;

    Ptr<DFTImplCustom> dft = createDFTCustom(dft_size, flags);
    dft->compute(_src, _dst, stream);
}
