#include "trackerKCFparallel.hpp"
#include <opencv2/cudaarithm.hpp>
#include "dft.cu"
#include "mulspectrums.cu"

#define returnFromUpdate() {fprintf(stderr, "Error in %s line %d while updating frame %d\n", __FILE__, __LINE__, frame);}

namespace cv{

  class TrackerKCFModel : public TrackerModel{
  public:
    TrackerKCFModel(TrackerKCF::Params /*params*/){}
    ~TrackerKCFModel(){}
  protected:
    void modelEstimationImpl( const std::vector<Mat>& /*responses*/ ){}
    void modelUpdateImpl(){}
  };
}

namespace helper {
    void MatType( Mat inputMat )
    {
    int inttype = inputMat.type();

    std::string r, a;
    uchar depth = inttype & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (inttype >> CV_CN_SHIFT);
    switch ( depth ) {
        case CV_8U:  r = "8U";   a = "Mat.at<uchar>(y,x)"; break;
        case CV_8S:  r = "8S";   a = "Mat.at<schar>(y,x)"; break;
        case CV_16U: r = "16U";  a = "Mat.at<ushort>(y,x)"; break;
        case CV_16S: r = "16S";  a = "Mat.at<short>(y,x)"; break;
        case CV_32S: r = "32S";  a = "Mat.at<int>(y,x)"; break;
        case CV_32F: r = "32F";  a = "Mat.at<float>(y,x)"; break;
        case CV_64F: r = "64F";  a = "Mat.at<double>(y,x)"; break;
        case CV_32FC2: r = "32FC2";  a = "Mat.at<complex float>(y,x)"; break;
        case CV_64FC2: r = "64FC2";  a = "Mat.at<complex double>(y,x)"; break;
        default:     r = "User"; a = "Mat.at<UKNOWN>(y,x)"; break;
    }
    r += "C";
    r += (chans+'0');
    std::cout << "Mat is of type " << r << " and should be accessed with " << a << std::endl;

    }
}

namespace cv {

     TackerKCFImplParallel::TackerKCFImplParallel( const TrackerKCF::Params &parameters ) :
         params( parameters )
     {
       isInit = false;
       resizeImage = false;
       use_custom_extractor_pca = false;
       use_custom_extractor_npca = false;
       #if TIME
       total_lines = num_steps;
       for (int i = 0; i < num_steps; i++) {
           cumulated_times[i] = 0;
       }
       #if TIME == 2
       for (int i = 0; i < num_steps - 1; i++) {
           total_lines += num_steps_details[i];
           for (int j = 0; j < max_num_details; j++) {
               cumulated_details_times[i][j] = 0;
           }
       }
       #endif
       #endif
     }

     void TackerKCFImplParallel::read( const cv::FileNode& fn ){
       params.read( fn );
     }

     void TackerKCFImplParallel::write( cv::FileStorage& fs ) const {
       params.write( fs );
     }

     bool TackerKCFImplParallel::initImpl( const Mat& image, const Rect2d& boundingBox ){
       #if TIME
       double startInit = CycleTimer::currentSeconds();
       #endif

       frame=0;
       roi = boundingBox;

       output_sigma=sqrt(roi.width*roi.height)*params.output_sigma_factor;
       output_sigma=-0.5/(output_sigma*output_sigma);

       if(params.resize && roi.width*roi.height>params.max_patch_size){
         resizeImage=true;
         roi.x/=2.0;
         roi.y/=2.0;
         roi.width/=2.0;
         roi.height/=2.0;
       }

       roi.x-=roi.width/2;
       roi.y-=roi.height/2;
       roi.width*=2;
       roi.height*=2;

       createHanningWindow(hann, roi.size(), CV_64F);

       Mat _layer[] = {hann, hann, hann, hann, hann, hann, hann, hann, hann, hann};
       merge(_layer, 10, hann_cn);

       y=Mat::zeros((int)roi.height,(int)roi.width,CV_64F);
       for(unsigned i=0;i<roi.height;i++){
         for(unsigned j=0;j<roi.width;j++){
           y.at<double>(i,j)=(i-roi.height/2+1)*(i-roi.height/2+1)+(j-roi.width/2+1)*(j-roi.width/2+1);
         }
       }

       y*=(double)output_sigma;
       cv::exp(y,y);

       fft2(y,yf);

       model=Ptr<TrackerKCFModel>(new TrackerKCFModel(params));

       if((params.desc_npca & GRAY) == GRAY)descriptors_npca.push_back(GRAY);
       if((params.desc_npca & CN) == CN)descriptors_npca.push_back(CN);
       if(use_custom_extractor_npca)descriptors_npca.push_back(CUSTOM);
       features_npca.resize(descriptors_npca.size());

       if((params.desc_pca & GRAY) == GRAY)descriptors_pca.push_back(GRAY);
       if((params.desc_pca & CN) == CN)descriptors_pca.push_back(CN);
       if(use_custom_extractor_pca)descriptors_pca.push_back(CUSTOM);
       features_pca.resize(descriptors_pca.size());

       CV_Assert(
         (params.desc_pca & GRAY) == GRAY
         || (params.desc_npca & GRAY) == GRAY
         || (params.desc_pca & CN) == CN
         || (params.desc_npca & CN) == CN
         || use_custom_extractor_pca
         || use_custom_extractor_npca
       );

       cuda::createContinuous(roi.size(), CV_8UC3, patch_data_gpu);
       cuda::createContinuous(roi.size(), CV_16U, indexes_gpu);
       hann_cn_gpu.upload(hann_cn);

       cuda::createContinuous(roi.size(), CV_64F, pca_data_gpu);

       Size complex_size(roi.size().width/2+1, roi.size().height);
       int num_channels = image.channels();
       cuda::createContinuous(complex_size, CV_64FC2, xyf_c_gpu);
       cuda::createContinuous(roi.size(), CV_64F, xyf_r_gpu);
       xf_data_gpu.resize(num_channels);
       yf_data_gpu.resize(num_channels);
       layers_data_gpu.resize(num_channels);
       xyf_v_gpu.resize(num_channels);
       for (int i = 0; i < num_channels; i++){
           cuda::createContinuous(roi.size(), CV_64F, layers_data_gpu[i]);
           cuda::createContinuous(complex_size, CV_64FC2, xf_data_gpu[i]);
           cuda::createContinuous(complex_size, CV_64FC2, yf_data_gpu[i]);
       }

       size_t ColorNames_size = 32768 * 10 * sizeof(double); //2^15 * 10
       cudaSafeCall(cudaMalloc((void**) &ColorNames_gpu, ColorNames_size));
       cudaSafeCall(cudaMemcpy(ColorNames_gpu, ColorNames, ColorNames_size, cudaMemcpyHostToDevice));

       #if TIME
       printInitializationTime(startInit);
       #endif

       return true;
     }

     // KCF MAIN --- HUGE PERFORMANCE ANALYSIS HERE
     bool TackerKCFImplParallel::updateImpl( const Mat& image, Rect2d& boundingBox ){
       #if TIME
       double startUpdate = CycleTimer::currentSeconds();
       #endif

       double minVal, maxVal;
       Point minLoc,maxLoc;

       Mat img=image.clone();
       CV_Assert(img.channels() == 1 || img.channels() == 3);

       if(resizeImage)resize(img,img,Size(img.cols/2,img.rows/2));

       #if TIME
       double startDetection = CycleTimer::currentSeconds();
       #endif

       if(frame>0){
         #if TIME == 2
         double startDetectionDetail = CycleTimer::currentSeconds();
         #endif

         for(unsigned i=0;i<descriptors_npca.size()-extractor_npca.size();i++){
           if(!getSubWindow(img,roi, features_npca[i], img_Patch, descriptors_npca[i]))returnFromUpdate();
         }

         #if TIME == 2
         updateTimeDetail(&startDetectionDetail, 0, 0);
         #endif

         for(unsigned i=0,j=(unsigned)(descriptors_npca.size()-extractor_npca.size());i<extractor_npca.size();i++,j++){
           if(!getSubWindow(img,roi, features_npca[j], extractor_npca[i]))returnFromUpdate();
         }
         if(features_npca.size()>0)merge(features_npca,X[1]);

         #if TIME == 2
         updateTimeDetail(&startDetectionDetail, 0, 1);
         #endif

         for(unsigned i=0;i<descriptors_pca.size()-extractor_pca.size();i++){
           if(!getSubWindow(img,roi, features_pca[i], img_Patch, descriptors_pca[i]))returnFromUpdate();
         }


         #if TIME == 2
         updateTimeDetail(&startDetectionDetail, 0, 2);
         #endif

         for(unsigned i=0,j=(unsigned)(descriptors_pca.size()-extractor_pca.size());i<extractor_pca.size();i++,j++){
           if(!getSubWindow(img,roi, features_pca[j], extractor_pca[i]))returnFromUpdate();
         }
         if(features_pca.size()>0)merge(features_pca,X[0]);

         #if TIME == 2
         updateTimeDetail(&startDetectionDetail, 0, 3);
         #endif

         if(params.desc_pca !=0){
           compress(proj_mtx,X[0],X[0],data_temp,compress_data);
           compress(proj_mtx,Z[0],Zc[0],data_temp,compress_data);
         }

         Zc[1] = Z[1];

         #if TIME == 2
         updateTimeDetail(&startDetectionDetail, 0, 4);
         #endif

         if(features_npca.size()==0){
           x = X[0];
           z = Zc[0];
         }else if(features_pca.size()==0){
           x = X[1];
           z = Z[1];
         }else{
           merge(X,2,x);
           merge(Zc,2,z);
         }

         #if TIME == 2
         updateTimeDetail(&startDetectionDetail, 0, 5);
         #endif

         denseGaussKernel(params.sigma,x,z,k,layers,vxf,vyf,vxyf,xy_data,xyf_data);

         #if TIME == 2
         updateTimeDetail(&startDetectionDetail, 0, 6);
         #endif

         fft2(k,kf);
         if(frame==1)spec2=Mat_<Vec2d >(kf.rows, kf.cols);

         #if TIME == 2
         updateTimeDetail(&startDetectionDetail, 0, 7);
         #endif

         if(params.split_coeff)
           calcResponse(alphaf,alphaf_den,kf,response, spec, spec2);
         else
           calcResponse(alphaf,kf,response, spec);

         #if TIME == 2
         updateTimeDetail(&startDetectionDetail, 0, 8);
         #endif

         minMaxLoc( response, &minVal, &maxVal, &minLoc, &maxLoc );
         roi.x+=(maxLoc.x-roi.width/2+1);
         roi.y+=(maxLoc.y-roi.height/2+1);

         #if TIME == 2
         updateTimeDetail(&startDetectionDetail, 0, 9);
         #endif
       }

       #if TIME
       updateTime(startDetection, 0);

       double startPatches = CycleTimer::currentSeconds();
       #endif

       #if TIME == 2
       double startPatchesDetail = startPatches;
       #endif

       boundingBox.x=(resizeImage?roi.x*2:roi.x)+(resizeImage?roi.width*2:roi.width)/4;
       boundingBox.y=(resizeImage?roi.y*2:roi.y)+(resizeImage?roi.height*2:roi.height)/4;
       boundingBox.width = (resizeImage?roi.width*2:roi.width)/2;
       boundingBox.height = (resizeImage?roi.height*2:roi.height)/2;

       #if TIME == 2
       updateTimeDetail(&startPatchesDetail, 1, 0);
       #endif

       // get non compressed descriptors
       for(unsigned i=0;i<descriptors_npca.size()-extractor_npca.size();i++){
         if(!getSubWindow(img,roi, features_npca[i], img_Patch, descriptors_npca[i]))returnFromUpdate();
       }

       #if TIME == 2
       updateTimeDetail(&startPatchesDetail, 1, 1);
       #endif


       //get non-compressed custom descriptors
       for(unsigned i=0,j=(unsigned)(descriptors_npca.size()-extractor_npca.size());i<extractor_npca.size();i++,j++){
         if(!getSubWindow(img,roi, features_npca[j], extractor_npca[i]))returnFromUpdate();
       }
       if(features_npca.size()>0)merge(features_npca,X[1]);

       #if TIME == 2
       updateTimeDetail(&startPatchesDetail, 1, 2);
       #endif

       for(unsigned i=0;i<descriptors_pca.size()-extractor_pca.size();i++){
         if(!getSubWindow(img,roi, features_pca[i], img_Patch, descriptors_pca[i]))returnFromUpdate();
       }

       #if TIME == 2
       updateTimeDetail(&startPatchesDetail, 1, 3);
       #endif

       for(unsigned i=0,j=(unsigned)(descriptors_pca.size()-extractor_pca.size());i<extractor_pca.size();i++,j++){
         if(!getSubWindow(img,roi, features_pca[j], extractor_pca[i]))returnFromUpdate();
       }
       if(features_pca.size()>0)merge(features_pca,X[0]);

       #if TIME == 2
       updateTimeDetail(&startPatchesDetail, 1, 4);
       #endif

       if(frame==0){
         Z[0] = X[0].clone();
         Z[1] = X[1].clone();
       }else{
         Z[0]=(1.0-params.interp_factor)*Z[0]+params.interp_factor*X[0];
         Z[1]=(1.0-params.interp_factor)*Z[1]+params.interp_factor*X[1];
       }

       #if TIME == 2
       updateTimeDetail(&startPatchesDetail, 1, 5);
       #endif

       #if TIME
       updateTime(startPatches, 1);
       double startCompression = CycleTimer::currentSeconds();
       #endif

       #if TIME == 2
       double startCompressionDetail = startCompression;
       #endif


       if(params.desc_pca !=0 || use_custom_extractor_pca){
         if(frame==0){
           layers_pca_data.resize(Z[0].channels());
           average_data.resize(Z[0].channels());
         }

         updateProjectionMatrix(Z[0],old_cov_mtx,proj_mtx,params.pca_learning_rate,params.compressed_size,layers_pca_data,average_data,data_pca, new_covar,w_data,u_data,vt_data);

         #if TIME == 2
         updateTimeDetail(&startCompressionDetail, 2, 0);
         #endif

         compress(proj_mtx,X[0],X[0],data_temp,compress_data);

         #if TIME == 2
         updateTimeDetail(&startCompressionDetail, 2, 1);
         #endif
       }

       if(features_npca.size()==0)
         x = X[0];
       else if(features_pca.size()==0)
         x = X[1];
       else
         merge(X,2,x);

       #if TIME == 2
       updateTimeDetail(&startCompressionDetail, 2, 2);
       #endif

       #if TIME
       updateTime(startCompression, 2);
       double startLeastSquares = CycleTimer::currentSeconds();
       #endif


       #if TIME == 2
       double startLeastSquaresDetail = startLeastSquares;
       #endif

       if(frame==0){
         layers.resize(x.channels());
         vxf.resize(x.channels());
         vyf.resize(x.channels());
         vxyf.resize(vyf.size());
         new_alphaf=Mat_<Vec2d >(yf.rows, yf.cols);
       }

       #if TIME == 2
       updateTimeDetail(&startLeastSquaresDetail, 3, 0);
       #endif

       denseGaussKernel(params.sigma,x,x,k,layers,vxf,vyf,vxyf,xy_data,xyf_data);

       #if TIME == 2
       updateTimeDetail(&startLeastSquaresDetail, 3, 1);
       #endif

       fft2(k,kf);

       #if TIME == 2
       updateTimeDetail(&startLeastSquaresDetail, 3, 2);
       #endif

       kf_lambda=kf+params.lambda;

       #if TIME == 2
       updateTimeDetail(&startLeastSquaresDetail, 3, 3);
       #endif

       double den;
       if(params.split_coeff){
         mulSpectrums(yf,kf,new_alphaf,0);
         mulSpectrums(kf,kf_lambda,new_alphaf_den,0);
       }else{
         for(int i=0;i<yf.rows;i++){
           for(int j=0;j<yf.cols;j++){
             den = 1.0/(kf_lambda.at<Vec2d>(i,j)[0]*kf_lambda.at<Vec2d>(i,j)[0]+kf_lambda.at<Vec2d>(i,j)[1]*kf_lambda.at<Vec2d>(i,j)[1]);

             new_alphaf.at<Vec2d>(i,j)[0]=
             (yf.at<Vec2d>(i,j)[0]*kf_lambda.at<Vec2d>(i,j)[0]+yf.at<Vec2d>(i,j)[1]*kf_lambda.at<Vec2d>(i,j)[1])*den;
             new_alphaf.at<Vec2d>(i,j)[1]=
             (yf.at<Vec2d>(i,j)[1]*kf_lambda.at<Vec2d>(i,j)[0]-yf.at<Vec2d>(i,j)[0]*kf_lambda.at<Vec2d>(i,j)[1])*den;
           }
         }
       }

       #if TIME == 2
       updateTimeDetail(&startLeastSquaresDetail, 3, 4);
       #endif

       if(frame==0){
         alphaf=new_alphaf.clone();
         if(params.split_coeff)alphaf_den=new_alphaf_den.clone();
       }else{
         alphaf=(1.0-params.interp_factor)*alphaf+params.interp_factor*new_alphaf;
         if(params.split_coeff)alphaf_den=(1.0-params.interp_factor)*alphaf_den+params.interp_factor*new_alphaf_den;
       }

       #if TIME == 2
       updateTimeDetail(&startLeastSquaresDetail, 3, 5);
       #endif

       #if TIME
       updateTime(startLeastSquares, 3);
       updateTime(startUpdate, 4);
       printAverageTimes();
       #endif

       frame++;

       return true;
     }


     // KCF !!!

     void TackerKCFImplParallel::createHanningWindow(OutputArray dest, const cv::Size winSize, const int type) const {
         CV_Assert( type == CV_32FC1 || type == CV_64FC1 );

         dest.create(winSize, type);
         Mat dst = dest.getMat();

         int rows = dst.rows, cols = dst.cols;

         AutoBuffer<double> _wc(cols);
         double * const wc = (double *)_wc;

         double coeff0 = 2.0 * CV_PI / (double)(cols - 1), coeff1 = 2.0f * CV_PI / (double)(rows - 1);
         for(int j = 0; j < cols; j++)
           wc[j] = 0.5 * (1.0 - cos(coeff0 * j));

         if(dst.depth() == CV_32F){
           for(int i = 0; i < rows; i++){
             float* dstData = dst.ptr<float>(i);
             double wr = 0.5 * (1.0 - cos(coeff1 * i));
             for(int j = 0; j < cols; j++)
               dstData[j] = (float)(wr * wc[j]);
           }
         }else{
           for(int i = 0; i < rows; i++){
             double* dstData = dst.ptr<double>(i);
             double wr = 0.5 * (1.0 - cos(coeff1 * i));
             for(int j = 0; j < cols; j++)
               dstData[j] = wr * wc[j];
           }
         }

     }

     void inline TackerKCFImplParallel::fft2(const Mat src, Mat & dest) const {
       dft(src,dest,DFT_COMPLEX_OUTPUT);
     }

     void inline TackerKCFImplParallel::fft2(const Mat src, std::vector<Mat> & dest, std::vector<Mat> & layers_data) const {
       split(src, layers_data);

       for(int i=0;i<src.channels();i++){
         dft(layers_data[i],dest[i],DFT_COMPLEX_OUTPUT);
       }
     }

     void inline TackerKCFImplParallel::cudafft2(int num_channels, std::vector<cuda::GpuMat> & dest, std::vector<cuda::GpuMat> & layers_data) {
       for (int i = 0; i < num_channels; i++) {
         cuda::dft(layers_data[i], dest[i], layers_data[i].size(), DFT_DOUBLE);
       }
     }

      void inline TackerKCFImplParallel::ifft2(const Mat src, Mat & dest) const {
        idft(src,dest,DFT_SCALE+DFT_REAL_OUTPUT);
      }

     void inline TackerKCFImplParallel::cudaifft2(const cuda::GpuMat src, cuda::GpuMat & dest) {
       cuda::GpuMat src_cce;
       src_cce = src;
       cv::Size dest_size((src.size().width -1)*2,src.size().height);


       cuda::dft(src_cce, dest, dest_size,
         (DFT_SCALE + DFT_REAL_OUTPUT) | DFT_INVERSE | DFT_DOUBLE);

     }

    void inline TackerKCFImplParallel::cce2full(const Mat src, Mat & dest) {

        Mat result(cv::Size((src.size().width-1)*2,src.size().height),src.type());
        for (int j=0; j < (src.size().width-1)*2;j++) {
            for (int i = 0; i < src.size().height;i++) {
                if (j <src.size().width-1) {
                    result.at<Vec2d>(i,j)[0] = src.at<Vec2d>(i,j)[0];
                    result.at<Vec2d>(i,j)[1] = src.at<Vec2d>(i,j)[1];
                } else {
                    // Complex conjugate
                    result.at<Vec2d>(i,j)[0] = src.at<Vec2d>(i,2*(src.size().width - 1) - j)[0];
                    result.at<Vec2d>(i,j)[1] =  - src.at<Vec2d>(i,2*(src.size().width -1) - j)[1];
                }
            }
        }
        dest = result;
    }

    void inline TackerKCFImplParallel::full2cce(const Mat src, Mat & dest) {
        cv::Rect roi(0, 0, src.size().width/2+1, src.size().height);
        dest = src(roi);
    }

     void inline TackerKCFImplParallel::pixelWiseMult(const std::vector<cuda::GpuMat> src1, const std::vector<cuda::GpuMat>  src2, std::vector<cuda::GpuMat>  & dest, const int flags, const bool conjB) const {
       for(unsigned i=0;i<src1.size();i++){
         cv::cuda::mulSpectrums(src1[i], src2[i], dest[i],flags,conjB);
       }
     }

     void inline TackerKCFImplParallel::sumChannels(std::vector<cuda::GpuMat> src, cuda::GpuMat & dest) const {
       src[0].copyTo(dest);
       for(unsigned i=1;i<src.size();i++){
         cuda::add(src[i],dest,dest);
       }
     }

     void inline TackerKCFImplParallel::updateProjectionMatrix(const Mat src, Mat & old_cov,Mat &  proj_matrix, double pca_rate, int compressed_sz,
                                                        std::vector<Mat> & layers_pca,std::vector<Scalar> & average, Mat pca_data, Mat new_cov, Mat w, Mat u, Mat vt) {
       GpuMat new_cov_gpu;

       double start = CycleTimer::currentSeconds();

       CV_Assert(compressed_sz<=src.channels());

       split(src,layers_pca);

       for (int i=0;i<src.channels();i++){
         average[i]=mean(layers_pca[i]);
         layers_pca[i]-=average[i];
       }

       merge(layers_pca,pca_data);
       pca_data=pca_data.reshape(1,src.rows*src.cols);

       pca_data_gpu.upload(pca_data);

       GpuMat src3;
       cuda::gemm(pca_data_gpu, pca_data_gpu, 1.0/(double)(src.rows*src.cols-1),
         src3, 0, new_cov_gpu, GEMM_1_T);
       new_cov_gpu.download(new_cov);

       if(old_cov.rows==0)old_cov=new_cov.clone();

       // calc PCA
       SVD::compute((1.0-pca_rate)*old_cov+pca_rate*new_cov, w, u, vt);

       proj_matrix=u(Rect(0,0,compressed_sz,src.channels())).clone();
       Mat proj_vars=Mat::eye(compressed_sz,compressed_sz,proj_matrix.type());
       for(int i=0;i<compressed_sz;i++){
         proj_vars.at<double>(i,i)=w.at<double>(i);
       }

       // update the covariance matrix
       old_cov=(1.0-pca_rate)*old_cov+pca_rate*proj_matrix*proj_vars*proj_matrix.t();
     }

     void inline TackerKCFImplParallel::compress(const Mat proj_matrix, const Mat src, Mat & dest, Mat & data, Mat & compressed) const {
       data=src.reshape(1,src.rows*src.cols);
       compressed=data*proj_matrix;
       dest=compressed.reshape(proj_matrix.cols,src.rows).clone();
     }

     bool TackerKCFImplParallel::getSubWindow(const Mat img, const Rect _roi, Mat& feat, Mat& patch, TrackerKCF::MODE desc) {

       Rect region=_roi;

       if((_roi.x+_roi.width<0)
         ||(_roi.y+_roi.height<0)
         ||(_roi.x>=img.cols)
         ||(_roi.y>=img.rows)
       )return false;

       if(_roi.x<0){region.x=0;region.width+=_roi.x;}
       if(_roi.y<0){region.y=0;region.height+=_roi.y;}
       if(_roi.x+_roi.width>img.cols)region.width=img.cols-_roi.x;
       if(_roi.y+_roi.height>img.rows)region.height=img.rows-_roi.y;
       if(region.width>img.cols)region.width=img.cols;
       if(region.height>img.rows)region.height=img.rows;

       patch=img(region).clone();

       int addTop,addBottom, addLeft, addRight;
       addTop=region.y-_roi.y;
       addBottom=(_roi.height+_roi.y>img.rows?_roi.height+_roi.y-img.rows:0);
       addLeft=region.x-_roi.x;
       addRight=(_roi.width+_roi.x>img.cols?_roi.width+_roi.x-img.cols:0);

       copyMakeBorder(patch,patch,addTop,addBottom,addLeft,addRight,BORDER_REPLICATE);
       if(patch.rows==0 || patch.cols==0)return false;

       switch(desc){
         case CN:
           CV_Assert(img.channels() == 3);
           extractCN(patch,feat);
           break;
         default: // GRAY
           if(img.channels()>1)
             cvtColor(patch,feat, CV_BGR2GRAY);
           else
             feat=patch;
           feat.convertTo(feat,CV_64F);
           feat=feat/255.0-0.5; // normalize to range -0.5 .. 0.5
           feat=feat.mul(hann); 
           break;
       }

       return true;

     }

     bool TackerKCFImplParallel::getSubWindow(const Mat img, const Rect _roi, Mat& feat, void (*f)(const Mat, const Rect, Mat& )) const{

       if((_roi.x+_roi.width<0)
         ||(_roi.y+_roi.height<0)
         ||(_roi.x>=img.cols)
         ||(_roi.y>=img.rows)
       )return false;

       f(img, _roi, feat);

       if(_roi.width != feat.cols || _roi.height != feat.rows){
         printf("error in customized function of features extractor!\n");
         printf("Rules: roi.width==feat.cols && roi.height = feat.rows \n");
       }

       Mat hann_win;
       std::vector<Mat> _layers;

       for(int i=0;i<feat.channels();i++)
         _layers.push_back(hann);

       merge(_layers, hann_win);

       feat=feat.mul(hann_win); // hann window filter

       return true;
     }

     __global__ void extractIndexKernel(const cuda::PtrStepSz<uchar3> input,
       cuda::PtrStep<ushort> output) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= 0 && x < input.cols && y >= 0 && y < input.rows) {
          uchar3 pixel = input(y,x);
          output.ptr(y)[x] = (floor((float)pixel.z/8)+32*floor((float)pixel.y/8)+32*32*floor((float)pixel.x/8));
        }
    }

    __global__ void extractCNKernel(const cuda::PtrStepSz<ushort> input,
      cuda::PtrStep<double[10]> output, const double *ColorNames) {
       int x = blockIdx.x * blockDim.x + threadIdx.x;
       int y = blockIdx.y * blockDim.y + threadIdx.y;
       int k = blockIdx.z * blockDim.z + threadIdx.z;


       if (x >= 0 && x < input.cols && y >= 0 && y < input.rows && k >= 0
         && k < 10) {
         short index = input(y,x);
         output.ptr(y)[x][k] = ColorNames[10*index + k];
         //output.ptr(y)[x] = (floor((float)pixel.z/8)+32*floor((float)pixel.y/8)+32*32*floor((float)pixel.x/8));
       }
   }

     void TackerKCFImplParallel::extractCN(Mat patch_data, Mat & cnFeatures) {
       if(cnFeatures.type() != CV_64FC(10)) {
         cnFeatures = Mat::zeros(patch_data.rows,patch_data.cols,CV_64FC(10));
       }

       patch_data_gpu.upload(patch_data);

       dim3 cthreads2d(32, 32);
       dim3 cblocks2d(
         static_cast<int>(std::ceil(patch_data_gpu.size().width /
           static_cast<double>(cthreads2d.x))),
         static_cast<int>(std::ceil(patch_data_gpu.size().height /
           static_cast<double>(cthreads2d.y))));

       extractIndexKernel<<<cblocks2d, cthreads2d>>>(patch_data_gpu, indexes_gpu);
       cudaSafeCall(cudaGetLastError());

       cuda::GpuMat cnFeatures_gpu;
       cuda::createContinuous(patch_data.size(), CV_64FC(10), cnFeatures_gpu);

       dim3 cthreads3d(32, 32, 1);
       dim3 cblocks3d(
         static_cast<int>(std::ceil(patch_data_gpu.size().width /
           static_cast<double>(cthreads3d.x))),
         static_cast<int>(std::ceil(patch_data_gpu.size().height /
           static_cast<double>(cthreads3d.y))),
         static_cast<int>(std::ceil(10 /
           static_cast<double>(cthreads3d.z))));

       extractCNKernel<<<cblocks3d, cthreads3d>>>(indexes_gpu, cnFeatures_gpu, ColorNames_gpu);
       cudaSafeCall(cudaGetLastError());

       cuda::multiply(cnFeatures_gpu, hann_cn_gpu, cnFeatures_gpu);

       cnFeatures_gpu.download(cnFeatures);
     }


     void TackerKCFImplParallel::denseGaussKernel(const double sigma, const Mat x_data, const Mat y_data, Mat & k_data,
                                           std::vector<Mat> & layers_data,std::vector<Mat> & xf_data,std::vector<Mat> & yf_data, std::vector<Mat> xyf_v, Mat xy, Mat xyf ) {

      int num_channels = x_data.channels();

       double normX = norm(x_data, NORM_L2SQR);
       double normY = norm(y_data, NORM_L2SQR);

       cv::cuda::Stream stream;

       split(x_data, layers_data);
       for (int i = 0; i < x_data.channels(); i++){
           layers_data_gpu[i].upload(layers_data[i], stream);
       }
       stream.waitForCompletion();

       cudafft2(x_data.channels(),xf_data_gpu,layers_data_gpu);

       split(y_data, layers_data);
       for (int i = 0; i < x_data.channels(); i++){
           layers_data_gpu[i].upload(layers_data[i], stream);
       }
       stream.waitForCompletion();

       cudafft2(y_data.channels(),yf_data_gpu,layers_data_gpu);


       pixelWiseMult(xf_data_gpu,yf_data_gpu,xyf_v_gpu,0,true);
       sumChannels(xyf_v_gpu,xyf_c_gpu);


       cudaifft2(xyf_c_gpu,xyf_r_gpu);

       xyf_r_gpu.download(xyf);


       if(params.wrap_kernel){
         shiftRows(xyf, x_data.rows/2);
         shiftCols(xyf, x_data.cols/2);
       }

       xy=(normX+normY-2*xyf)/(x_data.rows*x_data.cols*x_data.channels());

       for(int i=0;i<xy.rows;i++){
         for(int j=0;j<xy.cols;j++){
           if(xy.at<double>(i,j)<0.0)xy.at<double>(i,j)=0.0;
         }
       }

       double sig=-1.0/(sigma*sigma);
       xy=sig*xy;
       exp(xy, k_data);

     }

     void TackerKCFImplParallel::shiftRows(Mat& mat) const {

         Mat temp;
         Mat m;
         int _k = (mat.rows-1);
         mat.row(_k).copyTo(temp);
         for(; _k > 0 ; _k-- ) {
           m = mat.row(_k);
           mat.row(_k-1).copyTo(m);
         }
         m = mat.row(0);
         temp.copyTo(m);

     }

     void TackerKCFImplParallel::shiftRows(Mat& mat, int n) const {
         if( n < 0 ) {
           n = -n;
           flip(mat,mat,0);
           for(int _k=0; _k < n;_k++) {
             shiftRows(mat);
           }
           flip(mat,mat,0);
         }else{
           for(int _k=0; _k < n;_k++) {
             shiftRows(mat);
           }
         }
     }

     void TackerKCFImplParallel::shiftCols(Mat& mat, int n) const {
         if(n < 0){
           n = -n;
           flip(mat,mat,1);
           transpose(mat,mat);
           shiftRows(mat,n);
           transpose(mat,mat);
           flip(mat,mat,1);
         }else{
           transpose(mat,mat);
           shiftRows(mat,n);
           transpose(mat,mat);
         }
     }

     
     void TackerKCFImplParallel::calcResponse(const Mat alphaf_data, const Mat kf_data, Mat & response_data, Mat & spec_data) {

       mulSpectrums(alphaf_data,kf_data,spec_data,0,false);
       ifft2(spec_data,response_data);
     }

     void TackerKCFImplParallel::calcResponse(const Mat alphaf_data, const Mat _alphaf_den, const Mat kf_data, Mat & response_data, Mat & spec_data, Mat & spec2_data) {

       mulSpectrums(alphaf_data,kf_data,spec_data,0,false);

       double den;
       for(int i=0;i<kf_data.rows;i++){
         for(int j=0;j<kf_data.cols;j++){
           den=1.0/(_alphaf_den.at<Vec2d>(i,j)[0]*_alphaf_den.at<Vec2d>(i,j)[0]+_alphaf_den.at<Vec2d>(i,j)[1]*_alphaf_den.at<Vec2d>(i,j)[1]);
           spec2_data.at<Vec2d>(i,j)[0]=
             (spec_data.at<Vec2d>(i,j)[0]*_alphaf_den.at<Vec2d>(i,j)[0]+spec_data.at<Vec2d>(i,j)[1]*_alphaf_den.at<Vec2d>(i,j)[1])*den;
           spec2_data.at<Vec2d>(i,j)[1]=
             (spec_data.at<Vec2d>(i,j)[1]*_alphaf_den.at<Vec2d>(i,j)[0]-spec_data.at<Vec2d>(i,j)[0]*_alphaf_den.at<Vec2d>(i,j)[1])*den;
         }
       }

       ifft2(spec2_data,response_data);
     }

     void TackerKCFImplParallel::setFeatureExtractor(void (*f)(const Mat, const Rect, Mat&), bool pca_func){
       if(pca_func){
         extractor_pca.push_back(f);
         use_custom_extractor_pca = true;
       }else{
         extractor_npca.push_back(f);
         use_custom_extractor_npca = true;
       }
     }
}
