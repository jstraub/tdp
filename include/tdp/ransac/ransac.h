#pragma once
#include <algorithm>
#include <vector>
#include <tdp/manifold/SE3.h>
#include <tdp/data/image.h>
#include <tdp/eigen/dense.h>

namespace tdp {

template<class T>
class Model {
 public:
  Model() {}
  virtual ~Model() {}
  virtual bool Compute(const Image<T>& dataA, const Image<T>& dataB,
      const Image<Vector2ida>& assoc, std::vector<uint32_t>& ids,
      SE3f& T_ab) = 0;
  virtual size_t ConsensusSize(const Image<T>& dataA, const Image<T>& dataB,
      const Image<Vector2ida>& assoc, std::vector<uint32_t>& ids,
      const SE3f& T_ab, float thr) = 0;
 private:
};

class P3P : public Model<Vector3fda> {
 public:
  P3P() {}
  virtual ~P3P() {}
  virtual bool Compute(const Image<Vector3fda>& pcA, 
      const Image<Vector3fda>& pcB, 
      const Image<Vector2ida>& assoc, 
      std::vector<uint32_t>& ids, SE3f& T_ab) {
    Eigen::Vector3d meanA(0,0,0);
    Eigen::Vector3d meanB(0,0,0);
    for (size_t i=0; i<ids.size(); ++i) {
      meanA += pcA[assoc[ids[i]](0)].cast<double>();
      meanB += pcB[assoc[ids[i]](1)].cast<double>();
    }
    meanA /= ids.size();
    meanB /= ids.size();
    Eigen::Matrix3d outer = Eigen::Matrix3d::Zero();
    for (size_t i=0; i<ids.size(); ++i) {
      outer += (pcB[assoc[ids[i]](1)].cast<double>()-meanB)
        * (pcA[assoc[ids[i]](0)].cast<double>()-meanA).transpose();
    }
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(outer, Eigen::ComputeFullU |
        Eigen::ComputeFullV);
    double detUV = (svd.matrixV()*svd.matrixU().transpose()).determinant();
//    std::cout << detUV << "; " << svd.rank() 
//      << "; " << svd.singularValues().transpose() 
//      << std::endl;
//    std::cout << svd.matrixV() << std::endl << svd.matrixU() << std::endl;
    Eigen::Vector3d diag(1,1, detUV);
    Eigen::Matrix3d R_ab = svd.matrixV()*diag.asDiagonal()
      *svd.matrixU().transpose();
    Eigen::Vector3d t_ab = meanA - R_ab*meanB;
//    std::cout << R_ab << std::endl << t_ab.transpose() << std::endl;
    T_ab = SE3f(R_ab.cast<float>(), t_ab.cast<float>());
    return true;
  }

  virtual size_t ConsensusSize(const Image<Vector3fda>& pcA, 
      const Image<Vector3fda>& pcB,
      const Image<Vector2ida>& assoc, std::vector<uint32_t>& inlierIds, 
      const SE3f& T_ab, float thr) {
    size_t numInliers = 0;
    inlierIds.clear();
    inlierIds.reserve(assoc.Area());
    for (size_t i=0; i<assoc.Area(); ++i) {
      float dist = (pcA[assoc[i](0)] - T_ab * pcB[assoc[i](1)]).norm();
      if (dist < thr) {
        numInliers ++; 
        inlierIds.push_back(i);
      }
    }
    return numInliers;
  }
 private:
};

/* RANSAC algorithm
 */
template<class T>
class Ransac
{
public:
  Ransac(Model<T>* model) : model_(model)
  {}
  ~Ransac()
  {}

  SE3f Compute(const Image<T>& pcA, const Image<T>& pcB,
      const Image<Vector2ida>& assoc,
      size_t maxIt, float thr, size_t& numInliers)
  {
    SE3f T_ab;
    SE3f maxT_ab;
    size_t maxInlier = 0;

    std::vector<uint32_t> ids(assoc.Area());
    std::iota(ids.begin(), ids.end(), 0);

    std::vector<uint32_t> idsInlier;
    for (size_t it=0; it<maxIt; ++it) {
      std::random_shuffle(ids.begin(), ids.end());
      std::vector<uint32_t> idsModel(ids.begin(), ids.begin()+3);
      // compute model from the sampled datapoints
      if(!model_->Compute(pcA, pcB, assoc, idsModel, T_ab)) continue;
//      std::cout << T_ab << std::endl;
      size_t nInlier=model_->ConsensusSize(pcA, pcB, assoc, idsInlier, T_ab, thr);
//      std::cout<<"Ransac::find: nInlier="<<nInlier<<std::endl;
      // model is good enough -> remember
      if(nInlier > maxInlier){
        maxInlier = nInlier;
        maxT_ab = T_ab; 
//        std::cout<<"nInlier="<<nInlier<<std::endl;
//        std::cout<<"Ransac::find: LatestT_wc:"<<std::endl<<LatestT_wc;
      }
    }
    numInliers = model_->ConsensusSize(pcA, pcB, assoc, idsInlier, maxT_ab, thr);
    model_->Compute(pcA, pcB, assoc, idsInlier, T_ab);
//    std::cout<<"Inliers of best model: " << maxInlier << std::endl
//      << "Model: " << std::endl << maxT_ab << std::endl;
    return T_ab;
  }

  uint32_t getInlierCount() const { return mInlierCount;};

  Model<T>* model_;

private:
  uint32_t mInlierCount;
};

//template<class Model, class Desc>
//class Prosac
//{
//public:
//  Prosac(Random& rnd, const Model& model, const RansacParams& ransacParams)
//  : mModel(model), mRansacParams(ransacParams)
//  {};
//  ~Prosac()
//  {};
//
//  //PTAM uses T_wc for all computations!
//  SE3<double> find(vector<Assoc<Desc,uint32_t> >& pairing)
//    {
//   return findAmongBest(pairing,0);
//    };
//  //PTAM uses T_wc for all computations!
//  // nBest is ignored - just for Ransac compatibility
//  SE3<double> findAmongBest(vector<Assoc<Desc,uint32_t> >& pairing, uint32_t nBest=0)
//  {
//    sort(pairing.begin(),pairing.end());
//
//    if(pairing.size() < mModel.NumDataPerModel()) return SE3<double>();
//    TooN::SE3<double> T_wc;
//    TooN::SE3<double> LatestT_wc;
//    int32_t maxInlier=-1;
//    uint32_t Tmax=mRansacParams.getMaxIterations();
//    uint32_t m=mModel.NumDataPerModel();
//    uint32_t TN=nIterationsRansac(P_GOOD_SAMPLE, OUTLIERS_PROPORTION, m,INT_MAX); // iterations necessary according to RANSAC formula
//    uint32_t N=pairing.size();
//    uint32_t n=m;
//    uint32_t TnPrime=1, t=0;
//    uint32_t InlierNStar=0, kNStar=TN;
//    double Tn=TN;
//    for(uint32_t i=0; i<m; i++) {
//        Tn *= (double)(n-i)/(N-i);
//    }
//    std::cout<<"   #pairings="<<pairing.size()<<" Tmax="<<Tmax<<std::endl;
//    while(t<=kNStar && t<TN && t<Tmax)
//    {
//      if((t>TnPrime)&&(n<N))
//      { // adjust sampling size n
//        double TnNext=(Tn*(n+1))/(n+1-m);
//        n++;
//        TnPrime=TnPrime + uint32_t(ceil(TnNext-Tn));
//        Tn=TnNext;
//      }
//
//      std::vector<Assoc<Desc,uint32_t> > sample; sample.reserve(m);
//      if(t>TnPrime)
//      { // standard RANSAC - draw from complete set
//        mRnd.resetRepetitions();
//        while(sample.size()<m){
//          uint32_t id=mRnd.drawWithoutRepetition(N);
//          bool equalQueryFeatures=false;
//          for(uint32_t i=0; i<sample.size(); ++i)
//            if(sample[i].q==pairing[id].q)
//            {
//              equalQueryFeatures=true;
//              break;
//            }
//          // duplicates are possible since lsh might match a query feature to multiple features with equal dist
//          if(!equalQueryFeatures){
//            assert(pairing[id].m!=NULL);
//            sample.push_back(pairing[id]);
//          }else{
//            std::cout<<"Prosac: Avoided duplicated query feature in sample for model generation!"<<std::endl;
//          }
//        };
////        std::cout<<"@"<<t<<": draw "<<m<<" out of "<<n<<std::endl;
//      }else{
//        // prosac - draw from nth correspondence and the set U_{n-1}
//        sample.push_back(pairing[n]);
//        mRnd.resetRepetitions();
//        uint32_t j=1;
//        while(sample.size()<m){
//          uint32_t id=mRnd.drawWithoutRepetition(n-1);
//          bool equalQueryFeatures=false;
//          for(uint32_t i=0; i<sample.size(); ++i)
//            if(sample[i].q==pairing[id].q)
//            {
//              equalQueryFeatures=true;
//              break;
//            }
//          // duplicates are possible since lsh might match a query feature to multiple features with equal dist
//          if(!equalQueryFeatures){
//            assert(pairing[id].m!=NULL);
//            sample.push_back(pairing[id]);
//          }else{
//            std::cout<<"Prosac: Avoided duplicated query feature in sample for model generation!"<<std::endl;
//          }
//          // increase n in order to be able to find a model
//          if(j>=n-1) ++n;
//          ++j;
//        };
////        std::cout<<"@"<<t<<": draw "<<m<<" out of "<<n<<std::endl;
//      }
//
//      // compute model from the sampled datapoints
////      std::cout<<"Ransac::find: compute model! iteration "<<i<<std::endl;
//      ++t;
//      if(!mModel.compute(sample,LatestT_wc)) continue;
//      uint32_t nInlier=mModel.consensusSize(LatestT_wc.inverse(),pairing);;
////      std::cout<<"Ransac::find: nInlier="<<nInlier<<std::endl;
//
//      // model is good enough -> remember
////        mModel.refineModel(cs);
////        cs.clear();
////        mModel.consensusSet(data,sample,cs);
////        uint32_t nInlier=cs.size();
////        std::cout<<"nInlier="<<nInlier<<std::endl;
//
//      if(int32_t(nInlier)>maxInlier){
//        maxInlier=nInlier;
//        T_wc=LatestT_wc;
//        if(maxInlier>int32_t(InlierNStar))
//        { // estimate new upper bound on the number of iterations necessary to find a good model
//          // (maximality criterion from PROSAC paper)
//          InlierNStar=maxInlier;
//          kNStar=nIterationsRansac(1.0-ETA0,1.0-double(maxInlier)/double(N), m,Tmax);
//        }
////        std::cout<<"nInlier="<<nInlier<<std::endl;
////        std::cout<<"Ransac::find: LatestT_wc:"<<std::endl<<LatestT_wc;
//      }
//    }
////    if(! maxInlier>=mMinInliers){
////      return SE3<double>();
////    }else{
//      std::cout<<"Inliers of best model: "<<maxInlier<<" #trials="<<t<<std::endl
//          <<"Model: "<<std::endl<<T_wc;
//      mInlierCount=maxInlier;
//      return T_wc;
////    }
//  }
//
//  uint32_t getInlierCount() const { return mInlierCount;};
//
//  Random& mRnd;
//  Model mModel;
//
//  const RansacParams mRansacParams;
//
//private:
//  uint32_t mInlierCount;
//
////  maximality – the probability that a solution with more
//// than In∗ inliers in Un∗ exists and was not found after
//// k samples is smaller than η0 (typically set to 5%).
//  static const double ETA0=0.01;
//  static const double P_GOOD_SAMPLE=0.95;
//  static const double OUTLIERS_PROPORTION=0.95;
//
//  uint32_t nIterationsRansac(double pSampleAllInliers,
//      double pOutliers, uint32_t nSamples, uint32_t Nmax)
//  {
//    int32_t N=int32_t(ceil(log(1.0-pSampleAllInliers)/log(1.0-pow(1.0-pOutliers,double(nSamples)))));
//    if (N>=0 && uint32_t(N)<Nmax)
//      return uint32_t(N);
//    else
//      return Nmax;
//  };
//};

}
