#pragma once
#include <algorithm>
#include <vector>
#include <tdp/manifold/SE3.h>
#include <tdp/data/image.h>
#include <tdp/eigen/dense.h>
#include <tdp/features/brief.h>

namespace tdp {

template<class T>
class Model {
 public:
  Model() {}
  virtual ~Model() {}
  virtual bool Compute(const Image<T>& dataA, const Image<T>& dataB,
      const Image<int32_t>& assocBA, std::vector<uint32_t>& ids,
      SE3f& T_ab) = 0;
  virtual size_t ConsensusSize(const Image<T>& dataA, const Image<T>& dataB,
      const Image<int32_t>& assocBA, std::vector<uint32_t>& ids,
      const SE3f& T_ab, float thr) = 0;

  virtual bool Compute(const std::vector<T>& dataA, const std::vector<T>& dataB,
      const std::vector<int32_t>& assocBA, std::vector<uint32_t>& ids,
      SE3f& T_ab) = 0;
  virtual size_t ConsensusSize(const std::vector<T>& dataA, 
      const std::vector<T>& dataB,
      const std::vector<int32_t>& assocBA, std::vector<uint32_t>& ids,
      const SE3f& T_ab, float thr) = 0;
 private:
};

template<class T>
class P3P : public Model<T> {
 public:
  P3P() {}
  virtual ~P3P() {}

  static SE3f StatisticsToPose(const Eigen::Vector3d& meanA, 
      const Eigen::Vector3d& meanB, const Eigen::Matrix3d& cov) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov, Eigen::ComputeFullU |
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
    return SE3f(R_ab.cast<float>(), t_ab.cast<float>());
  }
};

class P3PBrief : public P3P<Brief> {
 public:
  P3PBrief() {}
  virtual ~P3PBrief() {}

  virtual bool Compute(const Image<Brief>& dataA, 
      const Image<Brief>& dataB, 
      const Image<int32_t>& assocBA, 
      std::vector<uint32_t>& ids, SE3f& T_ab) {
    Eigen::Vector3d meanA(0,0,0);
    Eigen::Vector3d meanB(0,0,0);
    for (size_t i=0; i<ids.size(); ++i) {
      meanA += dataA[ids[i]].p_c_.cast<double>();
      meanB += dataB[assocBA[ids[i]]].p_c_.cast<double>();
    }
    meanA /= ids.size();
    meanB /= ids.size();
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (size_t i=0; i<ids.size(); ++i) {
      cov += (dataB[ids[i]].p_c_.cast<double>()-meanB)
        * (dataA[assocBA[ids[i]]].p_c_.cast<double>()-meanA).transpose();
    }
    T_ab = StatisticsToPose(meanA, meanB, cov);
    return true;
  }

  virtual size_t ConsensusSize(const Image<Brief>& dataA, 
      const Image<Brief>& dataB,
      const Image<int32_t>& assocBA, std::vector<uint32_t>& inlierIds, 
      const SE3f& T_ab, float thr) {
    size_t numInliers = 0;
    inlierIds.clear();
    inlierIds.reserve(assocBA.Area());
    for (size_t i=0; i<assocBA.Area(); ++i) {
      float dist = (dataA[i].p_c_ - T_ab * dataB[assocBA[i]].p_c_).norm();
      if (dist < thr) {
        numInliers ++; 
        inlierIds.push_back(i);
      }
    }
    return numInliers;
  }

  virtual bool Compute(const std::vector<Brief>& dataA, 
      const std::vector<Brief>& dataB, 
      const std::vector<int32_t>& assocBA, 
      std::vector<uint32_t>& ids, SE3f& T_ab) {
    Eigen::Vector3d meanA(0,0,0);
    Eigen::Vector3d meanB(0,0,0);
    for (size_t i=0; i<ids.size(); ++i) {
      meanA += dataA[ids[i]].p_c_.cast<double>();
      meanB += dataB[assocBA[ids[i]]].p_c_.cast<double>();
    }
    meanA /= ids.size();
    meanB /= ids.size();
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (size_t i=0; i<ids.size(); ++i) {
      cov += (dataB[ids[i]].p_c_.cast<double>()-meanB)
        * (dataA[assocBA[ids[i]]].p_c_.cast<double>()-meanA).transpose();
    }
    T_ab = StatisticsToPose(meanA, meanB, cov);
    return true;
  }

  virtual size_t ConsensusSize(const std::vector<Brief>& dataA, 
      const std::vector<Brief>& dataB,
      const std::vector<int32_t>& assocBA, std::vector<uint32_t>& inlierIds, 
      const SE3f& T_ab, float thr) {
    size_t numInliers = 0;
    inlierIds.clear();
    inlierIds.reserve(assocBA.size());
    for (size_t i=0; i<assocBA.size(); ++i) {
      float dist = (dataA[i].p_c_ - T_ab * dataB[assocBA[i]].p_c_).norm();
      if (dist < thr) {
        numInliers ++; 
        inlierIds.push_back(i);
      }
    }
    return numInliers;
  }
 private:
};

class P3PVector3 : public P3P<Vector3fda> {
 public:
  P3PVector3() {}
  virtual ~P3PVector3() {}

  virtual bool Compute(const Image<Vector3fda>& pcA, 
      const Image<Vector3fda>& pcB, 
      const Image<int32_t>& assocBA, 
      std::vector<uint32_t>& ids, SE3f& T_ab) {
    Eigen::Vector3d meanA(0,0,0);
    Eigen::Vector3d meanB(0,0,0);
    for (size_t i=0; i<ids.size(); ++i) {
      meanA += pcA[ids[i]].cast<double>();
      meanB += pcB[assocBA[ids[i]]].cast<double>();
    }
    meanA /= ids.size();
    meanB /= ids.size();
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (size_t i=0; i<ids.size(); ++i) {
      cov += (pcB[ids[i]].cast<double>()-meanB)
        * (pcA[assocBA[ids[i]]].cast<double>()-meanA).transpose();
    }
    T_ab = StatisticsToPose(meanA, meanB, cov);
    return true;
  }

  virtual size_t ConsensusSize(const Image<Vector3fda>& pcA, 
      const Image<Vector3fda>& pcB,
      const Image<int32_t>& assocBA, std::vector<uint32_t>& inlierIds, 
      const SE3f& T_ab, float thr) {
    size_t numInliers = 0;
    inlierIds.clear();
    inlierIds.reserve(assocBA.Area());
    for (size_t i=0; i<assocBA.Area(); ++i) {
      float dist = (pcA[i] - T_ab * pcB[assocBA[i]]).norm();
      if (dist < thr) {
        numInliers ++; 
        inlierIds.push_back(i);
      }
    }
    return numInliers;
  }

  virtual bool Compute(const std::vector<Vector3fda>& pcA, 
      const std::vector<Vector3fda>& pcB, 
      const std::vector<int32_t>& assocBA, 
      std::vector<uint32_t>& ids, SE3f& T_ab) {
    Eigen::Vector3d meanA(0,0,0);
    Eigen::Vector3d meanB(0,0,0);
    for (size_t i=0; i<ids.size(); ++i) {
      meanA += pcA[ids[i]].cast<double>();
      meanB += pcB[assocBA[ids[i]]].cast<double>();
    }
    meanA /= ids.size();
    meanB /= ids.size();
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (size_t i=0; i<ids.size(); ++i) {
      cov += (pcB[ids[i]].cast<double>()-meanB)
        * (pcA[assocBA[ids[i]]].cast<double>()-meanA).transpose();
    }
    T_ab = StatisticsToPose(meanA, meanB, cov);
    return true;
  }

  virtual size_t ConsensusSize(const std::vector<Vector3fda>& pcA, 
      const std::vector<Vector3fda>& pcB,
      const std::vector<int32_t>& assocBA, std::vector<uint32_t>& inlierIds, 
      const SE3f& T_ab, float thr) {
    size_t numInliers = 0;
    inlierIds.clear();
    inlierIds.reserve(assocBA.size());
    for (size_t i=0; i<assocBA.size(); ++i) {
      float dist = (pcA[i] - T_ab * pcB[assocBA[i]]).norm();
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

  SE3f Compute(const Image<T>& dataA, const Image<T>& dataB,
      Image<int32_t>& assocBA, size_t maxIt, float thr, size_t& numInliers)
  {
    SE3f T_ab;
    SE3f maxT_ab;
    size_t maxInlier = 0;

    std::vector<uint32_t> ids(assocBA.Area());
    std::iota(ids.begin(), ids.end(), 0);

    std::vector<uint32_t> idsInlier;
    for (size_t it=0; it<maxIt; ++it) {
      std::random_shuffle(ids.begin(), ids.end());
      std::vector<uint32_t> idsModel(ids.begin(), ids.begin()+3);
      // compute model from the sampled datapoints
      if(!model_->Compute(dataA, dataB, assocBA, idsModel, T_ab)) continue;
      size_t nInlier=model_->ConsensusSize(dataA, dataB, assocBA,
          idsInlier, T_ab, thr);
      // model is good enough -> remember
      if(nInlier > maxInlier){
        maxInlier = nInlier;
        maxT_ab = T_ab; 
      }
    }
    numInliers = model_->ConsensusSize(dataA, dataB, assocBA,
        idsInlier, maxT_ab, thr);
    model_->Compute(dataA, dataB, assocBA, idsInlier, T_ab);

    std::sort(idsInlier.begin(), idsInlier.end());
    size_t j=0;
    for (size_t i=0; i<assocBA.Area(); ++i) {
      if (i == idsInlier[j]) {
        ++j;
      } else {
        assocBA[i] = -1;
      }
    }
//    std::cout<<"Inliers of best model: " << maxInlier << std::endl
//      << "Model: " << std::endl << maxT_ab << std::endl;
    return T_ab;
  }

  SE3f Compute(const std::vector<T>& dataA, const std::vector<T>& dataB,
      std::vector<int32_t>& assocBA, size_t maxIt, float thr, size_t& numInliers)
  {
    SE3f T_ab;
    SE3f maxT_ab;
    size_t maxInlier = 0;

    std::vector<uint32_t> ids(assocBA.size());
    std::iota(ids.begin(), ids.end(), 0);

    std::vector<uint32_t> idsInlier;
    for (size_t it=0; it<maxIt; ++it) {
      std::random_shuffle(ids.begin(), ids.end());
      std::vector<uint32_t> idsModel(ids.begin(), ids.begin()+3);
      // compute model from the sampled datapoints
      if(!model_->Compute(dataA, dataB, assocBA, idsModel, T_ab)) continue;
      size_t nInlier=model_->ConsensusSize(dataA, dataB, assocBA,
          idsInlier, T_ab, thr);
      // model is good enough -> remember
      if(nInlier > maxInlier){
        maxInlier = nInlier;
        maxT_ab = T_ab; 
      }
    }
    numInliers = model_->ConsensusSize(dataA, dataB, assocBA,
        idsInlier, maxT_ab, thr);
    model_->Compute(dataA, dataB, assocBA, idsInlier, T_ab);

    std::sort(idsInlier.begin(), idsInlier.end());
    size_t j=0;
    for (size_t i=0; i<assocBA.size(); ++i) {
      if (i == idsInlier[j]) {
        ++j;
      } else {
        assocBA[i] = -1;
      }
    }
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
