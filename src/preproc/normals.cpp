
#include <deque>
#include <assert.h>
#include <tdp/preproc/normals.h>
#include <tdp/utils/timer.hpp>
#include <tdp/clustering/managed_dpvmfmeans_simple.hpp>

namespace tdp {

#ifdef CUDA_FOUND
void Depth2Normals(
    const Image<float>& cuD,
    const Camera<float>& cam,
    Image<Vector3fda> cuN) {
  size_t wc = cuD.w_;
  size_t hc = cuD.h_;
  assert(wc%64 == 0);
  assert(hc%64 == 0);
  ManagedDeviceImage<float> cuDu(wc, hc);
  ManagedDeviceImage<float> cuDv(wc, hc);

  Gradient(cuD, cuDu, cuDv);

  float f = cam.params_(0);
  int uc = cam.params_(2);
  int vc = cam.params_(3);
  ComputeNormals(cuD, cuDu, cuDv, cuN, f, uc, vc);
}

//void Depth2Normals(
//    const Image<float>& cuD,
//    const Camera<float>& cam,
//    const SO3<float>& R_rc,
//    Image<Vector3fda> cuN) {
//  size_t wc = cuD.w_;
//  size_t hc = cuD.h_;
//  assert(wc%64 == 0);
//  assert(hc%64 == 0);
//  ManagedDeviceImage<float> cuDu(wc, hc);
//  ManagedDeviceImage<float> cuDv(wc, hc);
//
//  Gradient(cuD, cuDu, cuDv);
//
//  float f = cam.params_(0);
//  int uc = cam.params_(2);
//  int vc = cam.params_(3);
//  ComputeNormals(cuD, cuDu, cuDv, cuN, R_rc, f, uc, vc);
//}
#endif

bool NormalViaRMLS(
    const Image<Vector3fda>& pc, 
    uint32_t u0, uint32_t v0,
    uint32_t W, float inlierThr,
    Image<Vector4fda>& dpc, 
    Vector3fda& ni,
    float& curvature,
    Vector3fda& p
    ) {
  int32_t id0 = u0+v0*pc.w_;
  const Vector3fda& pc0 = pc[id0];
  if ( W <= u0 && u0 < pc.w_-W 
    && W <= v0 && v0 < pc.h_-W
    && IsValidData(pc0)) {

    Vector3fda pcvF = pc(u0,v0+1)-pc0;
    Vector3fda pcvB =-pc(u0,v0-1)+pc0;
    if (!IsValidData(pcvF) && !IsValidData(pcvB)) {
      return false;
    }
    Vector3fda pcuF = pc(u0+1,v0)-pc0;
    Vector3fda pcuB =-pc(u0-1,v0)+pc0;
    if (!IsValidData(pcuF) && !IsValidData(pcuB)) {
      return false;
    }
    Vector3fda pcv = Vector3fda::Zero(); 
    Vector3fda pcu = Vector3fda::Zero(); 
    int32_t id1=-1;
    int32_t id2=-1;

    if ((IsValidData(pcvF) && !IsValidData(pcvB))
        ||(IsValidData(pcvF) && IsValidData(pcvB) && pcvF.squaredNorm() < pcvB.squaredNorm())) {
      pcv = pcvF;
      id1 = u0+(v0+1)*pc.w_;
    }
    if ((IsValidData(pcvB) && !IsValidData(pcvF))
        ||(IsValidData(pcvB) && IsValidData(pcvF) && pcvF.squaredNorm() >= pcvB.squaredNorm())) {
      pcv = pcvB;
      id1 = u0+(v0-1)*pc.w_;
    }
    if ((IsValidData(pcuF) && !IsValidData(pcuB))
        ||(IsValidData(pcuF) && IsValidData(pcuB) && pcuF.squaredNorm() < pcuB.squaredNorm())) {
        pcu = pcuF;
        id2 = u0+1+v0*pc.w_;
    }
    if ((IsValidData(pcuB) && !IsValidData(pcuF))
        ||(IsValidData(pcuB) && IsValidData(pcuF) && pcuF.squaredNorm() >= pcuB.squaredNorm())) {
        pcu = pcuB;
        id2 = u0-1+v0*pc.w_;
    }
    Vector3fda n = (pcv.cross(pcu)).normalized();
    if ( fabs(1-n.norm()) > 1e-4) 
      return false;
//    std::cout << "dpc " << pcv.transpose() << ", " << pcu.transpose() << ", " << n.transpose() << std::endl;
//    std::cout << "ids " << id1 << ", " << id2 << std::endl;
//    std::cout << "pci " << pc[id1].transpose() << ", " << pc[id2].transpose() << ", " << pc0.transpose() << std::endl;
    tdp::Matrix3fda xOuter = pc0 * pc0.transpose() + pc[id1]*pc[id1].transpose() + pc[id2]*pc[id2].transpose();
    tdp::Vector3fda xSum = pc0 + pc[id1] + pc[id2];

//    std::cout << "init :" << n.transpose() << std::endl;
//    std::cout << pc[id1].transpose() << ", " << pc[id2].transpose() << ", " << pc0.transpose() << std::endl;

//    Vector3fda n = ((pc0-pc(u0+1,v0)).cross(pc0-pc(u0,v0+1))).normalized();
//    if (!IsValidData(n))
//      return false;
//    Eigen::Matrix3f xOuter = pc0 * pc0.transpose();
//    Eigen::Vector3f xSum = pc0;
//    int32_t id1 = u0+1+v0*pc.w_;
//    int32_t id2 = u0+(v0+1)*pc.w_;
//    xOuter += pc(u0+1,v0)*pc(u0+1,v0).transpose();
//    xOuter += pc(u0,v0+1)*pc(u0,v0+1).transpose();
//    xSum += pc(u0+1,v0);
//    xSum += pc(u0,v0+1);
    
    // TODO could also implement the incremental inverse computation
    // here - that gave 3x speedup over normal inverse
    // DONE Tried that and did not speed up

    // TODO could also try of speed, if I use a vector and just set
    // accepted datas second value to something huge; using partial
    // sort that should not be too bad?
    // DONE using std::vector was a bad idea as well
    std::deque<std::pair<int32_t, float>> errs;
    for (size_t u=u0-W; u<=u0+W; ++u) {
      for (size_t v=v0-W; v<=v0+W; ++v) {
        int32_t id = u+v*pc.w_;
        if (IsValidData(pc[id])
            && id != id0 && id != id1 && id != id2) {
          errs.emplace_back(id, n.dot(pc[id]));
        }
      }
    }

//    std::sort(errs.begin(), errs.end(), 
    std::partial_sort(errs.begin(), errs.begin()+1, errs.end(), 
        [&](const std::pair<int32_t,float>& l, 
          const std::pair<int32_t,float>& r){
          return l.second < r.second;
        });

    float a = n.dot(pc0);
//    std::cout << errs.size() << " " << a << std::endl; 
    int32_t i=0;
    while(errs.size() > 0 && errs.front().second - a < inlierThr) {
      for (int j=0; j < floor(pow(1.3,i)); ++j) {
        if (errs.size() == 0 || errs.front().second - a >= inlierThr) {
          break;
        }
//        std::cout << " -- " << j << " " << pc[errs.front().first].transpose() << std::endl;
        xOuter += pc[errs.front().first]*pc[errs.front().first].transpose();
        xSum += pc[errs.front().first];
        errs.pop_front();
      }
      n = (xOuter.ldlt().solve(xSum)).normalized();
      a = n.dot(pc0);
//      std::cout << "@" << errs.size() << ": " << n.transpose() << " " << xSum.transpose() << std::endl;
//      std::cout << xOuter << std::endl;

      for (auto& err : errs) {
        err.second = n.dot(pc[err.first]);
      }
      std::partial_sort(errs.begin(), errs.begin()+1, errs.end(), 
//      std::sort(errs.begin(), errs.end(), 
          [&](const std::pair<int32_t,float>& l, 
            const std::pair<int32_t,float>& r){
          return l.second < r.second;
          });
      ++i;
    }
    curvature = 0.; 
//    eig.eigenvalues().minCoeff(&id)/eig.eigenvalues().sum();
    ni = n * (n.dot(pc0)/pc0.norm()<0.?1.:-1.);
    p = pc0;
    return true;
  }
  return false;
}


bool NormalViaVoting(
    const Image<Vector3fda>& pc, 
    uint32_t u0, uint32_t v0,
    uint32_t W, float inlierThr,
    Image<Vector4fda>& dpc, 
    Vector3fda& ni,
    float& curvature,
    Vector3fda& p
    ) {
    float radiusStd =0;
    return NormalViaVoting(pc, u0, v0, W, inlierThr, dpc, ni,
        curvature, radiusStd, p);
}

bool NormalViaVoting(
    const Image<Vector3fda>& pc, 
    uint32_t u0, uint32_t v0,
    uint32_t W, float inlierThr,
    Image<Vector4fda>& dpc, 
    Vector3fda& ni,
    float& curvature,
    float& radiusStd,
    Vector3fda& p
    ) {
  if ( W <= u0 && u0 < pc.w_-W 
    && W <= v0 && v0 < pc.h_-W
    && IsValidData(pc(u0,v0))) {
    const Vector3fda& pc0 = pc(u0,v0);

    Vector3fda n = ((pc0-pc(u0+1,v0)).cross(pc0-pc(u0,v0+1))).normalized();
    if (!IsValidData(n))
      return false;
//    std::cout << "\t" << n.transpose() << std::endl;
    size_t N = 0;
    for (size_t u=u0-W; u<=u0+W; ++u) {
      for (size_t v=v0-W; v<=v0+W; ++v) {
        if (IsValidData(pc(u,v)) && u != u0 && v != v0) {
          dpc(u,v).topRows<3>() = pc0 - pc(u,v);
          dpc(u,v)(3) = dpc(u,v).topRows<3>().norm();
          ++N;
        } else {
          dpc(u,v)(3) = 0.;
        }
      }
    }
    if (N<4*W*W*inlierThr) 
      return false;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig;
    size_t Nprev = 0;
    Eigen::Matrix3f S;
    float orthoL = 0.;
    float orthoU = 0.;
    for (float dAng : {45./180.*M_PI,35./180.*M_PI,25./180.*M_PI,15./180.*M_PI}) {
      N = 0;
      S.fill(0.);
      p.fill(0.);
      orthoL = cos(0.5*M_PI-dAng);
      orthoU = cos(0.5*M_PI+dAng);
      for (size_t u=u0-W; u<=u0+W; ++u) {
        for (size_t v=v0-W; v<=v0+W; ++v) {
          if (dpc(u,v)(3) > 0.) {
            float ang = dpc(u,v).topRows<3>().dot(n);
            if (orthoU*dpc(u,v)(3) < ang && ang <= orthoL*dpc(u,v)(3)) {
//              S += dpc(u,v)*dpc(u,v).transpose();
              S(0,0) += pc(u,v)(0)*pc(u,v)(0);
              S(0,1) += pc(u,v)(0)*pc(u,v)(1);
              S(0,2) += pc(u,v)(0)*pc(u,v)(2);
              S(1,1) += pc(u,v)(1)*pc(u,v)(1);
              S(1,2) += pc(u,v)(1)*pc(u,v)(2);
              S(2,2) += pc(u,v)(2)*pc(u,v)(2);
              p += pc(u,v);
              N++;
            }
          }
        }
      }
      if (N<4*(W+1)*(W+1)*inlierThr) 
        return false;
      S(1,0) = S(0,1);
      S(2,0) = S(0,2);
      S(2,1) = S(1,2);
      int id = 0;
      n = (S.ldlt().solve(p)).normalized();
//      n = eig.eigenvectors().col(0).normalized();
//      std::cout << N << " " << Nprev << " " << 4*W*W << "\t" << n.transpose() << std::endl;
      if (N == Nprev) break;
      Nprev = N;
    }
    eig.computeDirect((S - p*p.transpose()/float(N))/float(N));
    ni = n*(n.dot(pc0)/pc0.norm()<0.?1.:-1.);
    curvature = 2.*(eig.eigenvalues()(1)*eig.eigenvalues()(2))/(eig.eigenvalues()(1)+eig.eigenvalues()(2));
    radiusStd = sqrtf(std::min(eig.eigenvalues()(1),eig.eigenvalues()(2)));
//    if (eig.eigenvectors().col(0).dot(n) < 0)
//      curvature *= -1;
//    float mu = 0;
//    for (size_t u=u0-W; u<=u0+W; ++u) {
//      for (size_t v=v0-W; v<=v0+W; ++v) {
//        if (dpc(u,v)(3) > 0. && u != u0 && v != v0) {
//          float ang = dpc(u,v).topRows<3>().dot(n);
//          if (orthoU*dpc(u,v)(3) < ang && ang <= orthoL*dpc(u,v)(3)) {
//            mu += dpc(u,v)(3); 
//          }
//        }
//      }
//    }
//    mu /= (N-1); // average dist to neighbors -> 
//    curvature = 2.*(ni.dot(pc0 - p))/(mu*mu);

//    tdp::Vector3fda mu = p;
//    N = 0;
//    p.fill(0.);
//    S.fill(0.);
//    for (size_t u=u0-W; u<=u0+W; ++u) {
//      for (size_t v=v0-W; v<=v0+W; ++v) {
//        if (dpc(u,v)(3) > 0.) {
//          S(0,0) += pc(u,v)(0)*pc(u,v)(0);
//          S(0,1) += pc(u,v)(0)*pc(u,v)(1);
//          S(0,2) += pc(u,v)(0)*pc(u,v)(2);
//          S(1,1) += pc(u,v)(1)*pc(u,v)(1);
//          S(1,2) += pc(u,v)(1)*pc(u,v)(2);
//          S(2,2) += pc(u,v)(2)*pc(u,v)(2);
//          p += pc(u,v);
//          N++;
//        }
//      }
//    }
//    S(1,0) = S(0,1);
//    S(2,0) = S(0,2);
//    S(2,1) = S(1,2);
//    eig.computeDirect(S - p*p.transpose()/float(N));
//    curvature = sqrtf(eig.eigenvalues().minCoeff());
//    p = mu;
    p = pc0;
    return true;
  }
  return false;
}

bool NormalViaClustering(
    const Image<Vector3fda>& pc, 
    uint32_t u0, 
    uint32_t v0,
    uint32_t W,
    Vector3fda& c
    ) {
  static std::random_device rd;
  static std::mt19937 gen(rd());

  ManagedDPvMFmeansSimple3fda dpvmf(cos(45.*M_PI/180.));

  if ( W <= u0 && u0 < pc.w_-W 
    && W <= v0 && v0 < pc.h_-W
    && IsValidData(pc(u0,v0))) {

    std::uniform_int_distribution<> uR(u0-W, u0+W);
    std::uniform_int_distribution<> vR(v0-W, v0+W);
    Vector3fda n;
    uint32_t N=0;
    float pi0=0., pi1=0.;
    while(pi0 - pi1 < 2.*sqrt(1./N)) {
      const Vector3fda& p0 = pc(uR(gen),vR(gen));
      const Vector3fda& p1 = pc(uR(gen),vR(gen));
      const Vector3fda& p2 = pc(uR(gen),vR(gen));
      n = ((p0-p1).cross(p0-p2)).normalized();
      if (!IsValidNormal(n)) continue;
      n *= (n(2)<0.?1.:-1.);
//      std::cout << n.transpose() << std::endl;
      dpvmf.addObservation(n);
      if (N % 100 == 30) {
        dpvmf.updateCenters();
        dpvmf.updateLabels();
      }
      N++;
      pi0 = 0.; pi1 = 0.;
      for (auto& Ni : dpvmf.GetNs()) if (Ni > pi0) pi0 = Ni;
      for (auto& Ni : dpvmf.GetNs()) if (Ni > pi1 && pi0!=Ni) pi1 = Ni;
      pi0 /= N;
      pi1 /= N;

//      if (N % 100 == 30) {
////      for (auto& Ni : dpvmf.GetNs()) std::cout << Ni << " ";
////      std::cout << std::endl;
//        std::cout << pi0 << " " << pi1  << " " << N 
//          << " " << pi0 - pi1 << " < " << 2.*sqrt(1./N) 
//          << std::endl;
//      }
    }
    auto itBest = std::max_element(dpvmf.GetNs().begin(), dpvmf.GetNs().end());
    c = dpvmf.GetCenter(std::distance(dpvmf.GetNs().begin(), itBest));
    c = c * (c(2)<0.?1.:-1.);
    return true;
  }
  return false;
}

void NormalsViaScatterAproxIntInvD(
    const Image<float>& rho, 
    const Image<Vector3fda>& ray, 
    const Image<Vector6dda>& outerInt, 
    uint32_t W, uint32_t step,
    Image<Vector3fda>& n) {
  float curv,rad;
  for(size_t u=W; u<n.w_-W; u+=step) {
    for(size_t v=W; v<n.h_-W; v+=step) {
      if(!NormalViaScatterAproxIntInvD(rho, ray, outerInt, u,v,W,n(u,v))) {
        n(u,v) << NAN,NAN,NAN;
      }
    }
  }
}

/// fastest method
bool NormalViaScatterAproxIntInvD(
    const Image<float>& rho, 
    const Image<Vector3fda>& ray, 
    const Image<Vector6dda>& outerInt, 
    uint32_t u0, 
    uint32_t v0,
    uint32_t W, 
    Vector3fda& n
    ) {
  if ( W < u0 && u0 < rho.w_-W 
    && W < v0 && v0 < rho.h_-W
    && rho(u0,v0) == rho(u0,v0)) {
//    const Vector3fda& pc0 = pc(u0,v0);
    tdp::Vector6fda outerS = (outerInt(u0-W-1, v0-W-1)
      - outerInt(u0-W-1, v0+W) 
      - outerInt(u0+W, v0-W-1) 
      + outerInt(u0+W, v0+W)).cast<float>();

    tdp::Matrix3fda S;
    S.row(0) = outerS.topRows<3>();
    S(1,1) = outerS(3);
    S(1,2) = outerS(4);
    S(2,2) = outerS(5);
    S(1,0) = S(0,1);
    S(2,0) = S(0,2);
    S(2,1) = S(1,2);

    tdp::Vector3fda xSum = tdp::Vector3fda::Zero();
    for (size_t v=v0-W; v<=v0+W; ++v) {
      float* rhoi = rho.RowPtr(v);
      tdp::Vector3fda* rayi = ray.RowPtr(v);
      for (size_t u=u0-W; u<=u0+W; ++u) {
        float d = rhoi[u];
        if (d == d && u != u0 && v != v0) {
          xSum += rayi[u]*d;
        }
      }
    }
    n = (S.ldlt().solve(xSum)).normalized();
    n *= (n.dot(ray(u0,v0))/ray(u0,v0).norm()<0.?1.:-1.);
    return true;
  }
  return false;
}

void NormalsViaScatterAproxInt(
    const Image<Vector3fda>& pc, 
    const Image<Vector3fda>& ray, 
    const Image<Vector6dda>& outerInt, 
    uint32_t W, uint32_t step,
    Image<Vector3fda>& n) {
  float curv,rad;
  for(size_t u=W; u<n.w_-W; u+=step) {
    for(size_t v=W; v<n.h_-W; v+=step) {
      if(!NormalViaScatterAproxInt(pc, ray, outerInt, u,v,W,n(u,v))) {
        n(u,v) << NAN,NAN,NAN;
      }
    }
  }
}

bool NormalViaScatterAproxInt(
    const Image<Vector3fda>& pc, 
    const Image<Vector3fda>& ray, 
    const Image<Vector6dda>& outerInt, 
    uint32_t u0, 
    uint32_t v0,
    uint32_t W, 
    Vector3fda& n
    ) {
  if ( W < u0 && u0 < pc.w_-W 
    && W < v0 && v0 < pc.h_-W
    && IsValidData(pc(u0,v0))) {
    const Vector3fda& pc0 = pc(u0,v0);
    tdp::Vector6fda outerS = (outerInt(u0-W-1, v0-W-1)
      - outerInt(u0-W-1, v0+W) 
      - outerInt(u0+W, v0-W-1) 
      + outerInt(u0+W, v0+W)).cast<float>();

    tdp::Matrix3fda S;
    S.row(0) = outerS.topRows<3>();
    S(1,1) = outerS(3);
    S(1,2) = outerS(4);
    S(2,2) = outerS(5);
    S(1,0) = S(0,1);
    S(2,0) = S(0,2);
    S(2,1) = S(1,2);

    tdp::Vector3fda xSum = tdp::Vector3fda::Zero();
    for (size_t v=v0-W; v<=v0+W; ++v) {
      tdp::Vector3fda* pi = pc.RowPtr(v);
      tdp::Vector3fda* rayi = ray.RowPtr(v);
      for (size_t u=u0-W; u<=u0+W; ++u) {
        float d = pi[u](2);
        if (d == d && u != u0 && v != v0) {
          xSum += rayi[u]/d;
        }
      }
    }
    n = (S.ldlt().solve(xSum)).normalized();
    n *= (n.dot(pc0)/pc0.norm()<0.?1.:-1.);
    return true;
  }
  return false;
}

void NormalsViaScatterAprox(
    const Image<Vector3fda>& pc, 
    const Image<Vector3fda>& ray, 
    const Image<Vector6fda>& outer, 
    uint32_t W, uint32_t step,
    Image<Vector3fda>& n) {
  float curv,rad;
  for(size_t u=W; u<n.w_-W; u+=step) {
    for(size_t v=W; v<n.h_-W; v+=step) {
      if(!NormalViaScatterAprox(pc, ray, outer, u,v,W,n(u,v))) {
        n(u,v) << NAN,NAN,NAN;
      }
    }
  }
}

bool NormalViaScatterAprox(
    const Image<Vector3fda>& pc, 
    const Image<Vector3fda>& ray, 
    const Image<Vector6fda>& outer, 
    uint32_t u0, 
    uint32_t v0,
    uint32_t W, 
    Vector3fda& n
    ) {
  if ( W <= u0 && u0 < pc.w_-W 
    && W <= v0 && v0 < pc.h_-W
    && IsValidData(pc(u0,v0))) {
    const Vector3fda& pc0 = pc(u0,v0);
    tdp::Matrix3fda S = tdp::Matrix3fda::Zero();
    tdp::Vector3fda xSum = tdp::Vector3fda::Zero();
    for (size_t u=u0-W; u<=u0+W; ++u) {
      for (size_t v=v0-W; v<=v0+W; ++v) {
        if (IsValidData(pc(u,v)) && u != u0 && v != v0) {
          S.row(0) += outer(u,v).topRows<3>();
          S(1,1) += outer(u,v)(3);
          S(1,2) += outer(u,v)(4);
          S(2,2) += outer(u,v)(5);
          xSum += ray(u,v)/pc(u,v)(2);
        }
      }
    }
    S(1,0) = S(0,1);
    S(2,0) = S(0,2);
    S(2,1) = S(1,2);
    n = (S.ldlt().solve(xSum)).normalized();
    n *= (n.dot(pc0)/pc0.norm()<0.?1.:-1.);
    return true;
  }
  return false;
}

bool NormalViaScatterUnconstrained(
    const Image<Vector3fda>& pc, 
    uint32_t u0, 
    uint32_t v0,
    uint32_t W, 
    Vector3fda& n,
    float& curvature,
    float& radiusStd
    ) {
  if ( W <= u0 && u0 < pc.w_-W 
    && W <= v0 && v0 < pc.h_-W
    && IsValidData(pc(u0,v0))) {
    const Vector3fda& pc0 = pc(u0,v0);
    Eigen::Matrix3f S = Eigen::Matrix3f::Zero();
    Eigen::Vector3f xSum = Eigen::Vector3f::Zero();
    size_t N = 0;
    for (size_t u=u0-W; u<=u0+W; ++u) {
      for (size_t v=v0-W; v<=v0+W; ++v) {
        if (IsValidData(pc(u,v)) && u != u0 && v != v0) {
          S(0,0) += pc(u,v)(0)*pc(u,v)(0);
          S(0,1) += pc(u,v)(0)*pc(u,v)(1);
          S(0,2) += pc(u,v)(0)*pc(u,v)(2);
          S(1,1) += pc(u,v)(1)*pc(u,v)(1);
          S(1,2) += pc(u,v)(1)*pc(u,v)(2);
          S(2,2) += pc(u,v)(2)*pc(u,v)(2);
          xSum += pc(u,v);
          N ++;
        }
      }
    }
    if (N<3) 
      return false;

    S(1,0) = S(0,1);
    S(2,0) = S(0,2);
    S(2,1) = S(1,2);
    n = (S.ldlt().solve(xSum)).normalized();
    n *= (n.dot(pc0)/pc0.norm()<0.?1.:-1.);
//    Eigen::Vector3f eig = ((S - xSum*xSum.transpose()/float(N))/float(N)).eigenvalues().real();
//    curvature = 2.*(eig(1)*eig(2))/(eig(1)+eig(2));
//    radiusStd = sqrtf(std::min(eig(1),eig(2)));
    return true;
  }
  return false;
}

void NormalsViaScatterUnconstrained(
    const Image<Vector3fda>& pc, 
    uint32_t W, uint32_t step,
    Image<Vector3fda>& n) {
  float curv,rad;
  for(size_t u=W; u<n.w_-W; u+=step) {
    for(size_t v=W; v<n.h_-W; v+=step) {
      if(!NormalViaScatterUnconstrained(pc, u,v,W,n(u,v),curv,rad)) {
        n(u,v) << NAN,NAN,NAN;
      }
    }
  }
}

bool NormalViaScatter(
    const Image<Vector3fda>& pc, 
    uint32_t u0, 
    uint32_t v0,
    uint32_t W, 
    Vector3fda& c
    ) {
  if ( W <= u0 && u0 < pc.w_-W 
    && W <= v0 && v0 < pc.h_-W
    && IsValidData(pc(u0,v0))) {
    const Vector3fda& pc0 = pc(u0,v0);
    Eigen::Matrix3f S = Eigen::Matrix3f::Zero();
    size_t N = 0;
    for (size_t u=u0-W; u<u0+W; ++u) {
      for (size_t v=v0-W; v<v0+W; ++v) {
        if (IsValidData(pc(u,v)) && u != u0 && v != v0) {
          S += (pc0-pc(u,v))*(pc0-pc(u,v)).transpose();
          N ++;
        }
      }
    }
    if (N<3) 
      return false;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig(S);
    int id = 0;
    float eval = eig.eigenvalues().minCoeff(&id);
    c = eig.eigenvectors().col(id).normalized();
    c *= (c(2)<0.?1.:-1.);
    return true;
  }
  return false;
}

void NormalsViaScatter(
    const Image<Vector3fda>& pc, 
    uint32_t W, uint32_t step,
    Image<Vector3fda>& n) {
  for(size_t u=W; u<n.w_-W; u+=step) {
    for(size_t v=W; v<n.h_-W; v+=step) {
      if(!NormalViaScatter(pc, u,v,W,n(u,v))) {
        n(u,v) << NAN,NAN,NAN;
      }
    }
  }
}

void NormalsViaVoting(
    Image<Vector3fda>& pc, 
    uint32_t W, uint32_t step,
    float inlierThr, 
    Image<Vector4fda>& dpc,
    Image<Vector3fda>& n,
    Image<float>& curv) {
  Vector3fda p;
  for(size_t u=W; u<n.w_-W; u+=step) {
    for(size_t v=W; v<n.h_-W; v+=step) {
      if(!NormalViaVoting(pc, u,v,W,inlierThr, dpc, n(u,v), curv(u,v), p)) {
        n(u,v) << NAN,NAN,NAN;
        curv(u,v) = NAN;
      } else {
        pc(u,v) = p;
      }
    }
  }
}

void NormalsViaRMLS(
    Image<Vector3fda>& pc, 
    uint32_t W, uint32_t step,
    float inlierThr, 
    Image<Vector4fda>& dpc,
    Image<Vector3fda>& n,
    Image<float>& curv) {
  Vector3fda p;
  for(size_t u=W; u<n.w_-W; u+=step) {
    for(size_t v=W; v<n.h_-W; v+=step) {
      if(!NormalViaRMLS(pc, u,v,W,inlierThr, dpc, n(u,v), curv(u,v), p)) {
        n(u,v) << NAN,NAN,NAN;
        curv(u,v) = NAN;
      } else {
        pc(u,v) = p;
      }
    }
  }
}

void NormalsViaClustering(
    const Image<Vector3fda>& pc, 
    uint32_t W, uint32_t step,
    Image<Vector3fda>& n) {
  for(size_t u=W; u<n.w_-W; u+=step) {
    for(size_t v=W; v<n.h_-W; v+=step) {
      if(!NormalViaClustering(pc, u,v,W,n(u,v))) {
        n(u,v) << NAN,NAN,NAN;
      }
    }
  }
}


}
