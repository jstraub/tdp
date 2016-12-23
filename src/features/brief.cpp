#include <tdp/features/brief.h>
namespace tdp {

  int ClosestBrief(const Brief& a, const Image<Brief>& bs, int* dist) {
    int minId = -1;
    int minDist = 257;
    if (a.IsValid()) {
      for (size_t i=0; i<bs.w_; ++i) {
        // iterate over pyramid levels
        for (size_t j=0; j<bs.h_; ++j) 
          if (bs(i,j).IsValid()) {
            int dist = Distance(a.desc_, bs(i,j).desc_);
            if (dist < minDist) {
              minDist = dist;
              minId = i;
            }
          }
      }
    }
    if (dist) *dist = minDist;
    return minId;
  }

  int ClosestBrief(const Brief& a, const std::vector<Brief*>& bs, int* dist) {
    int minId = -1;
    int minDist = 257;
    if (a.IsValid()) {
      for (size_t i=0; i<bs.size(); ++i) 
        if (bs[i]->IsValid()) {
          int dist = Distance(a.desc_, bs[i]->desc_);
          if (dist < minDist) {
            minDist = dist;
            minId = i;
          }
        }
    }
    if (dist) *dist = minDist;
    return minId;
  }
  
  bool ExtractBrief(const Image<uint8_t>& grey, 
      Brief& brief) {
    int32_t x = brief.pt_(0);
    int32_t y = brief.pt_(1);
//    if (!grey.Inside(x-16,y-16) || !grey.Inside(x+15, y+15)) {
    if (!grey.Inside(x-18,y-18) || !grey.Inside(x+18, y+18)) {
      brief.desc_.fill(0);
      return false;
    }
//    Image<uint8_t> patch = grey.GetRoi(x-16, y-16, 32,32);
    Image<uint8_t> patch = grey.GetRoi(x-18, y-18, 37,37);
    int intOrient = (int)floor((
      brief.orientation_ < 0. ? brief.orientation_ + 2*M_PI : brief.orientation_
        )/M_PI*180./12.);
    bool ret = ExtractOrb(patch, brief.desc_, intOrient);
    return ret;
  }

  void ExtractBrief(const Image<uint8_t>& grey, 
      const Image<Vector2ida>& pts,
      uint32_t frame, 
      ManagedHostImage<Brief>& briefs) {
    briefs.Reinitialise(pts.w_, 1);
    for (size_t i=0; i<pts.Area(); ++i) {
      briefs[i].pt_ = pts[i];
      briefs[i].lvl_= 0;
      briefs[i].frame_ = frame;
      briefs[i].orientation_= 0.;
      if(!tdp::ExtractBrief(grey, briefs[i])) {
        std::cout << pts[i].transpose() << " could not be extracted" << std::endl;
      }
    }
  }

  void ExtractBrief(const Image<uint8_t>& grey, 
      const Image<Vector2ida>& pts,
      const Image<float>& orientations,
      uint32_t frame, 
      ManagedHostImage<Brief>& briefs) {
    briefs.Reinitialise(pts.w_, 1);
    for (size_t i=0; i<pts.Area(); ++i) {
      briefs[i].pt_ = pts[i];
      briefs[i].lvl_= 0;
      briefs[i].frame_ = frame;
      briefs[i].orientation_= orientations[i];
      if(!tdp::ExtractBrief(grey, briefs[i])) {
        std::cout << pts[i].transpose() << " could not be extracted" << std::endl;
      }
    }
  }


  bool ExtractOrb(const Image<uint8_t>& patch, Vector8uda& desc, 
      int orientation) {
    switch (orientation) {
    case 0:
      ExtractOrb0(patch, desc);
      return true;
    case 1:
      ExtractOrb1(patch, desc);
      return true;
    case 2:
      ExtractOrb2(patch, desc);
      return true;
    case 3:
      ExtractOrb3(patch, desc);
      return true;
    case 4:
      ExtractOrb4(patch, desc);
      return true;
    case 5:
      ExtractOrb5(patch, desc);
      return true;
    case 6:
      ExtractOrb6(patch, desc);
      return true;
    case 7:
      ExtractOrb7(patch, desc);
      return true;
    case 8:
      ExtractOrb8(patch, desc);
      return true;
    case 9:
      ExtractOrb9(patch, desc);
      return true;
    case 10:
      ExtractOrb10(patch, desc);
      return true;
    case 11:
      ExtractOrb11(patch, desc);
      return true;
    case 12:
      ExtractOrb12(patch, desc);
      return true;
    case 13:
      ExtractOrb13(patch, desc);
      return true;
    case 14:
      ExtractOrb14(patch, desc);
      return true;
    case 15:
      ExtractOrb15(patch, desc);
      return true;
    case 16:
      ExtractOrb16(patch, desc);
      return true;
    case 17:
      ExtractOrb17(patch, desc);
      return true;
    case 18:
      ExtractOrb18(patch, desc);
      return true;
    case 19:
      ExtractOrb19(patch, desc);
      return true;
    case 20:
      ExtractOrb20(patch, desc);
      return true;
    case 21:
      ExtractOrb21(patch, desc);
      return true;
    case 22:
      ExtractOrb22(patch, desc);
      return true;
    case 23:
      ExtractOrb23(patch, desc);
      return true;
    case 24:
      ExtractOrb24(patch, desc);
      return true;
    case 25:
      ExtractOrb25(patch, desc);
      return true;
    case 26:
      ExtractOrb26(patch, desc);
      return true;
    case 27:
      ExtractOrb27(patch, desc);
      return true;
    case 28:
      ExtractOrb28(patch, desc);
      return true;
    case 29:
      ExtractOrb29(patch, desc);
      return true;
    }
    return false;
  }

  bool ExtractBrief(const Image<uint8_t>& patch, Vector8uda& desc, 
      int orientation) {
    switch (orientation) {
      case 0:
        ExtractBrief0(patch, desc);
        return true;
      case 1:
        ExtractBrief1(patch, desc);
        return true;
      case 2:
        ExtractBrief2(patch, desc);
        return true;
      case 3:
        ExtractBrief3(patch, desc);
        return true;
      case 4:
        ExtractBrief4(patch, desc);
        return true;
      case 5:
        ExtractBrief5(patch, desc);
        return true;
      case 6:
        ExtractBrief6(patch, desc);
        return true;
      case 7:
        ExtractBrief7(patch, desc);
        return true;
      case 8:
        ExtractBrief8(patch, desc);
        return true;
      case 9:
        ExtractBrief9(patch, desc);
        return true;
      case 10:
        ExtractBrief10(patch, desc);
        return true;
      case 11:
        ExtractBrief11(patch, desc);
        return true;
      case 12:
        ExtractBrief12(patch, desc);
        return true;
      case 13:
        ExtractBrief13(patch, desc);
        return true;
      case 14:
        ExtractBrief14(patch, desc);
        return true;
      case 15:
        ExtractBrief15(patch, desc);
        return true;
      case 16:
        ExtractBrief16(patch, desc);
        return true;
      case 17:
        ExtractBrief17(patch, desc);
        return true;
      case 18:
        ExtractBrief18(patch, desc);
        return true;
      case 19:
        ExtractBrief19(patch, desc);
        return true;
      case 20:
        ExtractBrief20(patch, desc);
        return true;
      case 21:
        ExtractBrief21(patch, desc);
        return true;
      case 22:
        ExtractBrief22(patch, desc);
        return true;
      case 23:
        ExtractBrief23(patch, desc);
        return true;
      case 24:
        ExtractBrief24(patch, desc);
        return true;
      case 25:
        ExtractBrief25(patch, desc);
        return true;
      case 26:
        ExtractBrief26(patch, desc);
        return true;
      case 27:
        ExtractBrief27(patch, desc);
        return true;
      case 28:
        ExtractBrief28(patch, desc);
        return true;
      case 29:
        ExtractBrief29(patch, desc);
        return true;
    }
    return false;
  }

}
