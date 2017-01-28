#include <iostream>

// g++ -Wall -std=c++11 testCCTP.cpp -o testCCTP 

template<typename T, int D>
struct Vector {
  int dim() const { return D; };
  T data[D];
};

template <typename Derived> class CameraBase;

template <template<typename T, int D> class Derived, typename T, int D>
class CameraBase<Derived<T,D>> {
  public:
  Vector<T,D> param;
};

template <class T, int D=4>
class Camera : public CameraBase<Camera<T,D>> {
  public:
    std::string TypeName() { return std::string("Pinhole"); }
};

template <class T, int D=7>
class CameraPoly3 : public CameraBase<Camera<T,D>> {
  public:
    std::string TypeName() { return std::string("Poly3"); }
};

template<typename Derived>
void PrintCam(const CameraBase<Derived>& cam) {
  std::cout << cam.param.dim() << std::endl;
}


int main() {
  Camera<float> cam;
  PrintCam(cam);
  std::cout << sizeof(cam)/4 << " should be " << 4 << std::endl;
  std::cout << cam.TypeName() << std::endl;
  CameraPoly3<float> camPoly;
  std::cout << sizeof(camPoly)/4 << " should be " << 7 << std::endl;
  std::cout << camPoly.TypeName() << std::endl;
}
