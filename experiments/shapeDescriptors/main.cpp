/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <iostream>
#include <cmath>
#include <complex>
#include <vector>
#include <cstdlib>
#include <random>


#include <pangolin/pangolin.h>
#include <pangolin/video/video_record_repeat.h>
#include <pangolin/gl/gltexturecache.h>
#include <pangolin/gl/glpixformat.h>
#include <pangolin/handler/handler_image.h>
#include <pangolin/utils/file_utils.h>
#include <pangolin/utils/timer.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/glsl.h>
#include <pangolin/gl/glvbo.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image_io.h>

#include <tdp/eigen/dense.h>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>

#include <tdp/preproc/depth.h>
#include <tdp/preproc/pc.h>
#include <tdp/camera/camera.h>
#ifdef CUDA_FOUND
#include <tdp/preproc/normals.h>
#endif

#include <tdp/io/tinyply.h>
#include <tdp/gl/shaders.h>
#include <tdp/gl/gl_draw.h>

#include <tdp/gui/gui.hpp>
#include <tdp/gui/quickView.h>

#include <tdp/nn/ann.h>
#include <tdp/manifold/S.h>
#include <tdp/manifold/SE3.h>
#include <tdp/data/managed_image.h>

#include <tdp/utils/status.h>
#include <tdp/utils/timer.hpp>
#include <tdp/eigen/std_vector.h>


#include <tdp/laplace_beltrami/laplace_beltrami.h>

int main( int argc, char* argv[] )
{
    std::string input;
    int nSamples = 100;
    tdp::ManagedHostImage<tdp::Vector3fda> pc_s(nSamples,1), pc_t(nSamples,1), pc_all;

    if(argc > 1){
        //if only one path given, p_t will be copied from p_s
        input = std::string(argv[1]);
        tdp::LoadPointCloudFromMesh(input, pc_all);
        std::cout << "input pc_s: " << input << std::endl;
        tdp::GetSamples(pc_all, pc_s, nSamples);

        if (argc >2){
            input = std::string(argv[2]);
            tdp::LoadPointCloudFromMesh(input, pc_all);
            tdp::GetSamples(pc_all, pc_t, nSamples);
            std::cout << "input pc_t: " << input << std::endl;
        } else{
        pc_t.ResizeCopyFrom(pc_s);
        }

    } else {
        std::srand(101);
        GetSphericalPc(pc_s, nSamples);
        std::srand(101);
        GetSphericalPc(pc_t, nSamples);
    }
    std::cout << "pc_s: " << pc_s.Area()<< std::endl;
    std::cout << "pc_t: " << pc_t.Area()<< std::endl;

}
