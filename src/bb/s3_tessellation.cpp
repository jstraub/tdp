/* Copyright (c) 2016, Julian Straub <jstraub@csail.mit.edu> Licensed
 * under the MIT license. See the license file LICENSE.
 */
#include <tdp/bb/s3_tessellation.h> 

namespace tdp {

std::vector<Tetrahedron4D> TessellateS3() {
  // Search for a north pole to split the sphere into two halfs of
  // equal size.  The standard north 1,0,0,0 dooes not work;
  S4f north;
  for (uint32_t i=0; i<1000; ++i) {
    north = S4f::Random();
    std::vector<Tetrahedron4D> tetrahedra = TessellateS3(north.vector());
    if ( tetrahedra.size() == 300)  {
      break;
    }
  }
  std::cout << "Found north for division of the sphere into two equal halfs" 
      << std::endl << north << std::endl;
  std::cout << "theta " << 2.*acos(north.vector()(0))*180./M_PI
    << std::endl << "axis= " 
    << north.vector().bottomRows(3).transpose()/sin(acos(north.vector()(0))) 
    << std::endl;
  return TessellateS3(north.vector());
}

std::vector<Tetrahedron4D> TessellateS3(const Eigen::Vector4f& north) {
  std::vector<Tetrahedron4D> tetrahedra;
  tetrahedra.reserve(600);
  
  Eigen::Matrix<float, 4, 120> vertices;
  vertices.fill(0.);
  uint32_t i = 0;
  for (float a=0; a<2; ++a) 
    for (float b=0; b<2; ++b) 
      for (float c=0; c<2; ++c) 
        for (float d=0; d<2; ++d) {
          vertices(0,i) = a*2. - 1.;
          vertices(1,i) = b*2. - 1.;
          vertices(2,i) = c*2. - 1.;
          vertices(3,i++) = d*2. - 1.;
        }
  for (uint32_t j=0; j<4; ++j) {
    vertices(j,i++) = 2.;
    vertices(j,i++) = -2.;
  }
  // Golden Ratio 
  float phi = 0.5 * (1. + sqrt(5.));
  // All even permutations
  // http://mathworld.wolfram.com/EvenPermutation.html
  Eigen::Matrix<uint32_t, 12, 4> evenPerms;
  evenPerms << 
    0,1,2,3,
    0,2,3,1,
    0,3,1,2,
    1,0,3,2,
    1,2,0,3,
    1,3,2,0,
    2,0,1,3,
    2,1,3,0,
    2,3,0,1,
    3,0,2,1,
    3,1,0,2,
    3,2,1,0;
  for (uint32_t j=0; j<12; ++j) 
    for (float a=0; a<2; ++a) 
      for (float b=0; b<2; ++b) 
        for (float c=0; c<2; ++c) {
          vertices(evenPerms(j,0),i) = (a*2.-1.)*phi;
          vertices(evenPerms(j,1),i) = (b*2.-1.);
          vertices(evenPerms(j,2),i) = (c*2.-1.)/phi;
          vertices(evenPerms(j,3),i++) = 0.;
        }
  vertices *= 0.5;
  assert(i == 120);
  // Filter out half of the sphere.
  uint32_t j = 0;
  for (i = 0; i < 120; ++i) {
    // this does a bit less than half the sphere (90)
//    if (acos(north.transpose() * vertices.col(i)) <= 90.*M_PI/180.){
    // This does a bit more than half the sphere (108)
    if (acos(north.transpose() * vertices.col(i)) <= 105.*M_PI/180.){
//    if (acos(north.transpose() * vertices.col(i)) <= 120.*M_PI/180.){
      vertices.col(j++) = vertices.col(i); 
//    } else {
    }
//      std::cout << acos(north.transpose() * vertices.col(i))*180./M_PI << std::endl;
  }

  uint32_t n_vertices = j;
//  std::cout << "Have " << n_vertices << " filtered vertices." << std::endl;
  // Precompute all pairwise tetrahedron edges based on the known
  // angular distance btween any two: 72 deg.
  Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> G(n_vertices, n_vertices);
  for (i = 0; i < n_vertices; ++i) 
    for (j = 0; j < n_vertices; ++j) {
      float ang = acos(vertices.col(i).transpose() * vertices.col(j)); 
      if (fabs(ang - 36.*M_PI/180.) < 1e-6) {
        G(i,j) = 1;
      } else {
        G(i,j) = 0;
      }
    }
//  std::cout << G.cast<uint32_t>().sum() << std::endl;

  // Compute all combinations "n_vertices choose 4"
  Combinations combinations(n_vertices, 4);
  // Find tetrahedra by looking at all possible vertex combinations and
  // selecting only the ones which have edges of the correct length
  // between them.
//  std::cout << "Found " << combinations.Get().size() 
//    << " combinations." << std::endl;
  for (i = 0; i < combinations.Get().size(); ++i) {
    Eigen::Map<const Eigen::Matrix<uint32_t, 1, 4>> p(&(combinations.Get()[i][0]));
    //std::cout << "checking combination: " << p << std::endl;
    if (G(p(0),p(1))==1 && G(p(0),p(2))==1 && G(p(0),p(3))==1 
        && G(p(1),p(2))==1 && G(p(1),p(3))==1 && G(p(2),p(3))==1) {
//      std::cout << " found tetrahedron" << std::endl;
      tetrahedra.push_back(Tetrahedron4D(
            vertices.col(p(0)),
            vertices.col(p(1)),
            vertices.col(p(2)),
            vertices.col(p(3))));
    }
  }
  return tetrahedra;
}

void TessellationTest(std::vector<Tetrahedron4D>& tetrahedra, uint32_t Nsamples) {
  uint32_t N = 0;
  for (uint32_t i=0; i<Nsamples; ++i) {
    S4f q = S4f::Random();
    for (const auto& tetra : tetrahedra) 
      if (tetra.Intersects(q.vector())) {
        ++N;
        break;
      }
  }
  std::cout << "fraction all over sphere intersected with tessellation: "
    << static_cast<float>(N)/static_cast<float>(Nsamples)
    << std::endl;
  N = 0.;
  for (uint32_t i=0; i<Nsamples; ++i) {
    S4f q = S4f::Random();
    q.vector()(0) = q.vector()(0) < 0. ? -q.vector()(0) : q.vector()(0);
    for (const auto& tetra : tetrahedra) 
      if (tetra.Intersects(q.vector())) {
        ++N;
        break;
      }
  }
  std::cout << "fraction on top half-sphere intersected with tessellation: "
    << static_cast<float>(N)/static_cast<float>(Nsamples)
    << std::endl;
  N = 0.;
  for (uint32_t i=0; i<Nsamples; ++i) {
    S4f q = S4f::Random();
    q.vector()(0) = q.vector()(0) < 0. ? q.vector()(0) : -q.vector()(0);
    for (const auto& tetra : tetrahedra) 
      if (tetra.Intersects(q.vector())) {
        ++N;
        break;
      }
  }
  std::cout << "fraction on bottom half-sphere intersected with tessellation: "
    << static_cast<float>(N)/static_cast<float>(Nsamples)
    << std::endl;
}
}
