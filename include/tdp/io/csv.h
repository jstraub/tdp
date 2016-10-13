#pragma once

#include <tdp/data/image.h>
#include <tdp/data/managed_image.h>
#include <tdp/eigen/dense.h>

namespace tdp {

void LoadPointCloudCsv(
    const std::string& path,
    ManagedHostImage<Vector3fda>& verts);

void SavePointCloudCsv(
    const std::string& path,
    const Image<Vector3fda>& verts,
    const Image<Vector3fda>& ns,
    std::vector<std::string>& comments);

}
