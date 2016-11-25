// This software is in the public domain. Where that dedication is not
// recognized, you are granted a perpetual, irrevocable license to copy,
// distribute, and modify this file as you see fit.
// Authored in 2015 by Dimitri Diakopoulos (http://www.dimitridiakopoulos.com)
// https://github.com/ddiakopoulos/tinyply

#include <cstring>
#include <tdp/io/tinyply.h>
#include <tdp/cuda/cuda.h>

namespace tdp {

void LoadPointCloud(
    const std::string& path,
    ManagedHostImage<Vector3fda>& verts) {

  std::vector<float> vertices;
  std::ifstream in(path, std::ios::binary);
  tinyply::PlyFile ply(in);

  for (auto e : ply.get_elements()) {
    std::cout << "element - " << e.name << " (" << e.size << ")" 
      << std::endl;
    for (auto p : e.properties) {
      std::cout << "\tproperty - " << p.name << " (" 
        << tinyply::PropertyTable[p.propertyType].str << ")" << std::endl;
    }
  }
  std::cout << std::endl;
  ply.request_properties_from_element("vertex", {"x", "y", "z"}, vertices);
  ply.read(in);
  std::cout << "loaded ply file: " << vertices.size() << std::endl;

  verts.Reinitialise(vertices.size()/3,1);
  std::memcpy(verts.ptr_, &vertices[0], verts.SizeBytes());
}

void LoadPointCloud(
    const std::string& path,
    ManagedHostImage<Vector3fda>& verts,
    ManagedHostImage<Vector3fda>& ns, bool verbose) {

  std::vector<float> vertices;
  std::vector<float> normals;
  std::ifstream in(path, std::ios::binary);
  tinyply::PlyFile ply(in);

  if (verbose) {
    for (auto e : ply.get_elements()) {
      std::cout << "element - " << e.name << " (" << e.size << ")" 
        << std::endl;
      for (auto p : e.properties) {
        std::cout << "\tproperty - " << p.name << " (" 
          << tinyply::PropertyTable[p.propertyType].str << ")" << std::endl;
      }
    }
    std::cout << std::endl;
  }
  ply.request_properties_from_element("vertex", {"x", "y", "z"}, vertices);
  ply.request_properties_from_element("vertex", {"nx", "ny", "nz"}, normals);
  ply.read(in);
  std::cout << "loaded ply file: " << vertices.size()
    << " normals: " << normals.size() << std::endl;

  verts.Reinitialise(vertices.size()/3,1);
  ns.Reinitialise(normals.size()/3,1);
  for (size_t i=0; i<vertices.size(); ++i) {
    verts[i/3](i%3) = vertices[i];
    ns[i/3](i%3) = normals[i];
//    if (i%3==2 &&  i < 10000) {
//      std::cout << verts[i/3].transpose()
//        << ", "<< ns[i/3].transpose() << std::endl;
//    }
  }
//  std::memcpy(verts.ptr_, &vertices[0], verts.SizeBytes());
//  std::memcpy(ns.ptr_, &normals[0], ns.SizeBytes());
}

void LoadMesh(
    const std::string& path,
    ManagedHostImage<Vector3fda>& verts,
    ManagedHostImage<Vector3uda>& tris) {

  std::vector<float> vertices;
  std::vector<uint32_t> triangles;
  std::ifstream in(path);
  tinyply::PlyFile ply(in);

  for (auto e : ply.get_elements()) {
    std::cout << "element - " << e.name << " (" << e.size << ")" 
      << std::endl;
    for (auto p : e.properties) {
      std::cout << "\tproperty - " << p.name << " (" 
        << tinyply::PropertyTable[p.propertyType].str << ")" << std::endl;
    }
  }
  std::cout << std::endl;
  ply.request_properties_from_element("vertex", {"x", "y", "z"}, vertices);
  ply.request_properties_from_element("face", {"vertex_indices"}, triangles);
  ply.read(in);
  std::cout << "loaded ply file: " << vertices.size() << " " << triangles.size() << std::endl;

  verts.Reinitialise(vertices.size()/3,1);
  tris.Reinitialise(triangles.size()/3,1);
  std::memcpy(verts.ptr_, &vertices[0], verts.SizeBytes());
  std::memcpy(tris.ptr_, &triangles[0], tris.SizeBytes());
}

void SavePointCloud(
    const std::string& path,
    const Image<Vector3fda>& pc,
    const Image<Vector3fda>& n,
    bool binary,
    std::vector<std::string> comments) {
  std::vector<float> verts;
  std::vector<float> norms;
  verts.reserve(pc.Area()*3);
  norms.reserve(n.Area()*3);
  // filter out NAN points.
  for (size_t i=0; i<pc.Area(); ++i)  {
    if (IsValidData(pc[i]) && IsValidNormal(n[i])) {
      verts.push_back(pc[i](0));
      verts.push_back(pc[i](1));
      verts.push_back(pc[i](2));
      norms.push_back(n[i](0));
      norms.push_back(n[i](1));
      norms.push_back(n[i](2));
    }
  }
  tinyply::PlyFile plyFile;
  plyFile.add_properties_to_element("vertex", {"x", "y", "z"}, verts);
  plyFile.add_properties_to_element("vertex", {"nx", "ny", "nz"}, norms);
  for (auto& comment : comments) 
    plyFile.comments.push_back(comment);
  std::ostringstream outStream;
  plyFile.write(outStream, binary);
  std::ofstream out(path);
  out << outStream.str();
  out.close();
}

void SavePointCloud(
    const std::string& path,
    const Image<Vector3fda>& pc,
    const Image<Vector3fda>& n,
    const Image<Vector3bda>& rgb,
    bool binary,
    std::vector<std::string> comments) {
  std::vector<float> verts;
  std::vector<float> norms;
  std::vector<uint8_t> rgbs;
  verts.reserve(pc.Area()*3);
  norms.reserve(n.Area()*3);
  rgbs.reserve(rgb.Area()*3);
  // filter out NAN points.
  for (size_t i=0; i<pc.Area(); ++i)  {
    if (IsValidData(pc[i]) && IsValidNormal(n[i])) {
      verts.push_back(pc[i](0));
      verts.push_back(pc[i](1));
      verts.push_back(pc[i](2));
      norms.push_back(n[i](0));
      norms.push_back(n[i](1));
      norms.push_back(n[i](2));
      rgbs.push_back(rgb[i](0));
      rgbs.push_back(rgb[i](1));
      rgbs.push_back(rgb[i](2));
    }
  }
  tinyply::PlyFile plyFile;
  plyFile.add_properties_to_element("vertex", {"x", "y", "z"}, verts);
  plyFile.add_properties_to_element("vertex", {"nx", "ny", "nz"}, norms);
  plyFile.add_properties_to_element("vertex", {"red", "green", "blue"}, rgbs);
  for (auto& comment : comments) 
    plyFile.comments.push_back(comment);
  std::ostringstream outStream;
  plyFile.write(outStream, binary);
  std::ofstream out(path);
  out << outStream.str();
  out.close();
}

}

using namespace tinyply;
using namespace std;

//////////////////
// PLY Property //
//////////////////

PlyProperty::PlyProperty(std::istream & is) : isList(false)
{
    parse_internal(is);
}

void PlyProperty::parse_internal(std::istream & is)
{
    string type;
    is >> type;
    if (type == "list")
    {
        string countType;
        is >> countType >> type;
        listType = property_type_from_string(countType);
        isList = true;
    }
    propertyType = property_type_from_string(type);
    is >> name;
}

/////////////////
// PLY Element //
/////////////////

PlyElement::PlyElement(std::istream & is)
{
    parse_internal(is);
}

void PlyElement::parse_internal(std::istream & is)
{
    is >> name >> size;
}

//////////////
// PLY File //
//////////////

PlyFile::PlyFile(std::istream & is)
{
    if (!parse_header(is))
    {
        throw std::runtime_error("file is not ply or encounted junk in header");
    }
}

bool PlyFile::parse_header(std::istream& is)
{
    std::string line;
//    bool gotMagic = false;
    while (std::getline(is, line))
    {
        std::istringstream ls(line);
        std::string token;
        ls >> token;
        if (token == "ply" || token == "PLY" || token == "")
        {
//            gotMagic = true;
            continue;
        }
        else if (token == "comment")    read_header_text(line, ls, comments, 7);
        else if (token == "format")     read_header_format(ls);
        else if (token == "element")    read_header_element(ls);
        else if (token == "property")   read_header_property(ls);
        else if (token == "obj_info")   read_header_text(line, ls, objInfo, 7);
        else if (token == "end_header") break;
        else return false;
    }
    return true;
}

void PlyFile::read_header_text(std::string line, std::istream & is, std::vector<std::string> place, int erase)
{
    place.push_back((erase > 0) ? line.erase(0, erase) : line);
}

void PlyFile::read_header_format(std::istream & is)
{
    std::string s;
    (is >> s);
    if (s == "binary_little_endian")
        isBinary = true;
    else if (s == "binary_big_endian")
        throw std::runtime_error("big endian formats are not supported!");
}

void PlyFile::read_header_element(std::istream & is)
{
    get_elements().emplace_back(is);
}

void PlyFile::read_header_property(std::istream & is)
{
    get_elements().back().properties.emplace_back(is);
}

uint32_t PlyFile::skip_property_binary(const PlyProperty & property, std::istream & is)
{
    static std::vector<char> skip(PropertyTable[property.propertyType].stride);
    if (property.isList)
    {
        uint32_t listSize = 0;
        uint32_t dummyCount = 0;
        read_property_binary(property.listType, &listSize, dummyCount, is);
        for (uint32_t i = 0; i < listSize; ++i) is.read(skip.data(), PropertyTable[property.propertyType].stride);
        return listSize;
    }
    else
    {
        is.read(skip.data(), PropertyTable[property.propertyType].stride);
        return 0;
    }
}

void PlyFile::skip_property_ascii(const PlyProperty & property, std::istream & is)
{
    std::string skip;
    if (property.isList)
    {
        int listSize;
        is >> listSize;
        for (int i = 0; i < listSize; ++i) is >> skip;
    }
    else is >> skip;
}

void PlyFile::read_property_binary(PlyProperty::Type t, void * dest, uint32_t & destOffset, std::istream & is)
{
    static std::vector<char> src(PropertyTable[t].stride);
    is.read(src.data(), PropertyTable[t].stride);
    switch (t)
    {
        case PlyProperty::Type::INT8:       ply_cast<int8_t>(dest, src.data());    break;
        case PlyProperty::Type::UINT8:      ply_cast<uint8_t>(dest, src.data());   break;
        case PlyProperty::Type::INT16:      ply_cast<int16_t>(dest, src.data());   break;
        case PlyProperty::Type::UINT16:     ply_cast<uint16_t>(dest, src.data());  break;
        case PlyProperty::Type::INT32:      ply_cast<int32_t>(dest, src.data());   break;
        case PlyProperty::Type::UINT32:     ply_cast<uint32_t>(dest, src.data());  break;
        case PlyProperty::Type::FLOAT32:    ply_cast<float>(dest, src.data());     break;
        case PlyProperty::Type::FLOAT64:    ply_cast<double>(dest, src.data());    break;
        case PlyProperty::Type::INVALID:    throw std::invalid_argument("invalid ply property");
    }
    destOffset += PropertyTable[t].stride;
}

void PlyFile::read_property_ascii(PlyProperty::Type t, void * dest, uint32_t & destOffset, std::istream & is)
{
    switch (t)
    {
        case PlyProperty::Type::INT8:       *((int8_t *)dest) = ply_read_ascii<int32_t>(is);        break;
        case PlyProperty::Type::UINT8:      *((uint8_t *)dest) = ply_read_ascii<uint32_t>(is);      break;
        case PlyProperty::Type::INT16:      ply_cast_ascii<int16_t>(dest, is);                      break;
        case PlyProperty::Type::UINT16:     ply_cast_ascii<uint16_t>(dest, is);                     break;
        case PlyProperty::Type::INT32:      ply_cast_ascii<int32_t>(dest, is);                      break;
        case PlyProperty::Type::UINT32:     ply_cast_ascii<uint32_t>(dest, is);                     break;
        case PlyProperty::Type::FLOAT32:    ply_cast_ascii<float>(dest, is);                        break;
        case PlyProperty::Type::FLOAT64:    ply_cast_ascii<double>(dest, is);                       break;
        case PlyProperty::Type::INVALID:    throw std::invalid_argument("invalid ply property");
    }
    destOffset += PropertyTable[t].stride;
}

void PlyFile::write_property_ascii(PlyProperty::Type t, std::ostringstream & os, uint8_t * src, uint32_t & srcOffset)
{
    switch (t)
    {
        case PlyProperty::Type::INT8:       os << static_cast<int32_t>(*reinterpret_cast<int8_t*>(src));    break;
        case PlyProperty::Type::UINT8:      os << static_cast<uint32_t>(*reinterpret_cast<uint8_t*>(src));  break;
        case PlyProperty::Type::INT16:      os << *reinterpret_cast<int16_t*>(src);     break;
        case PlyProperty::Type::UINT16:     os << *reinterpret_cast<uint16_t*>(src);    break;
        case PlyProperty::Type::INT32:      os << *reinterpret_cast<int32_t*>(src);     break;
        case PlyProperty::Type::UINT32:     os << *reinterpret_cast<uint32_t*>(src);    break;
        case PlyProperty::Type::FLOAT32:    os << *reinterpret_cast<float*>(src);       break;
        case PlyProperty::Type::FLOAT64:    os << *reinterpret_cast<double*>(src);      break;
        case PlyProperty::Type::INVALID:    throw std::invalid_argument("invalid ply property");
    }
    os << " ";
    srcOffset += PropertyTable[t].stride;
}

void PlyFile::write_property_binary(PlyProperty::Type t, std::ostringstream & os, uint8_t * src, uint32_t & srcOffset)
{
    os.write(reinterpret_cast<const char *>(src), PropertyTable[t].stride);
    srcOffset += PropertyTable[t].stride;
}

void PlyFile::read(std::istream & is)
{
    read_internal(is);
}

void PlyFile::write(std::ostringstream & os, bool isBinary)
{
    if (isBinary) write_binary_internal(os);
    else write_ascii_internal(os);
}

void PlyFile::write_binary_internal(std::ostringstream & os)
{
    isBinary = true;
    write_header(os);
    
    for (auto & e : elements)
    {
        for (int i = 0; i < e.size; ++i)
        {
            for (auto & p : e.properties)
            {
                auto & cursor = userDataTable[make_key(e.name, p.name)];
                if (p.isList)
                {
                    uint8_t listSize[4] = {0, 0, 0, 0};
                    memcpy(listSize, &p.listCount, sizeof(uint32_t));
                    uint32_t dummyCount = 0;
                    write_property_binary(p.listType, os, listSize, dummyCount);
                    for (int j = 0; j < p.listCount; ++j)
                    {
                        write_property_binary(p.propertyType, os, (cursor->data + cursor->offset), cursor->offset);
                    }
                }
                else
                {
                    write_property_binary(p.propertyType, os, (cursor->data + cursor->offset), cursor->offset);
                }
            }
        }
    }
}

void PlyFile::write_ascii_internal(std::ostringstream & os)
{
    write_header(os);
    
    for (auto & e : elements)
    {
        for (int i = 0; i < e.size; ++i)
        {
            for (auto & p : e.properties)
            {
                auto & cursor = userDataTable[make_key(e.name, p.name)];
                if (p.isList)
                {
                    os << p.listCount << " ";
                    for (int j = 0; j < p.listCount; ++j)
                    {
                        write_property_ascii(p.propertyType, os, (cursor->data + cursor->offset), cursor->offset);
                    }
                }
                else
                {
                    write_property_ascii(p.propertyType, os, (cursor->data + cursor->offset), cursor->offset);
                }
            }
            os << std::endl;
        }
    }
}

void PlyFile::write_header(std::ostringstream & os)
{
    const std::locale & fixLoc = std::locale("C");
    os.imbue(fixLoc);
    
    os << "ply" << std::endl;
    if (isBinary)
        os << ((isBigEndian) ? "format binary_big_endian 1.0" : "format binary_little_endian 1.0") << std::endl;
    else
        os << "format ascii 1.0" << std::endl;
    
    for (const auto & comment : comments)
        os << "comment " << comment << std::endl;
    
    for (auto & e : elements)
    {
        os << "element " << e.name << " " << e.size << std::endl;
        for (const auto & p : e.properties)
        {
            if (p.isList)
            {
                os << "property list " << PropertyTable[p.listType].str << " "
                << PropertyTable[p.propertyType].str << " " << p.name << std::endl;
            }
            else
            {
                os << "property " << PropertyTable[p.propertyType].str << " " << p.name << std::endl;
            }
        }
    }
    os << "end_header" << std::endl;
}

void PlyFile::read_internal(std::istream & is)
{
    std::function<void(PlyProperty::Type t, void * dest, uint32_t & destOffset, std::istream & is)> read;
    std::function<void(const PlyProperty & property, std::istream & is)> skip;
    if (isBinary)
    {
        read = [&](PlyProperty::Type t, void * dest, uint32_t & destOffset, std::istream & is) { read_property_binary(t, dest, destOffset, is); };
        skip = [&](const PlyProperty & property, std::istream & is) { skip_property_binary(property, is); };
    }
    else
    {
        read = [&](PlyProperty::Type t, void * dest, uint32_t & destOffset, std::istream & is) { read_property_ascii(t, dest, destOffset, is); };
        skip = [&](const PlyProperty & property, std::istream & is) { skip_property_ascii(property, is); };
    }
    
    for (auto & element : get_elements())
    {
        if (std::find(requestedElements.begin(), requestedElements.end(), element.name) != requestedElements.end())
        {
            for (int64_t count = 0; count < element.size; ++count)
            {
                for (auto & property : element.properties)
                {
                    if (auto & cursor = userDataTable[make_key(element.name, property.name)])
                    {
                        if (property.isList)
                        {
                            uint32_t listSize = 0;
                            uint32_t dummyCount = 0;
                            read(property.listType, &listSize, dummyCount, is);
                            if (cursor->realloc == false)
                            {
                                cursor->realloc = true;
                                resize_vector(property.propertyType, cursor->vector, listSize * element.size, cursor->data);
                            }
                            for (uint32_t i = 0; i < listSize; ++i)
                            {
                                read(property.propertyType, (cursor->data + cursor->offset), cursor->offset, is);
                            }
                        }
                        else
                        {
                            read(property.propertyType, (cursor->data + cursor->offset), cursor->offset, is);
                        }
                    }
                    else
                    {
                        skip(property, is);
                    }
                }
            }
        }
        else continue;
    }
}
