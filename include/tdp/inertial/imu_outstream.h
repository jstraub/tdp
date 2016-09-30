
#pragma once
#include <string>
#include <tdp/inertial/imu_obs.h>
#include <pangolin/utils/picojson.h>
#include <pangolin/utils/uri.h>
#include <pangolin/log/packetstream.h>

namespace tdp {

class ImuOutStream {
 public:
  ImuOutStream(const std::string& path, size_t buffer_size_bytes = 1000);
  ~ImuOutStream();

  bool IsOpen() const;
  void Open(const std::string& uri, const pangolin::json::value& properties);
  void Close();

  bool IsPipe() { return is_pipe; }

  int WriteStream(const ImuObs& data, 
      const pangolin::json::value& frame_properties = pangolin::json::value());

 private:
  std::string input_uri;
  const std::string filename;
  pangolin::json::value device_properties;
  pangolin::PacketStreamWriter packetstream;
  size_t packetstream_buffer_size_bytes;
  int packetstreamsrcid;
  bool first_frame;
  size_t total_frame_size;
  bool is_pipe;
};


}
