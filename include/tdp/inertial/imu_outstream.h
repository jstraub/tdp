
#pragma once
#include <pangolin/log/packetstream.h>
#include <pangolin/utils/picojson.h>
#include <pangolin/utils/uri.h>
#include <tdp/inertial/imu_obs.h>
#include <string>

namespace tdp {

class ImuOutStream {
 public:
  ImuOutStream(const std::string& path, size_t buffer_size_bytes = 1000);
  ~ImuOutStream();

  bool IsOpen() const;
  void Open(const std::string& uri, const picojson::value& properties);
  void Close();

  bool IsPipe() { return is_pipe; }

  int WriteStream(const ImuObs& data,
                  const picojson::value& frame_properties = picojson::value());

 private:
  std::string input_uri;
  const std::string filename;
  picojson::value device_properties;
  pangolin::PacketStreamWriter packetstream;
  size_t packetstream_buffer_size_bytes;
  int packetstreamsrcid;
  bool first_frame;
  size_t total_frame_size;
  bool is_pipe;
};
}
