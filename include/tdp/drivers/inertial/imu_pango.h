#pragma once
#include <pangolin/log/packetstream.h>
#include <tdp/inertial/imu_interface.h>

namespace tdp {

class ImuPango : public ImuInterface {

 public:
  ImuPango(const std::string& filename, bool realtime);
  virtual ~ImuPango();

  virtual bool GrabNext(ImuObs& obs);
  virtual bool GrabNewest(ImuObs& obs);

  virtual void Start();
  virtual void Stop();

  virtual pangolin::json::value GetProperties() const;
  virtual pangolin::json::value GetFrameProperties() const;
 private:

  void HandlePipeClosed();

  pangolin::PacketStreamReader reader;
  size_t size_bytes;
  pangolin::json::value device_properties;
  pangolin::json::value frame_properties;
  int src_id;
  const std::string filename;
  bool realtime;
  bool is_pipe;
  bool is_pipe_open;
  int pipe_fd;
};

}
